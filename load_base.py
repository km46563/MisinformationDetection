import datetime
import pickle

from dictionary.britannica import get_entries
import nltk
from nltk.app.wordnet_app import page_from_href
from spacy.lang.en import STOP_WORDS

nltk.download("punkt")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import wikipedia
import wikipediaapi

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="MisinformationDetection (contact: 283327@student.pwr.edu.pl)",
)
model = SentenceTransformer('all-MiniLM-L6-v2')

with open("database_evidences", "rb") as fp:
    evidences_db = pickle.load(fp)
with open("database_claims", "rb") as fp:
    claims_db = pickle.load(fp)
evidences_embeddings = model.encode(evidences_db)


def get_wikipedia(query):
    try:
        page = wikipedia.page(query)
        return page
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            return wikipedia.page(e.options[0])
        except:
            return None
    except wikipedia.exceptions.PageError:
        return None


def get_keywords(sentence):
    sent = nlp(sentence)
    keywords = set()
    for ent in sent.ents:
        if ent.label_ in {"PERSON", "NORP", "DATE", "LOC", "PRODUCT", "EVENT", "ORG", "GPE"}:
            lemma = " ".join([token.lemma_ for token in ent if not token.is_stop])
            if lemma and len(lemma) > 2:
                keywords.add(lemma.strip().lower())
    for chunk in sent.noun_chunks:
        chunk_text = chunk.text.strip().lower()
        if chunk_text not in STOP_WORDS and len(chunk_text) > 2:
            keywords.add(chunk_text)
    return list(keywords)


def search_wikipedia(claim):
    begin = datetime.datetime.now()
    #print("cl: ", claim)
    queries = get_keywords(claim)
    evidences = ""
    for query in queries:
        page = wiki.page(query)
        if page.exists():
            evidences += page.summary
    if evidences == "":
        return evidences
    sentences = sent_tokenize(evidences)
    tokenized_sent = [word_tokenize(sent.lower()) for sent in sentences]
    bm25 = BM25Okapi(tokenized_sent)
    tokenized_claim = word_tokenize(claim.lower())
    scores = bm25.get_scores(tokenized_claim)
    best_idx = scores.argmax()
    evidence = sentences[best_idx]
    #print("ev: ", evidence)
    end = datetime.datetime.now()
    time = end - begin
    #print(time)
    return evidence


def search_wikipedia_new(claim):
    begin = datetime.datetime.now()
    #print("cl: ", claim)
    queries = get_keywords(claim)
    best_evidence = ""
    best_score = -1
    tokenized_claim = word_tokenize(claim.lower())

    for query in queries:
        page = wiki.page(query)
        if not page.exists():
            continue
        text = page.text
        if not text or text.strip() == "":
            continue
        sentences = sent_tokenize(text)
        tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]
        if len(tokenized_sentences) == 0:
            continue
        bm25 = BM25Okapi(tokenized_sentences)
        scores = bm25.get_scores(tokenized_claim)
        max_idx = scores.argmax()
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_evidence = sentences[max_idx]

    #print("ev: ", best_evidence)
    end = datetime.datetime.now()
    time = end - begin
    #print(time)
    return best_evidence

def search_wikipedia_model(claim):
    queries = get_keywords(claim)
    claim_embedding = model.encode([claim])
    best_evidence = ""
    best_score = -1
    for query in queries:
        page = get_wikipedia(query)
        if not page:
            continue
        text = page.content
        if not text or text.strip() == "":
            continue
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            continue
        sentences_embedding = model.encode(sentences)
        similarities = cosine_similarity(claim_embedding, sentences_embedding)[0]
        max_idx = similarities.argmax()
        if similarities[max_idx] > best_score:
            best_score = similarities[max_idx]
            best_evidence = sentences[max_idx]
    return best_evidence

def search_offline(claim, treshold=0.5):
    claim_embedding = model.encode([claim])
    similarities = cosine_similarity(claim_embedding, evidences_embeddings)[0]
    max_idx = similarities.argmax()
    if similarities[max_idx] < treshold:
        evidence = search_wikipedia_model(claim)
    else:
        evidence = evidences_db[max_idx]
    return evidence
