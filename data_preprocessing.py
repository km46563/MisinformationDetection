from nltk.stem.snowball import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import unicodedata

nltk.download("punkt_tab")
nltk.download("stopwords")

real_news = pd.read_csv("News_dataset/True.csv", index_col=None)[["text"]]
fake_news = pd.read_csv("News_dataset/Fake.csv", index_col=None)[["text"]]

all_news = pd.concat([real_news.assign(label=True), fake_news.assign(label=False)])


# Normalizacja tekstu
def normalize_text(text):
    print("checkpoint1")
    new_text = []
    for sentence in tqdm(text):
        normalized_sentence = (
            unicodedata.normalize("NFD", sentence)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        normalized_sentence = normalized_sentence.lower()
        normalized_sentence = re.sub(r"[^a-zA-Z0-9\s]", "", normalized_sentence)
        normalized_sentence = re.sub(r"\s+", " ", normalized_sentence).strip()
        new_text.append(normalized_sentence)
    return new_text


# Stemowanie tekstu
def stem_text(text):
    print("checkpoint2")
    porter = PorterStemmer()
    stemmed_texts = []
    for sentence in tqdm(text):
        words = word_tokenize(sentence)
        stemmed_words = [porter.stem(word) for word in words]
        stemmed_text = "".join(stemmed_words)
        stemmed_texts.append(stemmed_text)
    return stemmed_texts


def preprocess_text(text):
    preprocessed_text = []

    for sentence in tqdm(text):
        sentence = re.sub(r"[^\w\s]", "", sentence)
        preprocessed_text.append(
            " ".join(
                token.lower()
                for token in str(sentence).split()
                if token not in stopwords.words("english")
            )
        )

    return preprocessed_text


preprocessed_text = all_news["text"].values
# preprocessed_text = normalize_text(preprocessed_text)
###preprocessed_text = stem_text(preprocessed_text)
preprocessed_text = preprocess_text(preprocessed_text)
all_news["text"] = preprocessed_text

all_news.to_csv("preprocessed_data.csv", index=False)

from sklearn.feature_extraction.text import CountVectorizer


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


common_words = get_top_n_words(all_news["text"], 15)
df1 = pd.DataFrame(common_words, columns=["words", "count"])

df1.groupby("words").sum()["count"].sort_values(ascending=False).plot(
    kind="bar",
    figsize=(10, 6),
    xlabel="Top Words",
    ylabel="Count",
    title="Bar Chart of Top Words Frequency",
)
plt.show()
# plt.savefig("most_common_words.png")

"""
class_counts = all_news["label"].value_counts()
plt.bar(class_counts.index, class_counts.values)
plt.xlabel("Klasa")
plt.ylabel("Liczność")
plt.title("Liczność wystąpień fałszywych i prawdziwych informacji w zbiorze")
plt.savefig("distofclasses.png")
"""
