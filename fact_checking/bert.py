from datasets import Dataset, load_dataset
import json
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    Trainer,
    TrainingArguments,
)

from load_base import get_wikipedia


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


def tokenize(data):
    return tokenizer(
        data["claim"],
        data["evidence"],
        truncation=True,
        padding="max_length",
        max_length=123,
    )


dataset = load_dataset("fever", "v2.0")
dataset = dataset["validation"]
print(dataset)
label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2, "Not Enough Info": 2}


processed_data = []
# i = 0
for data in dataset:
    print(type(data))
    print(data)
    evidence_text = get_wikipedia(data.get("evidence_wiki_url"))

    sentences = sent_tokenize(evidence_text)
    tokenized_sent = [word_tokenize(sent.lower()) for sent in sentences]
    bm25 = BM25Okapi(tokenized_sent)
    tokenized_claim = word_tokenize(data["claim"].lower())

    scores = bm25.get_scores(tokenized_claim)
    best_idx = scores.argmax()
    evidence = sentences[best_idx]

    processed_data.append(
        {"claim": data["claim"], "evidence": evidence, "label": data["label"]}
    )
    #   i += 1
    # if i == 100:
    #   break
with open("processed_data_bm.json", "w") as f:
    json.dump(processed_data, f)

with open("processed_data_bm.json", "r") as f:
    processed_data = json.load(f)

for data in processed_data:
    print(data["label"])
    data["label"] = label_map[data["label"]]

train_dataset = Dataset.from_list(processed_data)

train_dataset = train_dataset.train_test_split(test_size=0.1)
encoded_dataset = train_dataset.map(
    lambda x: tokenizer(
        x["claim"],
        x["evidence"],
        truncation=True,
        # batched=True,
        padding="max_length",
        max_length=123,
    )
)

args = TrainingArguments(
    output_dir="./fact_checking_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
)

trainer.train()
model.save_pretrained("./fact_checking_model/")
tokenizer.save_pretrained("./fact_checking_model/")

fact_checking = pipeline(
    "text-classification", model="./fact_checking_model", tokenizer=tokenizer
)

claim = "climate change is a hoax."
evidence = (
    "97% of climate scientists agree that climate change is real and caused by humans."
)
result = fact_checking(claim + " " + evidence)
print(result)
