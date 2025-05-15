from datasets import Dataset, load_dataset
import json
import nltk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    Trainer,
    TrainingArguments, BertForSequenceClassification,
)
from tqdm import tqdm
from fact_checking.load_base import search_wikipedia, search_wikipedia_new, search_wikipedia_model, search_offline


def tokenize(claim, evidence):
    return tokenizer(
        claim,
        str(evidence),
        truncation=True,
        padding="max_length",
        max_length=512,
    )

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(labels, preds), "f1_macro": f1_score(labels, preds, average="macro")}

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

fact_checking = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
)


def check_claim_bert(claim):
    evidence = search_offline(claim)
    print(type(claim))
    print(type(evidence))
    text = claim + "[SEP]" + evidence
    prediction = fact_checking(text)
    y_bert = label_map[prediction[0]["label"]]
    return y_bert

label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
"""
dataset = load_dataset("pietrolesci/nli_fever")
dataset = dataset["train"]

# label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2, "Not Enough Info": 2}


def filter_example(example):
    tokens = tokenizer(example["premise"], example["hypothesis"])["input_ids"]
    return len(tokens) <= 512


dataset = dataset.filter(filter_example)
print(dataset)
processed_data = []
# i = 0

for data in tqdm(dataset):
    processed_data.append(
        {
            "claim": data["premise"],
            "evidence": data["hypothesis"],
            "label": data["label"],
        }
    )
    #   i += 1
    # if i == 100:
    #   break
with open("processed_data.json", "w") as f:
    json.dump(processed_data, f)
"""
if __name__ == "__main__":

    with open("processed_data.json", "r") as f:
        processed_data = json.load(f)
    processed_data = processed_data[:50000]
    train_dataset = Dataset.from_list(processed_data)

    train_dataset = train_dataset.train_test_split(test_size=0.1)
    encoded_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["claim"],
            x["evidence"],
            truncation=True,
            # batched=True,
            padding="max_length",
            max_length=512,
        )
    )

    args = TrainingArguments(
        output_dir="./fact_checking_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained("./bert_model/")
    tokenizer.save_pretrained("./bert_model/")
