import json
import nltk
from sklearn.utils import shuffle

# nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import tqdm

from fact_checking.load_base import search_wikipedia, search_wikipedia_new


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, 0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        return self.fc(h_n.squeeze(0))


class FCDataset(Dataset):
    def __init__(self, data, max_length=512):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = encode(item["claim"] + " " + item["evidence"], self.max_length)
        y = item["label"]
        return torch.tensor(x), torch.tensor(y)


def encode(text, max_length=512):
    tokens = word_tokenize(text.lower())
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    return ids[:max_length] + [vocab["<PAD>"]] * max(0, max_length - len(ids))


def check_claim_rnn(claim):
    evidence = search_wikipedia_new(claim)
    text = encode(claim + " " + evidence)
    text = torch.tensor(text, dtype=torch.long).unsqueeze(0)
    text = text.to(device)

    with torch.no_grad():
        logits = model(text)
        probs = nn.functional.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        return predicted


processed_data = []
with open("processed_data.json", "r") as f:
    processed_data = json.load(f)

processed_data = processed_data[:10000]

all_text = [data["claim"] + " " + data["evidence"] for data in processed_data]
tokenized_text = [word_tokenize(text.lower()) for text in all_text]
vocab = {"<PAD>": 0, "<UNK>": 1}
for tokens in tokenized_text:
    for word in tokens:
        if word not in vocab:
            vocab[word] = len(vocab)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN(vocab_size=len(vocab)).to(device)
model.eval()

if __name__ == "__main__":
    dataset = FCDataset(processed_data)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in tqdm.tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, loss = {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "rnn_model.pt")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    print(f"Validation accuracy: {correct / total:.4f}")
