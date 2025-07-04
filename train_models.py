# train_models.py

import os
import glob
import pickle
import ssl
import sys
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# PyTorch imports for neural networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# — SSL workaround so nltk.download can run in locked-down environments —
try:
    _create_unverified_https = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https
except AttributeError:
    pass

# — Ensure NLTK data is present —
for pkg, path in [
    ("punkt", "tokenizers/punkt"),
    ("stopwords", "corpora/stopwords")
]:
    try:
        nltk.data.find(path)
    except LookupError:
        print(f"NLTK resource '{pkg}' not found; downloading…")
        nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))


# Neural Network Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        tokens = text.split()[:self.max_length]
        indices = [self.vocab.get(token, 0) for token in tokens]  # 0 for unknown words
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)


# CNN Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
        
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()


# LSTM Model
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=2, dropout=0.5):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)
        # Use the last hidden state
        x = hidden[-1]  # Take the last layer's hidden state
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()


# RNN Model
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=2, dropout=0.5):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        # Use the last hidden state
        x = hidden[-1]  # Take the last layer's hidden state
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()


def preprocess_text(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)


def load_text_data(data_dir: str):
    texts, labels = [], []
    for label in ("ai", "human"):
        pattern = os.path.join(data_dir, label, "*.txt")
        for path in tqdm(glob.glob(pattern), desc=f"Loading {label}"):
            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                raw = fp.read()
            texts.append(preprocess_text(raw))
            labels.append(1 if label == "ai" else 0)
    return texts, labels


def train_classical(texts, labels, model_dir):
    print("\n▶ Training classical ML models…")
    # TF–IDF
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    X = tfidf.fit_transform(texts)
    y = labels

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifiers = {
        "svm": SVC(kernel="linear", probability=True),
        "decision_tree": DecisionTreeClassifier(),
        "adaboost": AdaBoostClassifier(n_estimators=50)
    }

    os.makedirs(model_dir, exist_ok=True)

    # save vectorizer
    with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    for name, clf in classifiers.items():
        print(f" • {name.upper()}…")
        clf.fit(X_train, y_train)
        with open(os.path.join(model_dir, f"{name}_model.pkl"), "wb") as f:
            pickle.dump(clf, f)

    print("✔ Classical models saved.")


def build_vocabulary(texts, min_freq=2):
    """Build vocabulary from texts"""
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Create vocabulary with words that appear at least min_freq times
    vocab = {"<UNK>": 0, "<PAD>": 1}  # Unknown and padding tokens
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    texts, labels = zip(*batch)
    
    # Pad sequences to the same length
    texts = pad_sequence(texts, batch_first=True, padding_value=1)  # 1 is <PAD> token
    labels = torch.stack(labels)
    
    return texts, labels


def train_neural_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Train a neural network model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def train_neural_networks(texts, labels, model_dir):
    """Train CNN, LSTM, and RNN models"""
    print("\n▶ Training neural network models…")
    
    # Build vocabulary
    vocab = build_vocabulary(texts)
    vocab_size = len(vocab)
    print(f" • Vocabulary size: {vocab_size}")
    
    # Create dataset
    dataset = TextDataset(texts, labels, vocab)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Define models
    models = {
        "CNN": TextCNN(vocab_size),
        "LSTM": TextLSTM(vocab_size),
        "RNN": TextRNN(vocab_size)
    }
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Save vocabulary
    with open(os.path.join(model_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    
    # Train each model
    for name, model in models.items():
        print(f"\n • Training {name}…")
        trained_model = train_neural_model(model, train_loader, val_loader, epochs=5)
        
        # Save model
        model_path = os.path.join(model_dir, f"{name}.pkl")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'vocab_size': vocab_size,
            'model_class': type(trained_model).__name__
        }, model_path)
        print(f"   ✔ {name} model saved to {model_path}")
    
    print("✔ Neural network models saved.")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data", "training_data")
    model_dir = os.path.join(root, "models")

    texts, labels = load_text_data(data_dir)
    if not texts:
        print("❌ No data found; please put your .txt files under data/training_data/ai and …/human")
        sys.exit(1)

    print(f"📊 Loaded {len(texts)} text samples ({sum(labels)} AI, {len(labels) - sum(labels)} human)")

    # Train classical ML models
    train_classical(texts, labels, model_dir)
    
    # Train neural network models
    train_neural_networks(texts, labels, model_dir)


if __name__ == "__main__":
    main()