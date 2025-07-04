import os
import pickle

import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn

# — ensure NLTK data —
for pkg, path in [('punkt','tokenizers/punkt'), ('stopwords','corpora/stopwords')]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_resources():
    """Load TF–IDF vectorizer, vocabulary, and all models."""
    base = os.path.dirname(os.path.abspath(__file__))
    md = os.path.join(base, 'models')

    # 1) Vectorizer
    with open(os.path.join(md, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    # 2) Vocabulary (for PyTorch models)
    with open(os.path.join(md, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    # 3) Classical classifiers
    sk_models = {}
    for name in ['svm','decision_tree','adaboost']:
        fn = f"{name}_model.pkl"
        with open(os.path.join(md, fn),'rb') as f:
            sk_models[name.upper()] = pickle.load(f)

    # 4) PyTorch models
    dl_models = {}
    model_classes = {'CNN': TextCNN, 'LSTM': TextLSTM, 'RNN': TextRNN}
    
    for arch in ['CNN','LSTM','RNN']:
        model_path = os.path.join(md, f"{arch}.pkl")
        if os.path.exists(model_path):
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            vocab_size = checkpoint['vocab_size']
            
            # Create model instance
            model = model_classes[arch](vocab_size)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            dl_models[arch] = model

    # merge into one dict
    models = {**sk_models, **dl_models}
    return vectorizer, vocab, models

def preprocess_text(text: str) -> str:
    """lower, tokenize, remove stopwords & non-alpha"""
    txt = text.lower()
    toks = word_tokenize(txt)
    toks = [t for t in toks if t.isalpha() and t not in stop_words]
    return " ".join(toks)

def extract_text(uploaded) -> str:
    """Pull text from PDF, DOCX or raw TXT."""
    if uploaded.type=='application/pdf':
        with pdfplumber.open(uploaded) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    if 'wordprocessingml' in uploaded.type:
        doc = Document(uploaded)
        return "\n".join(p.text for p in doc.paragraphs)
    return uploaded.getvalue().decode('utf-8')

def predict(text: str,
            model_name: str,
            vectorizer: TfidfVectorizer,
            vocab: dict,
            models: dict,
            max_length=512) -> dict:
    """
    Return a dict {'Human':..., 'AI':...} for the given model.
    Sklearn models use predict_proba; PyTorch models use forward pass.
    """
    proc = preprocess_text(text)
    
    # Classical ML models
    if model_name in {'SVM','DECISION_TREE','ADABOOST'}:
        vec = vectorizer.transform([proc])
        proba = models[model_name].predict_proba(vec)[0]
    else:
        # PyTorch models
        # Convert text to indices
        tokens = proc.split()[:max_length]
        indices = [vocab.get(token, 0) for token in tokens]  # 0 for unknown words
        
        # Pad or truncate to max_length
        if len(indices) < max_length:
            indices.extend([1] * (max_length - len(indices)))  # 1 is <PAD> token
        else:
            indices = indices[:max_length]
        
        # Convert to tensor and predict
        input_tensor = torch.tensor([indices], dtype=torch.long)
        
        with torch.no_grad():
            model = models[model_name]
            p_ai = float(model(input_tensor))
        
        proba = [1 - p_ai, p_ai]

    return {'Human': proba[0], 'AI': proba[1]}


# PyTorch Model Classes (same as in train_models.py)
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
        x = self.embedding(x).transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()

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
        x = hidden[-1]
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()

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
        x = hidden[-1]
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()


def main():
    st.title("AI vs Human Text Detection")
    vect, vocab, models = load_resources()

    # ---- sidebar inputs ----
    up = st.sidebar.file_uploader("Upload Document", type=['pdf','docx','txt'])
    if up:
        text = extract_text(up)
    else:
        text = st.sidebar.text_area("Or paste text here")

    choices = list(models.keys())
    selected = st.sidebar.multiselect("Select Models", choices, default=choices[:3])

    if st.sidebar.button("Analyze") and text:
        # run each model
        results = {
            m: predict(text, m, vect, vocab, models)
            for m in selected
        }
        df = pd.DataFrame(results).T
        df['Prediction'] = np.where(df['AI']>df['Human'],'AI','Human')

        st.subheader("Prediction Results")
        st.table(df)

        st.subheader("Confidence Scores")
        for m,row in df.iterrows():
            st.write(f"**{m}** → AI: {row['AI']:.2f}, Human: {row['Human']:.2f}")
            st.progress(int(row['AI']*100))

        st.subheader("Text Statistics")
        st.write(f"Words: {len(text.split())}, Characters: {len(text)}, Sentences: {text.count('.')}")

        csv = df.to_csv(index=True).encode('utf-8')
        st.download_button("Download CSV", csv, "report.csv", "text/csv")

if __name__=="__main__":
    main()