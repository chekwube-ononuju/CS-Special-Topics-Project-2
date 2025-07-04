# AI vs Human Text Detection

This project is a Streamlit web application that analyzes text documents (PDF, Word, or plain text) and predicts whether the text was written by an AI or a human. The system provides:

- Upload or paste text input
- Choose from multiple trained models (SVM, Decision Tree, AdaBoost, CNN, LSTM, RNN)
- Instant predictions with confidence scores
- Text statistics and visualization
- Downloadable CSV reports

## Project Structure

```
ai_human_detection_project/
├── app.py                 # Main Streamlit application
├── train_models.py        # Script to train and save models/vectorizer
├── requirements.txt       # Python dependencies
├── models/                # Trained models and vectorizer
│   ├── svm_model.pkl         # Support Vector Machine
│   ├── decision_tree_model.pkl # Decision Tree
│   ├── adaboost_model.pkl    # AdaBoost
│   ├── tfidf_vectorizer.pkl  # TF-IDF vectorizer
│   ├── vocab.pkl             # Vocabulary for neural networks
│   ├── CNN.pkl               # Convolutional Neural Network (PyTorch)
│   ├── LSTM.pkl              # Long Short-Term Memory (PyTorch)
│   └── RNN.pkl               # Recurrent Neural Network (PyTorch)
├── data/
│   ├── training_data/     # Organized subfolders ai/ and human/ with .txt files
│   └── test_data/         # Test documents
└── notebooks/
    └── ai_human_detection.ipynb  # Data exploration and training details
```

## Setup & Installation

1. Clone this repository:
   ```zsh
   git clone https://github.com/chekwube-ononuju/CS-Special-Topics-Project-2.git
   cd CS-Special-Topics-Project-2
   ```
2. Create and activate a virtual environment:
   ```zsh
   python3 -m venv venv
   source venv/bin/activate
   # If your venv folder lives one level above (e.g., in the parent workspace), run:
   #   source ../venv/bin/activate
   ```
3. Upgrade pip, setuptools and wheel in the venv:
   ```zsh
   python -m pip install --upgrade pip setuptools wheel
   ```
4. Install dependencies (ensure you’re in the project folder where `requirements.txt` lives):
   ```zsh
   python -m pip install -r requirements.txt
   ```
   Note: if you see `ERROR: Could not open requirements file`, run:
   ```zsh
   cd ai_human_detection_project
   python -m pip install -r requirements.txt
   ```

5. Prepare your data:
   - Place AI-generated texts in `data/training_data/ai/` as `.txt` files
   - Place human-written texts in `data/training_data/human/` as `.txt` files
   - Ensure you have at least one `.txt` per folder; otherwise training will fail with "empty vocabulary".
   
   **Note**: The original `AI_Human.csv` dataset (1GB+) is not included in this repository due to GitHub's file size limits. You'll need to obtain your own dataset or use the existing `.txt` files in the training folders.

6. Train models (run this from the project root):
   ```zsh
   # ensure you're in the ai_human_detection_project folder
   cd /Users/chekzononuju/Documents/CS\ Special\ Topics\ Project\ 2/ai_human_detection_project
   source ../venv/bin/activate    # if venv is one level up
   python train_models.py
   ```

   **What this creates:**
   - Classical ML models: SVM, Decision Tree, AdaBoost (`.pkl` files)
   - Neural network models: CNN, LSTM, RNN (PyTorch `.pkl` files)
   - TF-IDF vectorizer for text preprocessing
   - Vocabulary file for neural networks

   **Regenerating models**
   If your `models/` directory is empty or model files have been removed, recreate them by running:
   ```zsh
   python train_models.py
   ```
   This will create all 6 models (3 classical ML + 3 neural networks) plus the vectorizer and vocabulary files.

7. Run the app:
   ```zsh
   streamlit run app.py
   ```

## Usage

- **Upload** a PDF, DOCX, or TXT file, or **paste** text directly.
- **Select** one or more models in the sidebar.
- Click **Analyze** to see predictions and visualizations.
- Download a detailed CSV report.

## Model Types

The system includes two categories of models:

**Classical Machine Learning Models:**
- **SVM (Support Vector Machine)**: Uses linear separation with TF-IDF features
- **Decision Tree**: Rule-based classifier with interpretable decisions  
- **AdaBoost**: Ensemble method combining weak learners

**Neural Network Models (PyTorch):**
- **CNN (Convolutional Neural Network)**: Uses convolutional layers for text pattern recognition
- **LSTM (Long Short-Term Memory)**: Recurrent network that captures long-term dependencies
- **RNN (Recurrent Neural Network)**: Basic recurrent architecture for sequence processing

All models are trained on the same dataset and can be used individually or in combination for ensemble predictions.

## Troubleshooting

If you encounter a `ModuleNotFoundError: No module named 'pip._internal.cli.main'` when installing dependencies, re-bootstrap and upgrade pip:
```zsh
python -m ensurepip --default-pip  # restore pip in your venv
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```
If you still see `ModuleNotFoundError: No module named 'pip._internal.cli.main'`, try bootstrapping pip directly:
```zsh
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
python -m pip install -r requirements.txt
```

## Recreate a fresh virtual environment

If your venv is broken, you can blow it away and start over:
```zsh
# 1. Change to project folder
cd "$(pwd)/ai_human_detection_project"

# 2. Deactivate & delete
deactivate
rm -rf venv

# 3. Recreate (upgrades pip, setuptools, wheel)
python3 -m venv venv --upgrade-deps

# 4. Activate
source venv/bin/activate

# 5. Install dependencies
python -m pip install -r requirements.txt
```

## License

This project is released under the MIT License.
