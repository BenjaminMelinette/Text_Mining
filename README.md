# Gaming Chat Toxicity Detector

A text mining project for detecting toxic messages in video game chats using **TF-IDF** and **Logistic Regression**.

## Overview

This project demonstrates a complete text mining pipeline for real-time toxicity detection in gaming chat messages. It features:

- Text preprocessing and cleaning
- TF-IDF feature extraction
- Logistic Regression classification
- Contextual feature engineering
- Interactive Streamlit demo
- Comprehensive Jupyter notebook analysis

## Project Structure

```
toxic-chat-detector/
├── data/
│   ├── raw/                    # Original dataset
│   │   └── gaming_chat_dataset.csv
│   ├── processed/              # Processed data (generated)
│   └── generate_dataset.py     # Dataset generator script
├── src/
│   ├── preprocessing.py        # Text cleaning and preprocessing
│   ├── model.py               # TF-IDF + Logistic Regression model
│   └── evaluate.py            # Evaluation metrics and visualization
├── notebooks/
│   └── analysis.ipynb         # Jupyter notebook with full analysis
├── models/                     # Trained models (generated)
│   ├── baseline_model.pkl
│   └── context_model.pkl
├── results/                    # Evaluation results (generated)
│   ├── metrics.json
│   └── visualizations/        # Charts and plots
├── main.py                     # Main training script
├── demo.py                     # Interactive Streamlit demo
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### 1. Clone or Download the Project

```bash
cd toxic-chat-detector
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Generate Dataset (Already Done)

The synthetic dataset has already been generated. To regenerate:

```bash
cd data
python generate_dataset.py
```

### Step 2: Train Models

Run the main training pipeline:

```bash
python main.py
```

This will:
- Load and preprocess the data
- Train baseline model (without context)
- Train context-adjusted model
- Evaluate both models
- Save models and metrics
- Display performance comparison

**Expected output:**
- Baseline accuracy: ~85-90%
- Context-adjusted accuracy: ~88-92%

### Step 3: Explore Jupyter Notebook

Launch Jupyter and open the analysis notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook includes:
- Data exploration and visualization
- Word clouds for toxic/non-toxic messages
- Model training and evaluation
- Feature importance analysis
- ROC curves and confusion matrices
- Context effect demonstrations

### Step 4: Try Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo.py
```

Features:
- **Live Detection**: Test individual messages in real-time
- **Batch Analysis**: Analyze multiple messages at once
- **Model Insights**: View top toxic indicators and model performance
- **Context Adjustment**: Toggle context-aware predictions

## Technical Details

### Why TF-IDF?

- **Captures word importance**: Identifies words that are distinctive to toxic messages
- **Reduces common word impact**: Words like "the", "a" get lower weights
- **Simple and interpretable**: Easy to understand and explain
- **Fast computation**: Suitable for real-time applications

### Why Logistic Regression?

- **Interpretable coefficients**: Shows which words contribute most to toxicity
- **Fast predictions**: Ideal for production deployment
- **Probabilistic output**: Provides confidence scores (0-1)
- **Works well with TF-IDF**: Strong baseline for text classification

### Context Adjustment

The context-adjusted model incorporates game environment information:

- **pvp_combat** (risk: 0.2): In-game combat, aggressive language more acceptable
- **casual** (risk: 0.3): Casual game mode, relaxed environment
- **team_chat** (risk: 0.5): Team coordination chat
- **post_game** (risk: 0.6): After match discussion
- **all_chat** (risk: 0.7): Public chat with all players
- **competitive** (risk: 0.8): Ranked mode, higher tension

Higher context scores lower the toxicity threshold, making the model more sensitive.

## Dataset

### Synthetic Dataset Characteristics

- **Size**: 2000 messages
- **Distribution**: ~55% non-toxic, ~45% toxic
- **Features**:
  - `message`: Raw chat message
  - `toxicity`: Toxicity score (0-1)
  - `context_score`: Game context risk level (0-1)
  - `context_type`: Type of game context
  - `label`: Binary label (0=non-toxic, 1=toxic)

### Example Messages

**Non-toxic:**
- "gg well played team!"
- "nice shot!"
- "good luck have fun"

**Toxic:**
- "you're trash"
- "uninstall the game noob"
- "worst player ever"

**Ambiguous (context-dependent):**
- "i'll kill you" (acceptable in PvP combat, toxic elsewhere)
- "ez noob" (borderline toxic)
- "destroyed you" (depends on context)

## Model Performance

### Baseline Model (Text Only)

| Metric | Score |
|--------|-------|
| Accuracy | ~87% |
| Precision | ~85% |
| Recall | ~88% |
| F1-Score | ~86% |

### Context-Adjusted Model

| Metric | Score |
|--------|-------|
| Accuracy | ~90% |
| Precision | ~88% |
| Recall | ~91% |
| F1-Score | ~89% |

**Improvement**: ~3-5% across all metrics

## Usage Examples

### Python API

```python
from src.preprocessing import TextPreprocessor
from src.model import ToxicityDetector

# Load model
model = ToxicityDetector.load('models/baseline_model.pkl')
preprocessor = TextPreprocessor()

# Predict single message
message = "gg well played!"
cleaned = preprocessor.clean_text(message)
prediction = model.predict([cleaned])[0]
probability = model.predict_proba([cleaned])[0]

print(f"Prediction: {'Toxic' if prediction == 1 else 'Non-Toxic'}")
print(f"Toxicity score: {probability[1]:.2%}")
```

### With Context

```python
# Load context model
model = ToxicityDetector.load('models/context_model.pkl')

# Predict with context
message = "i'll kill you"
cleaned = preprocessor.clean_text(message)
context_score = 0.2  # PvP combat context

prediction = model.predict([cleaned], [context_score])[0]
print(f"Prediction: {'Toxic' if prediction == 1 else 'Non-Toxic'}")
```

## Evaluation Metrics Explained

- **Accuracy**: Overall correctness (correct predictions / total predictions)
- **Precision**: Of predicted toxic messages, how many are truly toxic?
- **Recall**: Of all toxic messages, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (threshold-independent metric)

## Troubleshooting

### Models not found error

Run `python main.py` first to train the models.

### Import errors

Make sure all dependencies are installed: `pip install -r requirements.txt`

### Jupyter notebook kernel issues

```bash
python -m ipykernel install --user --name=toxic-detector
```

### Streamlit port already in use

```bash
streamlit run demo.py --server.port 8502
```

## Project Timeline

This project is designed to be completed in **1 day**:

- **Morning (4 hours)**:
  - Dataset generation and exploration (1h)
  - Model training and evaluation (2h)
  - Initial visualizations (1h)

- **Afternoon (4 hours)**:
  - Jupyter notebook completion (2h)
  - Streamlit demo development (1.5h)
  - Presentation preparation (0.5h)

## Resources

### Text Mining Concepts
- TF-IDF: [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- Logistic Regression: [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Related Work
- [Jigsaw Toxic Comment Classification (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Perspective API (Google)](https://perspectiveapi.com/)

## License

This project is for educational purposes (Text Mining Course).

## Author

Created for Year 5 Computer Science - Text Mining Course

---

**Note**: This is a pedagogical project demonstrating text mining principles. For production use, consider additional features like:
- Larger and more diverse training data
- Regular model retraining
- Human-in-the-loop validation
- Multi-language support
- Appeals process for false positives
