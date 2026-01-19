# Gaming Chat Toxicity Detector

A text mining project for detecting toxic messages in video game chats using **TF-IDF**, **Logistic Regression**, and **Learnable Word×Context Interactions**.

## Overview

This project demonstrates a complete text mining pipeline for real-time toxicity detection in gaming chat messages. It features:

- Advanced text preprocessing (leetspeak, abbreviations, evasion handling)
- TF-IDF feature extraction with n-grams
- **Learnable context-aware classification** (main contribution)
- Interactive Streamlit demo
- Comprehensive Jupyter notebook analysis

## Key Innovation: Learnable Context-Aware Detection

The main contribution is a **learnable word×context interaction** approach:

- Creates interaction features for top discriminative words × each context
- The model **learns from data** different weights for the same word in different contexts
- Example: "kill" in `pvp_combat` → low toxicity weight (normal gaming term)
- Example: "kill" in `casual` → higher toxicity weight (potentially threatening)
- After training, learned sensitivities can be extracted for interpretability

## Project Structure

```
toxic-chat-detector/
├── data/
│   ├── raw/                         # Original dataset
│   │   └── gaming_chat_dataset.csv
│   └── generate_dataset.py          # Dataset generator script
├── src/
│   ├── advanced_preprocessing.py    # Text cleaning (leetspeak, abbreviations)
│   ├── context_aware_model.py       # Learnable context-aware model (main)
│   ├── model.py                     # Baseline TF-IDF + LogReg model
│   ├── evaluate.py                  # Evaluation metrics
│   ├── model_comparison.py          # Cross-validation comparison
│   └── error_analysis.py            # Error analysis tools
├── notebooks/
│   └── analysis_m2.ipynb            # Full analysis notebook
├── models/                          # Trained models
│   ├── baseline_model.pkl
│   ├── context_aware_model.pkl      # Pre-trained context-aware model
│   └── preprocessor.pkl
├── results/                         # Evaluation results
│   └── visualizations/              # Charts and plots
├── main.py                          # Main training script
├── demo.py                          # Interactive Streamlit demo
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### 1. Clone or Download the Project

```bash
cd toxic-chat-detector
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Use Pre-trained Model (Recommended)

The context-aware model is already trained. Simply run the demo:

```bash
streamlit run demo.py
```

### Option 2: Retrain the Model

```bash
# Train and save the context-aware model
cd src
python context_aware_model.py
```

This will:
- Load and preprocess the dataset
- Train the learnable context-aware model
- Save to `models/context_aware_model.pkl`
- Display learned word sensitivities and context effects

### Option 3: Run Full Analysis Pipeline

```bash
python main.py
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline (LogReg) | 96.0% | 92.8% | 98.9% | 95.7% |
| Post-hoc Context | 96.5% | 92.8% | 100% | 96.3% |
| **Context-Aware (Learnable)** | **~100%** | **~100%** | **~100%** | **~100%** |

## Usage

### Python API

```python
from src.context_aware_model import ContextAwareToxicityDetector
from src.advanced_preprocessing import AdvancedTextPreprocessor

# Load pre-trained model
model = ContextAwareToxicityDetector.load('models/context_aware_model.pkl')
preprocessor = AdvancedTextPreprocessor(remove_stopwords=False)

# Predict with context
message = "i'll kill you"
cleaned = preprocessor.clean_text(message)
context = "pvp_combat"  # or: casual, competitive, team_chat, etc.

prediction = model.predict([cleaned], [context])[0]
proba = model.predict_proba([cleaned], [context])[0]

print(f"Prediction: {'Toxic' if prediction == 1 else 'Non-Toxic'}")
print(f"Toxicity probability: {proba[1]:.1%}")
```

### Compare Across Contexts

```python
# See how context changes the prediction
print(model.compare_contexts(cleaned))
```

Output:
```
    Context Toxic Prob Prediction
   all_chat      70.5%      TOXIC
     casual      32.4%  NON-TOXIC
competitive      71.0%      TOXIC
  post_game      72.9%      TOXIC
 pvp_combat      30.0%  NON-TOXIC
  team_chat      70.1%      TOXIC
```

### Extract Learned Insights

```python
# Get learned word sensitivities
sensitivities = model.get_learned_word_sensitivities()
print(sensitivities.head(10))

# Get learned context effects
context_effects = model.get_learned_context_effects()
print(context_effects)
```

## Technical Details

### Feature Construction (583 features total)

1. **Base TF-IDF features** (277 features)
   - Unigrams and bigrams
   - Min document frequency: 2
   - Max document frequency: 80%

2. **Context one-hot encoding** (6 features)
   - pvp_combat, casual, competitive, team_chat, post_game, all_chat

3. **Word×Context interaction features** (300 features)
   - Top 50 discriminative words × 6 contexts
   - Allows learning different weights per word per context

### Preprocessing Pipeline

- Lowercase conversion
- URL and mention removal
- **Leetspeak normalization** (tr4sh → trash)
- **Abbreviation expansion** (kys → kill yourself, gtfo → get the fuck out)
- **Spacing evasion detection** (k y s → kys)
- Unicode normalization

### Context Types

| Context | Effect | Description |
|---------|--------|-------------|
| pvp_combat | Reduces toxicity | Combat language is normal |
| casual | Reduces toxicity | Relaxed environment |
| team_chat | Moderate | Team coordination |
| post_game | Moderate | After match discussion |
| all_chat | Increases toxicity | Public visibility |
| competitive | Increases toxicity | High stakes, more tension |

## Streamlit Demo Features

- **Live Detection**: Test messages in real-time with context selection
- **Context Comparison**: See how the same message is classified across all contexts
- **Batch Analysis**: Analyze multiple messages at once
- **Model Insights**: View learned word sensitivities and context effects

## Jupyter Notebook

The notebook `notebooks/analysis_m2.ipynb` includes:

- Exploratory data analysis with visualizations
- Word clouds for toxic/non-toxic messages
- 5-fold cross-validation comparison (LogReg, SVM, NB, RF)
- Statistical significance testing
- Context-aware model training and evaluation
- Learned sensitivity extraction
- Error analysis

## Dataset

- **Size**: 2000 messages
- **Distribution**: ~55% non-toxic, ~45% toxic
- **Context types**: 6 game contexts
- **Features**: message, toxicity score, context_score, context_type, label

## Troubleshooting

### Model not found

Train the model first:
```bash
cd src && python context_aware_model.py
```

### Import errors

```bash
pip install -r requirements.txt
```

### Streamlit port in use

```bash
streamlit run demo.py --server.port 8502
```

## License

Educational project for Text Mining Course - Master 2.

## Author

Benjamin Melinette - January 2026