#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
import pickle
from typing import List, Dict, Tuple, Optional


class ContextAwareToxicityDetectorV2:
    """
    Enhanced context-aware model with explicit context-word interactions.

    Approach:
    1. Base TF-IDF features
    2. Context indicator features (one-hot encoded)
    3. Context-sensitive word score (computed separately)

    The model learns that certain words have different weights in different contexts.
    """

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features

        # TF-IDF for text features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        # One-hot encoder for context
        self.context_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')

        # Classifier
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )

        # Context-sensitive words with base toxicity
        # Format: word -> (base_toxicity, context_sensitivity)
        # base_toxicity: how toxic the word is in neutral context (0-1)
        # context_sensitivity: how much context affects it (0-1)
        self.word_toxicity = {
            # Combat words - low base toxicity, high sensitivity
            'kill': (0.3, 0.8),
            'killed': (0.3, 0.8),
            'destroy': (0.25, 0.8),
            'destroyed': (0.25, 0.8),
            'die': (0.3, 0.7),
            'dead': (0.25, 0.7),
            'murder': (0.5, 0.6),
            'rekt': (0.3, 0.7),
            'owned': (0.25, 0.6),

            # Insults - high base toxicity, low sensitivity
            'noob': (0.7, 0.2),
            'newbie': (0.6, 0.2),
            'trash': (0.8, 0.15),
            'garbage': (0.8, 0.15),
            'idiot': (0.85, 0.1),
            'stupid': (0.8, 0.1),
            'loser': (0.7, 0.2),

            # Severe - always toxic
            'yourself': (0.9, 0.0),  # Usually "kill yourself"

            # Positive - always non-toxic
            'nice': (0.05, 0.1),
            'good': (0.05, 0.1),
            'great': (0.05, 0.1),
            'thanks': (0.02, 0.0),
            'gg': (0.1, 0.3),  # Can be sarcastic
            'wp': (0.05, 0.1),
        }

        # Context toxicity modifiers (0 = reduces toxicity, 1 = increases)
        self.context_toxicity = {
            'pvp_combat': 0.2,    # Combat context - much less toxic
            'competitive': 0.4,   # Competitive - somewhat less
            'team_chat': 0.6,     # Team chat - moderate
            'post_game': 0.5,     # Post-game - moderate
            'all_chat': 0.7,      # All chat - fairly toxic
            'casual': 0.9,        # Casual - most toxic
        }

        self.is_fitted = False

    def _compute_context_word_score(self, text: str, context: str) -> float:
        """
        Compute a toxicity score based on context-sensitive words.

        For each word in the text:
        - If context-sensitive: score = base_toxicity * (1 - sensitivity * (1 - context_toxicity))
        - Example: "kill" (base=0.3, sens=0.8) in pvp_combat (ctx=0.2):
                   score = 0.3 * (1 - 0.8 * (1 - 0.2)) = 0.3 * 0.36 = 0.108
        - Same "kill" in casual (ctx=0.9):
                   score = 0.3 * (1 - 0.8 * (1 - 0.9)) = 0.3 * 0.92 = 0.276
        """
        words = text.lower().split()
        context_modifier = self.context_toxicity.get(context, 0.5)

        total_score = 0.0
        count = 0

        for word in words:
            if word in self.word_toxicity:
                base_tox, sensitivity = self.word_toxicity[word]
                # In low toxicity context, reduce score more for sensitive words
                adjusted = base_tox * (1 - sensitivity * (1 - context_modifier))
                total_score += adjusted
                count += 1

        return total_score / max(count, 1)

    def _create_features(self, X: List[str], contexts: List[str], fit: bool = False) -> csr_matrix:
        """
        Create feature matrix combining:
        1. TF-IDF text features
        2. One-hot context features
        3. Context-word interaction score
        """
        # 1. TF-IDF features
        if fit:
            X_tfidf = self.vectorizer.fit_transform(X)
        else:
            X_tfidf = self.vectorizer.transform(X)

        # 2. Context one-hot encoding
        context_array = np.array(contexts).reshape(-1, 1)
        if fit:
            X_context = self.context_encoder.fit_transform(context_array)
        else:
            X_context = self.context_encoder.transform(context_array)

        # 3. Context-word interaction scores
        interaction_scores = np.array([
            self._compute_context_word_score(text, ctx)
            for text, ctx in zip(X, contexts)
        ]).reshape(-1, 1)

        # Combine all features
        X_combined = hstack([X_tfidf, X_context, csr_matrix(interaction_scores)])

        return X_combined

    def fit(self, X: List[str], y: np.ndarray, contexts: List[str]):
        """Train the context-aware model."""
        X_features = self._create_features(X, contexts, fit=True)
        self.classifier.fit(X_features, y)
        self.is_fitted = True

        print(f"✓ Context-aware model v2 trained on {len(X)} samples")
        print(f"✓ TF-IDF features: {self.vectorizer.get_feature_names_out().shape[0]}")
        print(f"✓ Context types: {self.context_encoder.categories_[0].tolist()}")

        return self

    def predict(self, X: List[str], contexts: List[str]) -> np.ndarray:
        """Predict toxicity labels."""
        X_features = self._create_features(X, contexts, fit=False)
        return self.classifier.predict(X_features)

    def predict_proba(self, X: List[str], contexts: List[str]) -> np.ndarray:
        """Predict toxicity probabilities."""
        X_features = self._create_features(X, contexts, fit=False)
        return self.classifier.predict_proba(X_features)

    def compare_contexts(self, text: str) -> pd.DataFrame:
        """Compare predictions across all contexts."""
        data = []
        for ctx in sorted(self.context_toxicity.keys(), key=lambda x: self.context_toxicity[x]):
            proba = self.predict_proba([text], [ctx])[0]
            word_score = self._compute_context_word_score(text, ctx)
            data.append({
                'Context': ctx,
                'Context Toxicity': f"{self.context_toxicity[ctx]:.1f}",
                'Word Score': f"{word_score:.3f}",
                'Toxic Prob': f"{proba[1]:.1%}",
                'Prediction': 'TOXIC' if proba[1] > 0.5 else 'NON-TOXIC'
            })
        return pd.DataFrame(data)

    def explain_words(self, text: str) -> pd.DataFrame:
        """Explain how each word contributes in different contexts."""
        words = text.lower().split()
        data = []

        for word in words:
            if word in self.word_toxicity:
                base_tox, sensitivity = self.word_toxicity[word]
                for ctx, ctx_mod in self.context_toxicity.items():
                    adjusted = base_tox * (1 - sensitivity * (1 - ctx_mod))
                    data.append({
                        'Word': word,
                        'Context': ctx,
                        'Base Toxicity': f"{base_tox:.2f}",
                        'Sensitivity': f"{sensitivity:.2f}",
                        'Adjusted Score': f"{adjusted:.3f}"
                    })

        return pd.DataFrame(data)

    def get_context_sensitive_features(self) -> pd.DataFrame:
        """Get all context-sensitive features and their sensitivity scores."""
        data = []
        for word, (base_tox, sens) in sorted(self.word_toxicity.items(), key=lambda x: x[1][1], reverse=True):
            data.append({
                'Word/Phrase': word,
                'Base Toxicity': f"{base_tox:.0%}",
                'Context Sensitivity': f"{sens:.0%}",
                'Interpretation': 'Highly context-dependent' if sens > 0.5 else
                                 'Moderately context-dependent' if sens > 0.2 else
                                 'Context-independent'
            })
        return pd.DataFrame(data)


def demo_v2():
    """Demonstrate the enhanced context-aware model."""
    print("\n" + "=" * 80)
    print("CONTEXT-AWARE TOXICITY DETECTION v2 DEMO")
    print("=" * 80)

    # More realistic training data
    np.random.seed(42)

    # Generate training data with context-appropriate labeling
    training_data = [
        # Combat context - violence OK
        ("kill them all", 0, "pvp_combat"),
        ("i destroyed you", 0, "pvp_combat"),
        ("you're dead", 0, "pvp_combat"),
        ("nice kill", 0, "pvp_combat"),
        ("get rekt", 0, "pvp_combat"),

        # Casual context - violence NOT OK
        ("i'll kill you", 1, "casual"),
        ("you're dead", 1, "casual"),
        ("i'll destroy you", 1, "casual"),

        # Always toxic (insults)
        ("you're trash", 1, "pvp_combat"),
        ("noob team", 1, "competitive"),
        ("you're garbage", 1, "casual"),
        ("idiot player", 1, "team_chat"),

        # Always non-toxic
        ("nice shot", 0, "pvp_combat"),
        ("good game", 0, "post_game"),
        ("gg wp", 0, "post_game"),
        ("thanks for the game", 0, "casual"),
        ("great teamwork", 0, "team_chat"),
        ("well played", 0, "competitive"),
    ]

    messages, labels, contexts = zip(*training_data)
    messages = list(messages)
    labels = np.array(labels)
    contexts = list(contexts)

    # Train model
    model = ContextAwareToxicityDetectorV2(max_features=500)
    model.fit(messages, labels, contexts)

    # Test messages that should change with context
    test_cases = [
        "i'll kill you",
        "get destroyed",
        "you're trash",
        "nice shot",
    ]

    for msg in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Message: \"{msg}\"")
        print("=" * 60)

        comparison = model.compare_contexts(msg)
        print(comparison.to_string(index=False))

        # Show word-level explanation
        print(f"\nWord-level analysis:")
        word_analysis = model.explain_words(msg)
        if not word_analysis.empty:
            # Pivot for readability
            pivot = word_analysis.pivot(index='Word', columns='Context', values='Adjusted Score')
            print(pivot.to_string())


if __name__ == "__main__":
    demo_v2()
