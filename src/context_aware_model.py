#!/usr/bin/env python3
"""
Context-Aware Toxicity Detector - Learnable Word×Context Interactions

This model learns word sensitivities and context effects from data,
rather than using hardcoded dictionaries.

Approach:
1. Base TF-IDF features for each word
2. Word×Context interaction features (learned)
3. Extract interpretable sensitivities after training
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack, lil_matrix
from typing import List
import pickle


class ContextAwareToxicityDetector:
    """
    Learnable context-aware toxicity detector.

    Instead of hardcoded word sensitivities, this model learns:
    - Base word toxicity (from TF-IDF + classifier weights)
    - Context-specific adjustments (from word×context interaction features)

    After training, you can extract:
    - Which words are most context-sensitive
    - How each context modifies toxicity
    - Per-word, per-context toxicity scores
    """

    def __init__(self, max_features: int = 5000, min_df: int = 2,
                 interaction_top_k: int = 100):
        """
        Args:
            max_features: Maximum vocabulary size for TF-IDF
            min_df: Minimum document frequency for TF-IDF
            interaction_top_k: Number of top words to create interaction features for
                              (limits feature explosion)
        """
        self.max_features = max_features
        self.min_df = min_df
        self.interaction_top_k = interaction_top_k

        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.8
        )

        # Context encoder
        self.context_encoder = LabelEncoder()

        # Classifier
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            C=1.0
        )

        # Learned after fitting
        self.interaction_words_ = None  # Words with interaction features
        self.contexts_ = None           # List of context types
        self.n_base_features_ = None    # Number of TF-IDF features
        self.n_contexts_ = None         # Number of context types
        self.is_fitted = False

    def _select_interaction_words(self, X_tfidf: csr_matrix, y: np.ndarray) -> List[str]:
        """
        Select top words for interaction features based on:
        1. Overall frequency (TF-IDF sum)
        2. Discrimination power (difference between toxic/non-toxic)
        """
        feature_names = self.vectorizer.get_feature_names_out()

        # Calculate mean TF-IDF for toxic vs non-toxic
        toxic_mask = y == 1
        non_toxic_mask = y == 0

        mean_toxic = np.asarray(X_tfidf[toxic_mask].mean(axis=0)).flatten()
        mean_non_toxic = np.asarray(X_tfidf[non_toxic_mask].mean(axis=0)).flatten()

        # Discrimination score = absolute difference
        discrimination = np.abs(mean_toxic - mean_non_toxic)

        # Overall importance = frequency × discrimination
        frequency = np.asarray(X_tfidf.sum(axis=0)).flatten()
        importance = frequency * discrimination

        # Select top k words
        top_indices = np.argsort(importance)[-self.interaction_top_k:]
        selected_words = [feature_names[i] for i in top_indices]

        return selected_words

    def _create_interaction_features(self, X_tfidf: csr_matrix,
                                      contexts: List[str]) -> csr_matrix:
        """
        Create word×context interaction features.

        For each selected word and each context, create a feature that is:
        - The TF-IDF value of that word IF the context matches
        - 0 otherwise

        This allows the model to learn different weights for "kill" in
        "pvp_combat" vs "kill" in "casual".
        """
        n_samples = X_tfidf.shape[0]
        n_interaction_features = len(self.interaction_words_) * self.n_contexts_

        # Get indices of interaction words in vocabulary
        feature_names = list(self.vectorizer.get_feature_names_out())
        word_indices = []
        for word in self.interaction_words_:
            if word in feature_names:
                word_indices.append(feature_names.index(word))
            else:
                word_indices.append(-1)

        # Encode contexts
        context_encoded = self.context_encoder.transform(contexts)

        # Build interaction matrix (use lil_matrix for efficient construction)
        interaction_matrix = lil_matrix((n_samples, n_interaction_features))

        for sample_idx in range(n_samples):
            ctx_idx = context_encoded[sample_idx]
            for word_idx_local, word_idx_global in enumerate(word_indices):
                if word_idx_global >= 0:
                    # Feature index = word_idx * n_contexts + context_idx
                    feat_idx = word_idx_local * self.n_contexts_ + ctx_idx
                    tfidf_val = X_tfidf[sample_idx, word_idx_global]
                    if tfidf_val != 0:
                        interaction_matrix[sample_idx, feat_idx] = tfidf_val

        return interaction_matrix.tocsr()

    def _create_features(self, X: List[str], contexts: List[str],
                         fit: bool = False) -> csr_matrix:
        """
        Create full feature matrix:
        1. Base TF-IDF features
        2. Context one-hot encoding
        3. Word×Context interaction features
        """
        # 1. TF-IDF features
        if fit:
            X_tfidf = self.vectorizer.fit_transform(X)
            self.n_base_features_ = X_tfidf.shape[1]
        else:
            X_tfidf = self.vectorizer.transform(X)

        # 2. Context encoding
        if fit:
            self.context_encoder.fit(contexts)
            self.contexts_ = list(self.context_encoder.classes_)
            self.n_contexts_ = len(self.contexts_)

        # Context one-hot
        context_encoded = self.context_encoder.transform(contexts)
        context_onehot = np.zeros((len(contexts), self.n_contexts_))
        context_onehot[np.arange(len(contexts)), context_encoded] = 1
        context_onehot = csr_matrix(context_onehot)

        # 3. Word×Context interaction features
        X_interaction = self._create_interaction_features(X_tfidf, contexts)

        # Combine all features
        X_combined = hstack([X_tfidf, context_onehot, X_interaction])

        return X_combined

    def fit(self, X: List[str], y: np.ndarray, contexts: List[str]):
        """
        Train the model and learn word×context interactions.
        """
        # Initial TF-IDF fit to select interaction words
        X_tfidf = self.vectorizer.fit_transform(X)
        self.n_base_features_ = X_tfidf.shape[1]

        # Fit context encoder
        self.context_encoder.fit(contexts)
        self.contexts_ = list(self.context_encoder.classes_)
        self.n_contexts_ = len(self.contexts_)

        # Select words for interaction features
        self.interaction_words_ = self._select_interaction_words(X_tfidf, y)

        # Create full feature matrix
        X_features = self._create_features(X, contexts, fit=False)

        # Train classifier
        self.classifier.fit(X_features, y)
        self.is_fitted = True

        n_interaction = len(self.interaction_words_) * self.n_contexts_
        print(f"✓ Context-aware model trained on {len(X)} samples")
        print(f"✓ Base TF-IDF features: {self.n_base_features_}")
        print(f"✓ Context types: {self.contexts_}")
        print(f"✓ Interaction words: {len(self.interaction_words_)}")
        print(f"✓ Interaction features: {n_interaction}")
        print(f"✓ Total features: {X_features.shape[1]}")

        return self

    def predict(self, X: List[str], contexts: List[str]) -> np.ndarray:
        """Predict toxicity labels."""
        X_features = self._create_features(X, contexts, fit=False)
        return self.classifier.predict(X_features)

    def predict_proba(self, X: List[str], contexts: List[str]) -> np.ndarray:
        """Predict toxicity probabilities."""
        X_features = self._create_features(X, contexts, fit=False)
        return self.classifier.predict_proba(X_features)

    def get_learned_word_sensitivities(self) -> pd.DataFrame:
        """
        Extract learned word sensitivities from model coefficients.

        Returns a DataFrame with:
        - Word
        - Base coefficient (from TF-IDF features)
        - Per-context coefficients (from interaction features)
        - Sensitivity score (variance across contexts)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        coef = self.classifier.coef_.flatten()
        feature_names = list(self.vectorizer.get_feature_names_out())

        # Base coefficients (first n_base_features)
        base_coef = coef[:self.n_base_features_]

        # Interaction coefficients (after base + context one-hot)
        interaction_start = self.n_base_features_ + self.n_contexts_
        interaction_coef = coef[interaction_start:]

        data = []
        for word_idx, word in enumerate(self.interaction_words_):
            # Get base coefficient
            if word in feature_names:
                base_idx = feature_names.index(word)
                base = base_coef[base_idx]
            else:
                base = 0.0

            # Get per-context coefficients
            context_coefs = {}
            for ctx_idx, ctx in enumerate(self.contexts_):
                feat_idx = word_idx * self.n_contexts_ + ctx_idx
                if feat_idx < len(interaction_coef):
                    context_coefs[ctx] = interaction_coef[feat_idx]
                else:
                    context_coefs[ctx] = 0.0

            # Sensitivity = std of context coefficients (higher = more context-dependent)
            sensitivity = np.std(list(context_coefs.values()))

            row = {
                'word': word,
                'base_coef': base,
                'sensitivity': sensitivity,
            }
            row.update({f'coef_{ctx}': context_coefs[ctx] for ctx in self.contexts_})
            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values('sensitivity', ascending=False)

        return df

    def get_learned_context_effects(self) -> pd.DataFrame:
        """
        Extract learned context effects.

        Returns a DataFrame with:
        - Context type
        - Base context coefficient (from one-hot features)
        - Average interaction effect
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        coef = self.classifier.coef_.flatten()

        # Context one-hot coefficients
        context_coef = coef[self.n_base_features_:self.n_base_features_ + self.n_contexts_]

        # Average interaction effect per context
        interaction_start = self.n_base_features_ + self.n_contexts_
        interaction_coef = coef[interaction_start:]

        data = []
        for ctx_idx, ctx in enumerate(self.contexts_):
            # Context one-hot coefficient
            base_effect = context_coef[ctx_idx]

            # Average of all word interactions for this context
            ctx_interactions = []
            for word_idx in range(len(self.interaction_words_)):
                feat_idx = word_idx * self.n_contexts_ + ctx_idx
                if feat_idx < len(interaction_coef):
                    ctx_interactions.append(interaction_coef[feat_idx])

            avg_interaction = np.mean(ctx_interactions) if ctx_interactions else 0.0

            data.append({
                'context': ctx,
                'base_effect': base_effect,
                'avg_word_interaction': avg_interaction,
                'total_effect': base_effect + avg_interaction
            })

        df = pd.DataFrame(data)
        df = df.sort_values('total_effect', ascending=False)

        return df

    def get_word_context_matrix(self) -> pd.DataFrame:
        """
        Get full word×context coefficient matrix for visualization.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        coef = self.classifier.coef_.flatten()
        feature_names = list(self.vectorizer.get_feature_names_out())

        # Base coefficients
        base_coef = coef[:self.n_base_features_]

        # Interaction coefficients
        interaction_start = self.n_base_features_ + self.n_contexts_
        interaction_coef = coef[interaction_start:]

        # Build matrix: effective coefficient = base + interaction
        matrix_data = {}
        for word_idx, word in enumerate(self.interaction_words_):
            if word in feature_names:
                base_idx = feature_names.index(word)
                base = base_coef[base_idx]
            else:
                base = 0.0

            row = {}
            for ctx_idx, ctx in enumerate(self.contexts_):
                feat_idx = word_idx * self.n_contexts_ + ctx_idx
                interaction = interaction_coef[feat_idx] if feat_idx < len(interaction_coef) else 0.0
                row[ctx] = base + interaction

            matrix_data[word] = row

        df = pd.DataFrame(matrix_data).T
        df.index.name = 'word'

        return df

    def compare_contexts(self, text: str) -> pd.DataFrame:
        """Compare predictions across all contexts for a given text."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        data = []
        for ctx in self.contexts_:
            proba = self.predict_proba([text], [ctx])[0]
            data.append({
                'Context': ctx,
                'Toxic Prob': f"{proba[1]:.1%}",
                'Prediction': 'TOXIC' if proba[1] > 0.5 else 'NON-TOXIC'
            })

        return pd.DataFrame(data)

    def explain_prediction(self, text: str, context: str) -> pd.DataFrame:
        """
        Explain which words contributed most to the prediction.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Get TF-IDF representation
        X_tfidf = self.vectorizer.transform([text])
        feature_names = list(self.vectorizer.get_feature_names_out())
        coef = self.classifier.coef_.flatten()

        # Get context index
        ctx_idx = self.context_encoder.transform([context])[0]

        # Calculate contribution of each word
        data = []
        tfidf_array = X_tfidf.toarray().flatten()

        for feat_idx, (tfidf_val, word) in enumerate(zip(tfidf_array, feature_names)):
            if tfidf_val > 0:
                # Base contribution
                base_contrib = tfidf_val * coef[feat_idx]

                # Interaction contribution (if word is in interaction set)
                interaction_contrib = 0.0
                if word in self.interaction_words_:
                    word_idx = self.interaction_words_.index(word)
                    interaction_feat_idx = self.n_base_features_ + self.n_contexts_ + \
                                          word_idx * self.n_contexts_ + ctx_idx
                    if interaction_feat_idx < len(coef):
                        interaction_contrib = tfidf_val * coef[interaction_feat_idx]

                total_contrib = base_contrib + interaction_contrib

                data.append({
                    'word': word,
                    'tfidf': tfidf_val,
                    'base_contrib': base_contrib,
                    'context_contrib': interaction_contrib,
                    'total_contrib': total_contrib
                })

        df = pd.DataFrame(data)
        df = df.sort_values('total_contrib', ascending=False)

        return df

    def save(self, path: str):
        """Save the trained model to a file."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'ContextAwareToxicityDetector':
        """Load a trained model from a file."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {path}")
        return model
