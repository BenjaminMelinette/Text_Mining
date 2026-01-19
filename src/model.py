import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Dict
import os


class ToxicityDetector:
    """Toxicity detection model using TF-IDF + Logistic Regression."""

    def __init__(self, max_features=5000, use_context=False, context_weight=0.3):
        """
        Initialize the toxicity detector.

        Args:
            max_features: Maximum number of features for TF-IDF
            use_context: Whether to incorporate context scores
            context_weight: Weight for context adjustment (0-1)
        """
        self.max_features = max_features
        self.use_context = use_context
        self.context_weight = context_weight

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )

        # Initialize Logistic Regression
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )

        self.is_trained = False

    def train(self, texts: List[str], labels: List[int], context_scores: List[float] = None):
        """
        Train the model.

        Args:
            texts: List of preprocessed text messages
            labels: List of binary labels (0=non-toxic, 1=toxic)
            context_scores: Optional list of context scores (0-1)
        """
        # Transform texts to TF-IDF features
        X = self.vectorizer.fit_transform(texts)

        # Train classifier
        self.classifier.fit(X, labels)
        self.is_trained = True

        print(f"✓ Model trained on {len(texts)} samples")
        print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def predict(self, texts: List[str], context_scores: List[float] = None) -> np.ndarray:
        """
        Predict toxicity labels.

        Args:
            texts: List of preprocessed text messages
            context_scores: Optional list of context scores

        Returns:
            Array of predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Get base predictions
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)

        # Apply context adjustment if enabled
        if self.use_context and context_scores is not None:
            predictions = self._adjust_with_context(texts, predictions, context_scores)

        return predictions

    def predict_proba(self, texts: List[str], context_scores: List[float] = None) -> np.ndarray:
        """
        Predict toxicity probabilities.

        Args:
            texts: List of preprocessed text messages
            context_scores: Optional list of context scores

        Returns:
            Array of probabilities for each class [prob_non_toxic, prob_toxic]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Get base probabilities
        X = self.vectorizer.transform(texts)
        probas = self.classifier.predict_proba(X)

        # Apply context adjustment if enabled
        if self.use_context and context_scores is not None:
            probas = self._adjust_proba_with_context(probas, context_scores)

        return probas

    def _adjust_with_context(self, texts: List[str], predictions: np.ndarray,
                            context_scores: List[float]) -> np.ndarray:
        """
        Adjust predictions based on context scores.

        High context score = higher risk environment = lower threshold for toxicity
        Low context score = safer environment = higher threshold for toxicity

        Args:
            texts: Original texts
            predictions: Base predictions
            context_scores: Context scores (0-1)

        Returns:
            Adjusted predictions
        """
        # Get probabilities
        X = self.vectorizer.transform(texts)
        probas = self.classifier.predict_proba(X)[:, 1]  # Probability of toxic class

        adjusted_predictions = []
        for i, (pred, proba, context) in enumerate(zip(predictions, probas, context_scores)):
            # Adjust threshold based on context
            # Higher context score = lower threshold (more likely to flag as toxic)
            threshold = 0.5 - (context * self.context_weight)
            adjusted_pred = 1 if proba >= threshold else 0
            adjusted_predictions.append(adjusted_pred)

        return np.array(adjusted_predictions)

    def _adjust_proba_with_context(self, probas: np.ndarray,
                                   context_scores: List[float]) -> np.ndarray:
        """
        Adjust probabilities based on context scores.

        Args:
            probas: Base probabilities
            context_scores: Context scores (0-1)

        Returns:
            Adjusted probabilities
        """
        adjusted_probas = probas.copy()

        for i, context in enumerate(context_scores):
            # Increase toxic probability based on context
            toxic_boost = context * self.context_weight
            adjusted_probas[i, 1] = min(1.0, probas[i, 1] + toxic_boost)
            adjusted_probas[i, 0] = 1.0 - adjusted_probas[i, 1]

        return adjusted_probas

    def get_feature_importance(self, top_n=20) -> Tuple[List[str], List[float]]:
        """
        Get the most important features (words) for toxicity detection.

        Args:
            top_n: Number of top features to return

        Returns:
            Tuple of (feature_names, coefficients)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        # Get feature names and coefficients
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.classifier.coef_[0]

        # Get top positive coefficients (toxic indicators)
        top_indices = np.argsort(coefficients)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_coefs = [coefficients[i] for i in top_indices]

        return top_features, top_coefs

    def save(self, model_path='models/toxicity_model.pkl'):
        """
        Save the trained model.

        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'max_features': self.max_features,
            'use_context': self.use_context,
            'context_weight': self.context_weight
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Model saved to {model_path}")

    @classmethod
    def load(cls, model_path='models/toxicity_model.pkl'):
        """
        Load a trained model.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded ToxicityDetector instance
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        detector = cls(
            max_features=model_data['max_features'],
            use_context=model_data['use_context'],
            context_weight=model_data['context_weight']
        )
        detector.vectorizer = model_data['vectorizer']
        detector.classifier = model_data['classifier']
        detector.is_trained = True

        print(f"✓ Model loaded from {model_path}")
        return detector


def demo_model():
    """Demonstrate model training on sample data."""
    from preprocessing import TextPreprocessor

    # Sample data
    train_texts = [
        "gg well played",
        "nice game everyone",
        "you're trash",
        "uninstall noob",
        "good try",
        "you suck",
        "thanks team",
        "worst player ever"
    ]
    train_labels = [0, 0, 1, 1, 0, 1, 0, 1]

    # Preprocess
    preprocessor = TextPreprocessor()
    clean_texts = preprocessor.preprocess_batch(train_texts)

    # Train model
    print("Training model...")
    detector = ToxicityDetector(max_features=100)
    detector.train(clean_texts, train_labels)

    # Test predictions
    test_texts = ["nice shot", "you're terrible"]
    clean_test = preprocessor.preprocess_batch(test_texts)

    predictions = detector.predict(clean_test)
    probas = detector.predict_proba(clean_test)

    print("\nPredictions:")
    for text, pred, proba in zip(test_texts, predictions, probas):
        label = "TOXIC" if pred == 1 else "NON-TOXIC"
        confidence = proba[pred] * 100
        print(f"  '{text}' -> {label} ({confidence:.1f}% confidence)")

    # Show feature importance
    print("\nTop toxic indicators:")
    features, coefs = detector.get_feature_importance(top_n=5)
    for feat, coef in zip(features, coefs):
        print(f"  {feat}: {coef:.3f}")


if __name__ == "__main__":
    demo_model()
