#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ModelComparator:
    """
    Compare multiple classifiers for toxicity detection.
    Provides cross-validation, statistical tests, and detailed analysis.
    """

    def __init__(self, max_features=5000):
        """
        Initialize model comparator.

        Args:
            max_features: Maximum TF-IDF features
        """
        self.max_features = max_features

        # TF-IDF vectorizer (shared across all models)
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        # Models to compare
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'Linear SVM': LinearSVC(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'Naive Bayes': MultinomialNB(
                alpha=1.0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        }

        self.results = {}
        self.cv_scores = {}

    def fit_transform(self, X_train):
        """Fit TF-IDF vectorizer on training data."""
        return self.vectorizer.fit_transform(X_train)

    def transform(self, X):
        """Transform data using fitted vectorizer."""
        return self.vectorizer.transform(X)

    def cross_validate_all(self, X, y, cv=5):
        """
        Perform cross-validation on all models.

        Args:
            X: Text data (preprocessed)
            y: Labels
            cv: Number of folds

        Returns:
            Dictionary with CV scores for each model
        """
        print("=" * 70)
        print("CROSS-VALIDATION COMPARISON (5-Fold)")
        print("=" * 70)

        # Transform text to TF-IDF
        X_tfidf = self.vectorizer.fit_transform(X)

        # Cross-validation for each model
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")

            # Multiple metrics
            accuracy_scores = cross_val_score(model, X_tfidf, y, cv=cv_splitter, scoring='accuracy')
            precision_scores = cross_val_score(model, X_tfidf, y, cv=cv_splitter, scoring='precision')
            recall_scores = cross_val_score(model, X_tfidf, y, cv=cv_splitter, scoring='recall')
            f1_scores = cross_val_score(model, X_tfidf, y, cv=cv_splitter, scoring='f1')

            self.cv_scores[name] = {
                'accuracy': accuracy_scores,
                'precision': precision_scores,
                'recall': recall_scores,
                'f1': f1_scores
            }

            print(f"  Accuracy:  {accuracy_scores.mean():.4f} ± {accuracy_scores.std():.4f}")
            print(f"  Precision: {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
            print(f"  Recall:    {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
            print(f"  F1-Score:  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

        return self.cv_scores

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train all models and evaluate on test set.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Dictionary with results for each model
        """
        print("\n" + "=" * 70)
        print("MODEL COMPARISON ON TEST SET")
        print("=" * 70)

        # Transform text
        X_train_tfidf = self.fit_transform(X_train)
        X_test_tfidf = self.transform(X_test)

        for name, model in self.models.items():
            print(f"\n--- {name} ---")

            # Train
            model.fit(X_train_tfidf, y_train)

            # Predict
            y_pred = model.predict(X_test_tfidf)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'model': model
            }

            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")

        return self.results

    def statistical_significance_test(self):
        """
        Perform paired t-test to check if differences are statistically significant.

        Returns:
            DataFrame with p-values
        """
        if not self.cv_scores:
            print("Run cross_validate_all() first!")
            return None

        print("\n" + "=" * 70)
        print("STATISTICAL SIGNIFICANCE TEST (Paired t-test on F1 scores)")
        print("=" * 70)
        print("\nNull hypothesis: No difference between models (p > 0.05)")
        print("-" * 70)

        models = list(self.cv_scores.keys())
        n_models = len(models)

        # Create p-value matrix
        p_values = np.zeros((n_models, n_models))

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    # Paired t-test on F1 scores
                    t_stat, p_value = stats.ttest_rel(
                        self.cv_scores[model1]['f1'],
                        self.cv_scores[model2]['f1']
                    )
                    p_values[i, j] = p_value
                else:
                    p_values[i, j] = 1.0

        # Create DataFrame
        p_df = pd.DataFrame(p_values, index=models, columns=models)

        print("\nP-values matrix:")
        print(p_df.round(4).to_string())

        print("\n" + "-" * 70)
        print("Interpretation:")
        print("  p < 0.05: Statistically significant difference")
        print("  p >= 0.05: No significant difference")

        # Find significant differences
        print("\nSignificant differences found:")
        found_significant = False
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j and p_values[i, j] < 0.05:
                    winner = model1 if self.cv_scores[model1]['f1'].mean() > self.cv_scores[model2]['f1'].mean() else model2
                    print(f"  {model1} vs {model2}: p={p_values[i, j]:.4f} → {winner} is significantly better")
                    found_significant = True

        if not found_significant:
            print("  None - all models perform similarly (no significant difference)")

        return p_df

    def get_comparison_table(self):
        """
        Generate comparison table for all models.

        Returns:
            DataFrame with comparison metrics
        """
        if not self.cv_scores:
            print("Run cross_validate_all() first!")
            return None

        data = []
        for name, scores in self.cv_scores.items():
            data.append({
                'Model': name,
                'Accuracy': f"{scores['accuracy'].mean():.4f} ± {scores['accuracy'].std():.4f}",
                'Precision': f"{scores['precision'].mean():.4f} ± {scores['precision'].std():.4f}",
                'Recall': f"{scores['recall'].mean():.4f} ± {scores['recall'].std():.4f}",
                'F1-Score': f"{scores['f1'].mean():.4f} ± {scores['f1'].std():.4f}",
                'Mean F1': scores['f1'].mean()
            })

        df = pd.DataFrame(data)
        df = df.sort_values('Mean F1', ascending=False)

        return df

    def print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "=" * 70)
        print("SUMMARY: MODEL COMPARISON")
        print("=" * 70)

        df = self.get_comparison_table()
        if df is None:
            return

        print("\nRanking by F1-Score (5-fold cross-validation):")
        print("-" * 70)
        for i, row in df.iterrows():
            print(f"  {row['Model']:<25} F1: {row['F1-Score']}")

        # Best model
        best_model = df.iloc[0]['Model']
        best_f1 = df.iloc[0]['Mean F1']

        print("\n" + "-" * 70)
        print(f"BEST MODEL: {best_model}")
        print(f"F1-Score: {best_f1:.4f}")
        print("-" * 70)

        return df


def run_model_comparison(X, y):
    """
    Run complete model comparison.

    Args:
        X: Preprocessed text data
        y: Labels

    Returns:
        ModelComparator instance with results
    """
    comparator = ModelComparator(max_features=5000)

    # Cross-validation
    comparator.cross_validate_all(X, y, cv=5)

    # Statistical significance
    comparator.statistical_significance_test()

    # Summary
    comparator.print_summary()

    return comparator


if __name__ == "__main__":
    # Demo with sample data
    import sys
    sys.path.insert(0, '.')

    from advanced_preprocessing import AdvancedTextPreprocessor

    # Load data
    print("Loading data...")
    df = pd.read_csv('../data/raw/gaming_chat_dataset.csv')

    # Preprocess
    preprocessor = AdvancedTextPreprocessor()
    X = preprocessor.preprocess_batch(df['message'].tolist())
    y = df['label'].values

    # Run comparison
    comparator = run_model_comparison(X, y)
