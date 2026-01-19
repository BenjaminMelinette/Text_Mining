import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
    roc_curve
)
from typing import Dict, Tuple
import json
import os


class ModelEvaluator:
    """Evaluation metrics for toxicity detection model."""

    def __init__(self):
        self.metrics = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }

        # Add ROC-AUC if probabilities provided
        if y_proba is not None:
            try:
                self.metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                self.metrics['roc_auc'] = None

        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()

        # Add derived metrics
        tn, fp, fn, tp = cm.ravel()
        self.metrics['true_negatives'] = int(tn)
        self.metrics['false_positives'] = int(fp)
        self.metrics['false_negatives'] = int(fn)
        self.metrics['true_positives'] = int(tp)

        # Specificity
        self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return self.metrics

    def print_metrics(self, title="Model Evaluation"):
        """Print metrics in a formatted table."""
        if not self.metrics:
            print("No metrics available. Run evaluate() first.")
            return

        print("\n" + "=" * 60)
        print(f"{title:^60}")
        print("=" * 60)

        print(f"\n{'Metric':<20} {'Value':>15}")
        print("-" * 40)
        print(f"{'Accuracy':<20} {self.metrics['accuracy']:>14.4f}")
        print(f"{'Precision':<20} {self.metrics['precision']:>14.4f}")
        print(f"{'Recall':<20} {self.metrics['recall']:>14.4f}")
        print(f"{'F1-Score':<20} {self.metrics['f1_score']:>14.4f}")
        print(f"{'Specificity':<20} {self.metrics['specificity']:>14.4f}")

        if self.metrics.get('roc_auc'):
            print(f"{'ROC-AUC':<20} {self.metrics['roc_auc']:>14.4f}")

        print("\n" + "-" * 40)
        print("Confusion Matrix:")
        print("-" * 40)
        print(f"{'':>15} {'Predicted Non-Toxic':>20} {'Predicted Toxic':>20}")
        print(f"{'Actual Non-Toxic':<15} {self.metrics['true_negatives']:>20} {self.metrics['false_positives']:>20}")
        print(f"{'Actual Toxic':<15} {self.metrics['false_negatives']:>20} {self.metrics['true_positives']:>20}")
        print("=" * 60 + "\n")

    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Print detailed classification report."""
        print("\n" + "=" * 60)
        print("Classification Report")
        print("=" * 60)
        print(classification_report(
            y_true, y_pred,
            target_names=['Non-Toxic', 'Toxic'],
            digits=4
        ))

    def save_metrics(self, output_path='results/metrics.json'):
        """
        Save metrics to JSON file.

        Args:
            output_path: Path to save metrics
        """
        if not self.metrics:
            print("No metrics to save. Run evaluate() first.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create a copy without numpy types for JSON serialization
        metrics_to_save = {}
        for key, value in self.metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_to_save[key] = float(value)
            else:
                metrics_to_save[key] = value

        with open(output_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)

        print(f"✓ Metrics saved to {output_path}")

    def compare_models(self, baseline_metrics: Dict, context_metrics: Dict):
        """
        Compare baseline model with context-adjusted model.

        Args:
            baseline_metrics: Metrics from model without context
            context_metrics: Metrics from model with context
        """
        print("\n" + "=" * 80)
        print("Model Comparison: Baseline vs. Context-Adjusted")
        print("=" * 80)

        print(f"\n{'Metric':<20} {'Baseline':>15} {'With Context':>15} {'Improvement':>15}")
        print("-" * 70)

        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

        for metric in metrics_to_compare:
            baseline_val = baseline_metrics.get(metric, 0)
            context_val = context_metrics.get(metric, 0)

            if baseline_val is None or context_val is None:
                continue

            improvement = context_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0

            print(f"{metric:<20} {baseline_val:>14.4f} {context_val:>14.4f} "
                  f"{improvement_pct:>13.2f}%")

        print("=" * 80 + "\n")


def demo_evaluation():
    """Demonstrate evaluation metrics."""
    # Sample predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
    y_proba = np.array([
        [0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3],
        [0.6, 0.4], [0.85, 0.15], [0.15, 0.85], [0.25, 0.75], [0.45, 0.55]
    ])

    evaluator = ModelEvaluator()
    evaluator.evaluate(y_true, y_pred, y_proba)
    evaluator.print_metrics("Demo Evaluation")
    evaluator.print_classification_report(y_true, y_pred)


if __name__ == "__main__":
    demo_evaluation()
