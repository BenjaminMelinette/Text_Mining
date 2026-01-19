#!/usr/bin/env python3

import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict
import re


class ErrorAnalyzer:
    """
    Comprehensive error analysis for toxicity detection model.
    Identifies patterns in misclassifications.
    """

    def __init__(self):
        self.false_positives = []
        self.false_negatives = []
        self.true_positives = []
        self.true_negatives = []

    def analyze(self, messages: List[str], y_true: np.ndarray,
                y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict:
        """
        Perform comprehensive error analysis.

        Args:
            messages: Original text messages
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            Dictionary with analysis results
        """
        self.false_positives = []
        self.false_negatives = []
        self.true_positives = []
        self.true_negatives = []

        for i, (msg, true, pred) in enumerate(zip(messages, y_true, y_pred)):
            proba = y_proba[i][1] if y_proba is not None else None

            entry = {
                'message': msg,
                'true_label': true,
                'predicted': pred,
                'probability': proba
            }

            if true == 0 and pred == 1:
                self.false_positives.append(entry)
            elif true == 1 and pred == 0:
                self.false_negatives.append(entry)
            elif true == 1 and pred == 1:
                self.true_positives.append(entry)
            else:
                self.true_negatives.append(entry)

        return {
            'false_positives': len(self.false_positives),
            'false_negatives': len(self.false_negatives),
            'true_positives': len(self.true_positives),
            'true_negatives': len(self.true_negatives),
            'total': len(messages)
        }

    def get_false_positives(self, n=10) -> List[Dict]:
        """Get top N false positives (non-toxic flagged as toxic)."""
        # Sort by probability (highest first - most confident mistakes)
        sorted_fp = sorted(
            self.false_positives,
            key=lambda x: x['probability'] if x['probability'] else 0,
            reverse=True
        )
        return sorted_fp[:n]

    def get_false_negatives(self, n=10) -> List[Dict]:
        """Get top N false negatives (toxic missed by model)."""
        # Sort by probability (lowest first - most confident mistakes)
        sorted_fn = sorted(
            self.false_negatives,
            key=lambda x: x['probability'] if x['probability'] else 1,
            reverse=False
        )
        return sorted_fn[:n]

    def get_borderline_cases(self, threshold=0.1, n=10) -> List[Dict]:
        """Get cases where probability was close to 0.5 (uncertain)."""
        all_cases = (self.false_positives + self.false_negatives +
                    self.true_positives + self.true_negatives)

        borderline = [
            case for case in all_cases
            if case['probability'] and abs(case['probability'] - 0.5) < threshold
        ]

        return sorted(borderline, key=lambda x: abs(x['probability'] - 0.5))[:n]

    def analyze_word_patterns(self) -> Dict:
        """Analyze word patterns in errors."""
        # Words in false positives
        fp_words = []
        for entry in self.false_positives:
            fp_words.extend(entry['message'].lower().split())

        # Words in false negatives
        fn_words = []
        for entry in self.false_negatives:
            fn_words.extend(entry['message'].lower().split())

        return {
            'false_positive_common_words': Counter(fp_words).most_common(10),
            'false_negative_common_words': Counter(fn_words).most_common(10)
        }

    def analyze_message_length(self) -> Dict:
        """Analyze if message length correlates with errors."""
        fp_lengths = [len(e['message'].split()) for e in self.false_positives]
        fn_lengths = [len(e['message'].split()) for e in self.false_negatives]
        tp_lengths = [len(e['message'].split()) for e in self.true_positives]
        tn_lengths = [len(e['message'].split()) for e in self.true_negatives]

        return {
            'false_positives': {
                'mean': np.mean(fp_lengths) if fp_lengths else 0,
                'std': np.std(fp_lengths) if fp_lengths else 0,
                'count': len(fp_lengths)
            },
            'false_negatives': {
                'mean': np.mean(fn_lengths) if fn_lengths else 0,
                'std': np.std(fn_lengths) if fn_lengths else 0,
                'count': len(fn_lengths)
            },
            'true_positives': {
                'mean': np.mean(tp_lengths) if tp_lengths else 0,
                'std': np.std(tp_lengths) if tp_lengths else 0,
                'count': len(tp_lengths)
            },
            'true_negatives': {
                'mean': np.mean(tn_lengths) if tn_lengths else 0,
                'std': np.std(tn_lengths) if tn_lengths else 0,
                'count': len(tn_lengths)
            }
        }

    def categorize_errors(self) -> Dict:
        """Categorize errors by type."""
        categories = {
            'ambiguous_language': [],      # Messages that could be either
            'context_dependent': [],        # Messages that need context
            'borderline_toxicity': [],      # Mild toxicity
            'missed_slurs': [],             # Missed offensive words
            'false_alarm_gaming': [],       # Gaming terms flagged incorrectly
        }

        # Keywords for categorization
        gaming_terms = ['kill', 'destroy', 'rekt', 'owned', 'noob', 'gg', 'ez']
        ambiguous_terms = ['you', 'bad', 'worst', 'terrible']

        for entry in self.false_positives:
            msg = entry['message'].lower()
            if any(term in msg for term in gaming_terms):
                categories['false_alarm_gaming'].append(entry)
            elif any(term in msg for term in ambiguous_terms):
                categories['ambiguous_language'].append(entry)

        for entry in self.false_negatives:
            msg = entry['message'].lower()
            if entry['probability'] and entry['probability'] > 0.3:
                categories['borderline_toxicity'].append(entry)
            else:
                categories['missed_slurs'].append(entry)

        return categories

    def print_report(self):
        """Print comprehensive error analysis report."""
        print("\n" + "=" * 80)
        print("ERROR ANALYSIS REPORT")
        print("=" * 80)

        # Summary
        total = (len(self.false_positives) + len(self.false_negatives) +
                len(self.true_positives) + len(self.true_negatives))

        print(f"\n1. SUMMARY")
        print("-" * 40)
        print(f"   Total samples analyzed: {total}")
        print(f"   True Positives:  {len(self.true_positives):4d} (toxic correctly identified)")
        print(f"   True Negatives:  {len(self.true_negatives):4d} (non-toxic correctly identified)")
        print(f"   False Positives: {len(self.false_positives):4d} (non-toxic flagged as toxic)")
        print(f"   False Negatives: {len(self.false_negatives):4d} (toxic missed)")

        # Error rates
        if total > 0:
            fp_rate = len(self.false_positives) / total * 100
            fn_rate = len(self.false_negatives) / total * 100
            print(f"\n   False Positive Rate: {fp_rate:.2f}%")
            print(f"   False Negative Rate: {fn_rate:.2f}%")

        # False Positives
        print(f"\n2. FALSE POSITIVES (Non-toxic flagged as toxic)")
        print("-" * 40)
        if self.false_positives:
            fps = self.get_false_positives(5)
            for i, fp in enumerate(fps, 1):
                prob = f"{fp['probability']:.1%}" if fp['probability'] else "N/A"
                print(f"   {i}. \"{fp['message']}\"")
                print(f"      Confidence: {prob}")
        else:
            print("   None! Model has no false positives.")

        # False Negatives
        print(f"\n3. FALSE NEGATIVES (Toxic messages missed)")
        print("-" * 40)
        if self.false_negatives:
            fns = self.get_false_negatives(5)
            for i, fn in enumerate(fns, 1):
                prob = f"{fn['probability']:.1%}" if fn['probability'] else "N/A"
                print(f"   {i}. \"{fn['message']}\"")
                print(f"      Toxicity score: {prob}")
        else:
            print("   None! Model catches all toxic messages.")

        # Borderline cases
        print(f"\n4. BORDERLINE CASES (Probability close to 0.5)")
        print("-" * 40)
        borderline = self.get_borderline_cases(threshold=0.15, n=5)
        if borderline:
            for i, case in enumerate(borderline, 1):
                label = "TOXIC" if case['true_label'] == 1 else "NON-TOXIC"
                pred = "TOXIC" if case['predicted'] == 1 else "NON-TOXIC"
                correct = "✓" if case['true_label'] == case['predicted'] else "✗"
                prob = f"{case['probability']:.1%}" if case['probability'] else "N/A"
                print(f"   {i}. \"{case['message']}\"")
                print(f"      True: {label}, Predicted: {pred} {correct}, Prob: {prob}")
        else:
            print("   No borderline cases found.")

        # Word patterns
        print(f"\n5. WORD PATTERN ANALYSIS")
        print("-" * 40)
        patterns = self.analyze_word_patterns()

        print("   Common words in False Positives:")
        if patterns['false_positive_common_words']:
            for word, count in patterns['false_positive_common_words'][:5]:
                print(f"      '{word}': {count}")
        else:
            print("      None")

        print("\n   Common words in False Negatives:")
        if patterns['false_negative_common_words']:
            for word, count in patterns['false_negative_common_words'][:5]:
                print(f"      '{word}': {count}")
        else:
            print("      None")

        # Message length analysis
        print(f"\n6. MESSAGE LENGTH ANALYSIS")
        print("-" * 40)
        lengths = self.analyze_message_length()
        print(f"   False Positives: mean={lengths['false_positives']['mean']:.1f} words")
        print(f"   False Negatives: mean={lengths['false_negatives']['mean']:.1f} words")
        print(f"   True Positives:  mean={lengths['true_positives']['mean']:.1f} words")
        print(f"   True Negatives:  mean={lengths['true_negatives']['mean']:.1f} words")

        # Conclusions
        print(f"\n7. CONCLUSIONS & RECOMMENDATIONS")
        print("-" * 40)

        if len(self.false_positives) > len(self.false_negatives):
            print("   → Model is OVERLY AGGRESSIVE (more false positives)")
            print("   → Consider raising the decision threshold")
            print("   → Review gaming-specific terms that trigger false alarms")
        elif len(self.false_negatives) > len(self.false_positives):
            print("   → Model is TOO LENIENT (more false negatives)")
            print("   → Consider lowering the decision threshold")
            print("   → Add more toxic examples to training data")
        else:
            print("   → Model is WELL-BALANCED")
            print("   → Continue monitoring for edge cases")

        if len(self.false_negatives) == 0:
            print("   → EXCELLENT: 100% recall achieved!")

        print("\n" + "=" * 80)

    def export_errors_to_csv(self, output_path='results/error_analysis.csv'):
        """Export all errors to CSV for manual review."""
        errors = []

        for fp in self.false_positives:
            errors.append({
                'message': fp['message'],
                'true_label': 'non-toxic',
                'predicted': 'toxic',
                'probability': fp['probability'],
                'error_type': 'false_positive'
            })

        for fn in self.false_negatives:
            errors.append({
                'message': fn['message'],
                'true_label': 'toxic',
                'predicted': 'non-toxic',
                'probability': fn['probability'],
                'error_type': 'false_negative'
            })

        if errors:
            df = pd.DataFrame(errors)
            df.to_csv(output_path, index=False)
            print(f"✓ Errors exported to {output_path}")
        else:
            print("No errors to export!")


def demo_error_analysis():
    """Demonstrate error analysis with sample data."""
    # Sample data
    messages = [
        "gg well played",
        "you're trash",
        "nice game",
        "worst player",
        "good luck",
        "noob team",
        "nice shot",
        "uninstall"
    ]
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])  # 1 FP, 1 FN
    y_proba = np.array([
        [0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.55, 0.45],
        [0.95, 0.05], [0.3, 0.7], [0.45, 0.55], [0.1, 0.9]
    ])

    analyzer = ErrorAnalyzer()
    analyzer.analyze(messages, y_true, y_pred, y_proba)
    analyzer.print_report()


if __name__ == "__main__":
    demo_error_analysis()
