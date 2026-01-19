#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from advanced_preprocessing import AdvancedTextPreprocessor
from model import ToxicityDetector
from context_aware_model import ContextAwareToxicityDetector
from context_aware_model_v2 import ContextAwareToxicityDetectorV2


def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def main():
    print_header("COMPLETE MODEL COMPARISON: CONTEXT-AWARE APPROACHES")

    # ========================================
    # Load and preprocess data
    # ========================================
    print("\n1. Loading and preprocessing data...")
    df = pd.read_csv('data/raw/gaming_chat_dataset.csv')

    preprocessor = AdvancedTextPreprocessor(remove_stopwords=False)
    df['cleaned_text'] = preprocessor.preprocess_batch(df['message'].tolist())

    X = df['cleaned_text'].values
    y = df['label'].values
    contexts = df['context_type'].values
    context_scores = df['context_score'].values

    # Split
    X_train, X_test, y_train, y_test, ctx_train, ctx_test, scores_train, scores_test = train_test_split(
        X, y, contexts, context_scores, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")

    # ========================================
    # Train all models
    # ========================================
    print_header("MODEL TRAINING")

    models = {}

    # Model 1: Baseline
    print("\n→ Training Model 1: Baseline (no context)...")
    models['Baseline'] = ToxicityDetector(max_features=5000, use_context=False)
    models['Baseline'].train(X_train.tolist(), y_train)

    # Model 2: Post-hoc adjustment
    print("\n→ Training Model 2: Post-hoc context adjustment...")
    models['Post-hoc'] = ToxicityDetector(max_features=5000, use_context=True, context_weight=0.3)
    models['Post-hoc'].train(X_train.tolist(), y_train, scores_train.tolist())

    # Model 3: Context-aware v1 (feature modulation)
    print("\n→ Training Model 3: Context-aware v1 (feature modulation)...")
    models['Context-v1'] = ContextAwareToxicityDetector(max_features=5000)
    models['Context-v1'].fit(X_train.tolist(), y_train, ctx_train.tolist())

    # Model 4: Context-aware v2 (word-level + context features)
    print("\n→ Training Model 4: Context-aware v2 (word-level + context features)...")
    models['Context-v2'] = ContextAwareToxicityDetectorV2(max_features=5000)
    models['Context-v2'].fit(X_train.tolist(), y_train, ctx_train.tolist())

    # ========================================
    # Evaluate on test set
    # ========================================
    print_header("TEST SET EVALUATION")

    results = []

    # Baseline
    y_pred = models['Baseline'].predict(X_test.tolist())
    results.append({
        'Model': 'Baseline (no context)',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })

    # Post-hoc
    y_pred = models['Post-hoc'].predict(X_test.tolist(), scores_test.tolist())
    results.append({
        'Model': 'Post-hoc adjustment',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })

    # Context-v1
    y_pred = models['Context-v1'].predict(X_test.tolist(), ctx_test.tolist())
    results.append({
        'Model': 'Context-aware v1 (feature mod)',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })

    # Context-v2
    y_pred = models['Context-v2'].predict(X_test.tolist(), ctx_test.tolist())
    results.append({
        'Model': 'Context-aware v2 (word-level)',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    # ========================================
    # Context Effect Demonstration
    # ========================================
    print_header("CONTEXT EFFECT DEMONSTRATION")

    test_messages = [
        "i'll kill you",
        "get destroyed noob",
        "you're trash",
        "nice shot",
        "gg ez",
    ]

    print("\nHow predictions change with context (Context-aware v2):\n")

    for msg in test_messages:
        cleaned = preprocessor.clean_text(msg)
        print(f"Message: \"{msg}\" → \"{cleaned}\"")
        print("-" * 70)

        comparison = models['Context-v2'].compare_contexts(cleaned)
        # Show only key contexts
        key_contexts = ['pvp_combat', 'casual', 'competitive']
        filtered = comparison[comparison['Context'].isin(key_contexts)]
        print(filtered.to_string(index=False))
        print()

    # ========================================
    # Analyze by context type
    # ========================================
    print_header("PERFORMANCE BY CONTEXT TYPE")

    context_results = []

    for ctx_type in df['context_type'].unique():
        mask = ctx_test == ctx_type
        if sum(mask) > 10:  # Only if enough samples
            X_ctx = X_test[mask]
            y_ctx = y_test[mask]
            ctx_ctx = ctx_test[mask]
            scores_ctx = scores_test[mask]

            # Baseline
            y_pred_baseline = models['Baseline'].predict(X_ctx.tolist())

            # Context-v2
            y_pred_v2 = models['Context-v2'].predict(X_ctx.tolist(), ctx_ctx.tolist())

            context_results.append({
                'Context': ctx_type,
                'N': sum(mask),
                'Baseline F1': f1_score(y_ctx, y_pred_baseline, zero_division=0),
                'Context-v2 F1': f1_score(y_ctx, y_pred_v2, zero_division=0),
                'Improvement': f1_score(y_ctx, y_pred_v2, zero_division=0) - f1_score(y_ctx, y_pred_baseline, zero_division=0)
            })

    ctx_df = pd.DataFrame(context_results)
    ctx_df = ctx_df.sort_values('Improvement', ascending=False)
    print("\n" + ctx_df.to_string(index=False))

    # ========================================
    # Save results
    # ========================================
    print_header("SAVING RESULTS")

    results_df.to_csv('results/context_model_comparison.csv', index=False)
    ctx_df.to_csv('results/performance_by_context.csv', index=False)

    print("✓ Results saved to results/context_model_comparison.csv")
    print("✓ Performance by context saved to results/performance_by_context.csv")

    # ========================================
    # Visualization
    # ========================================
    print_header("GENERATING VISUALIZATION")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Model comparison
    ax = axes[0]
    x = np.arange(len(results_df))
    width = 0.2
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, results_df[metric], width, label=metric, color=colors[i], alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Baseline', 'Post-hoc', 'Context-v1', 'Context-v2'], rotation=15)
    ax.legend(loc='lower right')
    ax.set_ylim([0.85, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Context effect on "kill" word
    ax = axes[1]
    contexts_order = ['pvp_combat', 'competitive', 'post_game', 'team_chat', 'all_chat', 'casual']
    test_msg = "i'll kill you"
    cleaned = preprocessor.clean_text(test_msg)

    probs = []
    for ctx in contexts_order:
        p = models['Context-v2'].predict_proba([cleaned], [ctx])[0][1]
        probs.append(p)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(contexts_order)))
    ax.barh(contexts_order, probs, color=colors)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Toxic Probability')
    ax.set_title(f'Context Effect: "{test_msg}"', fontweight='bold')
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig('results/visualizations/context_effect.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to results/visualizations/context_effect.png")
    plt.show()

    # ========================================
    # Summary
    # ========================================
    print_header("KEY FINDINGS")

    print("""
1. CONTEXT MATTERS: The same message can be toxic or non-toxic depending
   on the game context.

2. WORD-LEVEL MODULATION: Words like "kill" and "destroy" have different
   toxicity weights in combat vs casual contexts.

3. INSULTS ARE ALWAYS TOXIC: Words like "trash", "noob", "idiot" remain
   toxic regardless of context.

4. BEST APPROACH: Context-aware v2 (word-level modulation) provides the
   most intuitive and explainable results.

5. TRADE-OFF: More sophisticated context handling increases complexity
   but provides more nuanced classification.
""")


if __name__ == "__main__":
    main()
