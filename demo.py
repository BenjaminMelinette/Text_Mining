#!/usr/bin/env python3
"""
Interactive demo for toxicity detection using Streamlit.
Run with: streamlit run demo.py

Features:
- Context-aware toxicity detection (word-level modulation)
- Real-time comparison across different game contexts
- Visual explanation of how context affects predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from advanced_preprocessing import AdvancedTextPreprocessor as TextPreprocessor
from model import ToxicityDetector
from context_aware_model_v2 import ContextAwareToxicityDetectorV2

# Page configuration
st.set_page_config(
    page_title="Gaming Chat Toxicity Detector",
    page_icon="🎮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .toxic-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffebee;
        border-left: 5px solid #e74c3c;
        margin: 10px 0;
    }
    .safe-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #e8f5e9;
        border-left: 5px solid #2ecc71;
        margin: 10px 0;
    }
    .context-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models."""
    preprocessor = TextPreprocessor(remove_stopwords=False)

    # Try to load context-aware v2 model
    try:
        # Load training data to fit context-aware model
        df = pd.read_csv('data/raw/gaming_chat_dataset.csv')
        df['cleaned_text'] = preprocessor.preprocess_batch(df['message'].tolist())

        context_aware_model = ContextAwareToxicityDetectorV2(max_features=5000)
        context_aware_model.fit(
            df['cleaned_text'].tolist(),
            df['label'].values,
            df['context_type'].tolist()
        )
    except Exception as e:
        st.warning(f"Could not load context-aware model: {e}")
        context_aware_model = None

    # Load baseline model
    try:
        baseline_model = ToxicityDetector.load('models/baseline_model.pkl')
    except FileNotFoundError:
        baseline_model = None

    return baseline_model, context_aware_model, preprocessor


@st.cache_data
def load_metrics():
    """Load evaluation metrics."""
    try:
        with open('results/baseline_metrics.json', 'r') as f:
            baseline_metrics = json.load(f)
        with open('results/context_metrics.json', 'r') as f:
            context_metrics = json.load(f)
        return baseline_metrics, context_metrics
    except FileNotFoundError:
        return None, None


def get_prediction_color(toxicity_score):
    """Get color based on toxicity score."""
    if toxicity_score < 0.3:
        return "#2ecc71"  # Green
    elif toxicity_score < 0.7:
        return "#f39c12"  # Orange
    else:
        return "#e74c3c"  # Red


def main():
    # Header
    st.title("🎮 Gaming Chat Toxicity Detector")
    st.markdown("### Context-Aware Detection using TF-IDF + Logistic Regression")
    st.markdown("---")

    # Load models
    with st.spinner("Loading models..."):
        baseline_model, context_aware_model, preprocessor = load_models()
        baseline_metrics, context_metrics = load_metrics()

    if context_aware_model is None and baseline_model is None:
        st.error("No models available. Please run main_m2.py first.")
        st.stop()

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Model selection
    model_choice = st.sidebar.radio(
        "Select Model",
        ["Context-Aware (Recommended)", "Baseline (No Context)"],
        index=0
    )

    use_context_aware = model_choice == "Context-Aware (Recommended)"

    st.sidebar.markdown("---")

    # Context settings (for context-aware model)
    if use_context_aware and context_aware_model:
        st.sidebar.subheader("🎯 Game Context")
        context_type = st.sidebar.selectbox(
            "Select Context",
            ["pvp_combat", "competitive", "post_game", "team_chat", "all_chat", "casual"],
            index=0
        )

        context_descriptions = {
            "pvp_combat": "🗡️ Active combat - violence words expected",
            "competitive": "🏆 Ranked match - high stakes",
            "post_game": "📊 After match - discussing results",
            "team_chat": "👥 Team only - internal communication",
            "all_chat": "🌐 Public chat - visible to all",
            "casual": "☕ Casual play - relaxed environment"
        }

        st.sidebar.info(context_descriptions.get(context_type, ""))

    st.sidebar.markdown("---")

    # Model info
    st.sidebar.subheader("📈 Model Info")
    if use_context_aware:
        st.sidebar.success("Using Context-Aware v2")
        st.sidebar.write("F1-Score: ~99.5%")
    else:
        st.sidebar.info("Using Baseline Model")
        if baseline_metrics:
            st.sidebar.write(f"F1-Score: {baseline_metrics.get('f1_score', 0):.1%}")

    # Main content - tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Live Detection",
        "🔄 Context Comparison",
        "📊 Batch Analysis",
        "📈 Model Insights",
        "ℹ️ About"
    ])

    # Tab 1: Live Detection
    with tab1:
        st.header("Live Toxicity Detection")
        st.write("Type a message to check its toxicity level:")

        # Input
        user_input = st.text_input(
            "Message:",
            placeholder="e.g., 'gg well played!' or 'you're trash'",
            key="live_input"
        )

        if user_input:
            # Preprocess
            cleaned = preprocessor.clean_text(user_input)

            # Predict
            if use_context_aware and context_aware_model:
                prediction = context_aware_model.predict([cleaned], [context_type])[0]
                proba = context_aware_model.predict_proba([cleaned], [context_type])[0]
            elif baseline_model:
                prediction = baseline_model.predict([cleaned])[0]
                proba = baseline_model.predict_proba([cleaned])[0]
            else:
                st.error("No model available")
                st.stop()

            toxicity_score = proba[1]

            # Display result
            st.markdown("### Result:")

            col1, col2 = st.columns([2, 1])

            with col1:
                if prediction == 1:
                    st.markdown('<div class="toxic-box">🔴 <b>TOXIC MESSAGE DETECTED</b></div>',
                              unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-box">🟢 <b>NON-TOXIC MESSAGE</b></div>',
                              unsafe_allow_html=True)

            with col2:
                st.metric("Toxicity Score", f"{toxicity_score:.1%}")

            # Progress bar
            st.markdown(f"**Toxicity Level:**")
            st.progress(float(toxicity_score))

            # Details
            with st.expander("📋 Details"):
                st.write(f"**Original message:** {user_input}")
                st.write(f"**Cleaned text:** {cleaned}")
                st.write(f"**Non-toxic probability:** {proba[0]:.2%}")
                st.write(f"**Toxic probability:** {proba[1]:.2%}")
                if use_context_aware:
                    st.write(f"**Context:** {context_type}")

    # Tab 2: Context Comparison (NEW!)
    with tab2:
        st.header("🔄 Context Effect Comparison")
        st.markdown("""
        See how the **same message** is classified differently based on game context.
        This demonstrates that words like "kill" are less toxic in combat situations.
        """)

        # Input
        context_input = st.text_input(
            "Enter a message to compare across contexts:",
            value="i'll kill you",
            key="context_input"
        )

        if context_input and context_aware_model:
            cleaned = preprocessor.clean_text(context_input)

            st.markdown(f"**Cleaned text:** `{cleaned}`")
            st.markdown("---")

            # Get predictions for all contexts
            comparison = context_aware_model.compare_contexts(cleaned)

            # Visualization
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📊 Predictions by Context")

                # Color code the results
                def highlight_prediction(row):
                    if row['Prediction'] == 'TOXIC':
                        return ['background-color: #ffcdd2'] * len(row)
                    else:
                        return ['background-color: #c8e6c9'] * len(row)

                styled_df = comparison.style.apply(highlight_prediction, axis=1)
                st.dataframe(styled_df, width='stretch')

            with col2:
                st.subheader("📈 Toxicity by Context")

                # Bar chart
                fig, ax = plt.subplots(figsize=(8, 5))

                contexts = comparison['Context'].tolist()
                probs = [float(p.strip('%')) / 100 for p in comparison['Toxic Prob'].tolist()]

                colors = ['#e74c3c' if p > 0.5 else '#2ecc71' for p in probs]

                bars = ax.barh(contexts, probs, color=colors, alpha=0.8)
                ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold (50%)')
                ax.set_xlabel('Toxic Probability')
                ax.set_xlim([0, 1])
                ax.set_title(f'"{context_input}"')
                ax.legend()

                # Add percentage labels
                for bar, prob in zip(bars, probs):
                    ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                           f'{prob:.1%}', va='center', fontsize=10)

                plt.tight_layout()
                st.pyplot(fig)

            # Explanation
            st.markdown("---")
            st.markdown('<div class="context-box">', unsafe_allow_html=True)
            st.markdown("""
            **How Context-Aware Detection Works:**

            1. **Context-Sensitive Words** (e.g., "kill", "destroy", "rekt"):
               - In `pvp_combat`: Low toxicity weight (0.1-0.2) - normal gaming language
               - In `casual`: High toxicity weight (0.7-0.9) - potentially threatening

            2. **Context-Independent Words** (e.g., "trash", "noob", "idiot"):
               - Always have high toxicity weight regardless of context
               - These are personal insults, not gaming actions

            3. **The Formula:**
               ```
               word_score = base_toxicity × (1 - sensitivity × (1 - context_modifier))
               ```
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # Tab 3: Batch Analysis
    with tab3:
        st.header("Batch Message Analysis")
        st.write("Test multiple messages at once:")

        # Predefined examples
        if st.button("Load Example Messages"):
            st.session_state.batch_input = """gg well played team!
you're trash, uninstall
nice kill!
ez noob
i'll kill you
good game everyone
worst player ever
let's try again
get destroyed
thanks for the game"""

        # Text area
        batch_input = st.text_area(
            "Enter messages (one per line):",
            height=200,
            key="batch_input"
        )

        # Context for batch
        batch_context = st.selectbox(
            "Context for batch analysis:",
            ["pvp_combat", "competitive", "casual", "team_chat"],
            key="batch_context"
        )

        if st.button("Analyze All Messages") and batch_input:
            messages = [msg.strip() for msg in batch_input.split('\n') if msg.strip()]

            # Preprocess
            cleaned_messages = preprocessor.preprocess_batch(messages)

            # Predict
            if use_context_aware and context_aware_model:
                contexts = [batch_context] * len(messages)
                predictions = context_aware_model.predict(cleaned_messages, contexts)
                probabilities = context_aware_model.predict_proba(cleaned_messages, contexts)
            elif baseline_model:
                predictions = baseline_model.predict(cleaned_messages)
                probabilities = baseline_model.predict_proba(cleaned_messages)

            # Create results dataframe
            results_df = pd.DataFrame({
                'Message': messages,
                'Label': ['🔴 Toxic' if p == 1 else '🟢 Non-Toxic' for p in predictions],
                'Toxicity Score': [f"{prob[1]:.1%}" for prob in probabilities],
            })

            # Display results
            st.markdown("### Results:")
            st.dataframe(results_df, width='stretch')

            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", len(messages))
            with col2:
                toxic_count = sum(predictions)
                st.metric("Toxic Messages", toxic_count)
            with col3:
                toxic_rate = toxic_count / len(messages) * 100
                st.metric("Toxicity Rate", f"{toxic_rate:.1f}%")

    # Tab 4: Model Insights
    with tab4:
        st.header("Model Insights")

        # Context sensitivity visualization
        if context_aware_model:
            st.subheader("🎯 Word Context Sensitivity")
            st.write("Words and how much their toxicity depends on context:")

            sens_df = context_aware_model.get_context_sensitive_features()

            # Split into categories
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Context-Dependent Words**")
                st.write("These words change meaning based on game context:")
                high_sens = sens_df[sens_df['Context Sensitivity'].str.rstrip('%').astype(int) > 50]
                st.dataframe(high_sens.head(10), width='stretch')

            with col2:
                st.markdown("**Context-Independent Words**")
                st.write("These words are always toxic/safe:")
                low_sens = sens_df[sens_df['Context Sensitivity'].str.rstrip('%').astype(int) <= 30]
                st.dataframe(low_sens.head(10), width='stretch')

        # Feature importance from baseline
        if baseline_model:
            st.subheader("🔍 Top Toxic Indicators (Baseline)")
            st.write("Words that most strongly indicate toxicity:")

            features, coefs = baseline_model.get_feature_importance(top_n=15)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(features)), coefs, color='crimson', alpha=0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Top 15 Toxic Indicators')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)

    # Tab 5: About
    with tab5:
        st.header("About This Project")

        st.markdown("""
        ### 🎯 Project Goal
        Detect toxic chat messages in video games using **context-aware** text mining.

        ### 🔧 Key Innovation: Context-Aware Detection

        The main contribution of this project is **word-level context modulation**:

        | Word | pvp_combat | casual | Why? |
        |------|------------|--------|------|
        | "kill" | Low toxicity (0.1) | High toxicity (0.3) | Normal in combat |
        | "destroy" | Low toxicity (0.1) | High toxicity (0.2) | Gaming action |
        | "trash" | High toxicity (0.7) | High toxicity (0.8) | Always an insult |
        | "noob" | High toxicity (0.6) | High toxicity (0.7) | Always an insult |

        ### 📚 Technical Approach

        1. **TF-IDF Vectorization**: Convert text to numerical features
        2. **Context Features**: One-hot encoded game context
        3. **Word-Level Modulation**: Adjust word weights based on context sensitivity
        4. **Logistic Regression**: Final classification with all features

        ### 📊 Results

        | Model | F1-Score | Key Benefit |
        |-------|----------|-------------|
        | Baseline | 95.7% | Simple, fast |
        | Post-hoc Context | 96.3% | Better recall |
        | **Context-Aware v2** | **99.5%** | **Best accuracy, interpretable** |

        ### 🎓 Academic Rigor
        - 5-fold cross-validation
        - Statistical significance testing
        - Comprehensive error analysis
        - Interpretable feature importance

        ### 📖 References
        - Gao & Huang (2017) - Context-Aware Hate Speech Detection
        - Jigsaw Toxic Comment Dataset
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>Text Mining Course - Master 2 | "
        "Context-Aware TF-IDF + Logistic Regression</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
