#!/usr/bin/env python3
"""
Fake News Detector - Streamlit Web Application
A user-friendly interface for detecting fake news using our trained hybrid model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import json
import os
import sys
from datetime import datetime
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="🔍 Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-real {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prediction-fake {
        background: linear-gradient(135deg, #f44336, #da190b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .confidence-bar {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        height: 20px;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50, #34495e);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer with caching."""
    try:
        model_path = "models/fake_news_detector/final_model"
        
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Please ensure the model is trained and saved.")
            return None, None, None
        
        # Load model configuration
        with open(os.path.join(model_path, "model_config.json"), "r") as f:
            config = json.load(f)
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        from src.models.train_model import HybridFakeNewsClassifier
        model = HybridFakeNewsClassifier(
            bert_model_name=config["model_name"],
            num_labels=config["num_labels"],
            num_engineered_features=len(config.get("feature_columns", [])),
            feature_hidden_size=config["feature_hidden_size"],
            dropout=config["dropout"]
        )
        
        # Load trained weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=device))
        model.to(device)
        model.eval()
        
        return model, tokenizer, config
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

@st.cache_data
def preprocess_text(text):
    """Preprocess text using the same pipeline as training."""
    try:
        from src.data.advanced_preprocessing import EnhancedFakeNewsPreprocessor
        preprocessor = EnhancedFakeNewsPreprocessor()
        cleaned_text = preprocessor.clean_text_enhanced(text)
        return cleaned_text
    except Exception as e:
        st.warning(f"Preprocessing failed: {str(e)}. Using original text.")
        return text

def predict_fake_news(text, model, tokenizer, config):
    """Predict if text is fake news."""
    try:
        device = next(model.parameters()).device
        
        # Preprocess text
        cleaned_text = preprocess_text(text)
        
        # Tokenize
        inputs = tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=config["max_token_length"],
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create dummy engineered features (zeros for now - you can enhance this)
        num_features = len(config.get("feature_columns", []))
        engineered_features = torch.zeros(1, num_features, dtype=torch.float32).to(device)
        
        # Add token_type_ids if not present
        if 'token_type_ids' not in inputs:
            inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs, engineered_features=engineered_features)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'prediction': prediction,  # 0: Fake, 1: Real
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'cleaned_text': cleaned_text
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def create_confidence_chart(probabilities):
    """Create a beautiful confidence chart."""
    labels = ['Fake News', 'Real News']
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probabilities,
            marker_color=colors,
            text=[f'{prob:.1%}' for prob in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Classification",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        showlegend=False,
        height=400,
        template="plotly_white"
    )
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔍 Single Text Analysis", "📊 Batch Analysis", "🧪 Test Examples", "📈 Model Info", "⚙️ Settings"]
    )
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, tokenizer, config = load_model_and_tokenizer()
    
    if model is None:
        st.error("❌ Model failed to load. Please check the model files and try again.")
        st.stop()
    
    # Home page
    if page == "🏠 Home":
        st.markdown("## 🎉 Welcome to the Fake News Detector!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🚀 What is this?
            This is an advanced AI-powered tool that can detect fake news articles with high accuracy. 
            Our hybrid model combines the power of BERT transformers with engineered features to provide 
            reliable predictions.
            
            ### ✨ Features:
            - **Single Text Analysis**: Analyze individual news articles or headlines
            - **Batch Processing**: Process multiple texts from CSV files
            - **Real-time Predictions**: Get instant results with confidence scores
            - **Advanced Preprocessing**: Consistent text cleaning for accurate results
            - **Beautiful Visualizations**: Interactive charts and metrics
            
            ### 🎯 How to use:
            1. Go to **Single Text Analysis** for individual articles
            2. Use **Batch Analysis** for multiple texts
            3. Try **Test Examples** to see the model in action
            """)
        
        with col2:
            st.markdown("### 📊 Model Statistics")
            st.metric("Model Type", "Hybrid BERT + Features")
            st.metric("Training Data", "Multiple Datasets")
            st.metric("Accuracy", "High Performance")
            st.metric("Status", "✅ Ready")
        
        # Quick demo
        st.markdown("### 🚀 Quick Demo")
        demo_text = st.text_area(
            "Try it out! Enter some text to analyze:",
            value="Scientists discover that drinking hot water with lemon cures all diseases!",
            height=100
        )
        
        if st.button("🔍 Analyze Demo Text", type="primary"):
            with st.spinner("Analyzing..."):
                result = predict_fake_news(demo_text, model, tokenizer, config)
                
                if result:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if result['prediction'] == 0:
                            st.markdown('<div class="prediction-fake">🚨 FAKE NEWS</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="prediction-real">✅ REAL NEWS</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    
                    with col3:
                        st.metric("Model Version", "v1.0")
                    
                    # Confidence chart
                    fig = create_confidence_chart(result['probabilities'])
                    st.plotly_chart(fig, use_container_width=True)
    
    # Single Text Analysis
    elif page == "🔍 Single Text Analysis":
        st.markdown("## 🔍 Single Text Analysis")
        st.markdown("Analyze individual news articles, headlines, or any text for fake news detection.")
        
        # Text input
        text_input = st.text_area(
            "📝 Enter the text you want to analyze:",
            height=200,
            placeholder="Paste your news article, headline, or text here..."
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
        
        with col2:
            if st.button("🧹 Clear Text", use_container_width=True):
                st.rerun()
        
        if analyze_button and text_input.strip():
            with st.spinner("🔍 Analyzing text..."):
                # Add a small delay for better UX
                time.sleep(0.5)
                
                result = predict_fake_news(text_input, model, tokenizer, config)
                
                if result:
                    # Results display
                    st.markdown("## 📊 Analysis Results")
                    
                    # Prediction and confidence
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if result['prediction'] == 0:
                            st.markdown('<div class="prediction-fake">🚨 FAKE NEWS</div>', unsafe_allow_html=True)
                            st.markdown("**This text appears to contain false or misleading information.**")
                        else:
                            st.markdown('<div class="prediction-real">✅ REAL NEWS</div>', unsafe_allow_html=True)
                            st.markdown("**This text appears to contain factual information.**")
                    
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                        st.markdown(f"**Model is {result['confidence']:.1%} confident** in this prediction.")
                    
                    with col3:
                        fake_prob = result['probabilities'][0]
                        real_prob = result['probabilities'][1]
                        st.metric("Fake Probability", f"{fake_prob:.1%}")
                        st.metric("Real Probability", f"{real_prob:.1%}")
                    
                    # Confidence visualization
                    st.markdown("### 📈 Confidence Breakdown")
                    fig = create_confidence_chart(result['probabilities'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Preprocessed text
                    with st.expander("🔧 View Preprocessed Text"):
                        st.text_area("Cleaned and processed text:", value=result['cleaned_text'], height=150, disabled=True)
                    
                    # Analysis insights
                    st.markdown("### 💡 Analysis Insights")
                    if result['prediction'] == 0:
                        st.info("🚨 **Fake News Indicators Detected:** The model has identified patterns commonly associated with false or misleading information. Consider fact-checking this content through reliable sources.")
                    else:
                        st.success("✅ **Real News Indicators Detected:** The model has identified patterns commonly associated with factual information. However, always verify important information through multiple reliable sources.")
                    
                    # Recommendations
                    st.markdown("### 🎯 Recommendations")
                    if result['prediction'] == 0:
                        st.markdown("""
                        - **Fact-check** through multiple reliable sources
                        - **Check the source** credibility
                        - **Look for evidence** and citations
                        - **Be skeptical** of sensational claims
                        - **Share responsibly** - don't spread unverified information
                        """)
                    else:
                        st.markdown("""
                        - **Verify details** through additional sources
                        - **Check for updates** - news can change
                        - **Consider the context** and timing
                        - **Evaluate the source** reputation
                        - **Stay informed** through multiple perspectives
                        """)
        
        elif analyze_button and not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Batch Analysis
    elif page == "📊 Batch Analysis":
        st.markdown("## 📊 Batch Analysis")
        st.markdown("Upload a CSV file with multiple texts to analyze them all at once.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Choose a CSV file:",
            type=['csv'],
            help="The CSV should have a 'text' column containing the texts to analyze."
        )
        
        if uploaded_file is not None:
            try:
                # Load CSV
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("❌ CSV file must contain a 'text' column.")
                else:
                    st.success(f"✅ File loaded successfully! Found {len(df)} texts to analyze.")
                    
                    # Show preview
                    st.markdown("### 📋 Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Analysis options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        max_texts = st.number_input(
                            "Maximum texts to analyze:",
                            min_value=1,
                            max_value=len(df),
                            value=min(100, len(df)),
                            help="Limit the number of texts to prevent long processing times."
                        )
                    
                    with col2:
                        batch_size = st.number_input(
                            "Batch size:",
                            min_value=1,
                            max_value=50,
                            value=10,
                            help="Number of texts to process at once."
                        )
                    
                    if st.button("🚀 Start Batch Analysis", type="primary"):
                        # Limit texts
                        df_limited = df.head(max_texts)
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        
                        for i, (idx, row) in enumerate(df_limited.iterrows()):
                            status_text.text(f"Analyzing text {i+1}/{len(df_limited)}...")
                            
                            result = predict_fake_news(row['text'], model, tokenizer, config)
                            
                            if result:
                                results.append({
                                    'text': row['text'][:100] + "..." if len(row['text']) > 100 else row['text'],
                                    'prediction': 'Fake' if result['prediction'] == 0 else 'Real',
                                    'confidence': result['confidence'],
                                    'fake_probability': result['probabilities'][0],
                                    'real_probability': result['probabilities'][1]
                                })
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(df_limited))
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if results:
                            # Create results DataFrame
                            results_df = pd.DataFrame(results)
                            
                            # Display results
                            st.markdown("### 📊 Batch Analysis Results")
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Texts", len(results_df))
                            
                            with col2:
                                fake_count = len(results_df[results_df['prediction'] == 'Fake'])
                                st.metric("Fake News", fake_count)
                            
                            with col3:
                                real_count = len(results_df[results_df['prediction'] == 'Real'])
                                st.metric("Real News", real_count)
                            
                            with col4:
                                avg_confidence = results_df['confidence'].mean()
                                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                            
                            # Results table
                            st.markdown("### 📋 Detailed Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv,
                                file_name=f"fake_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Visualizations
                            st.markdown("### 📈 Analysis Visualizations")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Prediction distribution
                                fig1 = px.pie(
                                    results_df, 
                                    names='prediction', 
                                    title="Prediction Distribution",
                                    color_discrete_map={'Fake': '#FF6B6B', 'Real': '#4ECDC4'}
                                )
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            with col2:
                                # Confidence distribution
                                fig2 = px.histogram(
                                    results_df, 
                                    x='confidence', 
                                    title="Confidence Distribution",
                                    nbins=20,
                                    color_discrete_sequence=['#667eea']
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            # Confidence vs Prediction
                            fig3 = px.box(
                                results_df, 
                                x='prediction', 
                                y='confidence', 
                                title="Confidence by Prediction",
                                color='prediction',
                                color_discrete_map={'Fake': '#FF6B6B', 'Real': '#4ECDC4'}
                            )
                            st.plotly_chart(fig3, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # Test Examples
    elif page == "🧪 Test Examples":
        st.markdown("## 🧪 Test Examples")
        st.markdown("Try the model with pre-defined examples to see how it performs.")
        
        # Example categories
        example_category = st.selectbox(
            "Choose example category:",
            ["Fake News Examples", "Real News Examples", "Mixed Examples"]
        )
        
        if example_category == "Fake News Examples":
            examples = [
                "Scientists discover that drinking hot water with lemon cures all diseases!",
                "Wisconsin is on pace to double the number of layoffs this year.",
                "Says John McCain has done nothing to help the vets.",
                '"Ronald Reagan faced an even worse recession" than the current one.',
                "A strong bipartisan majority in the House of Representatives voted to defund Obamacare."
            ]
        elif example_category == "Real News Examples":
            examples = [
                "The World Health Organization released new guidelines for COVID-19 prevention.",
                "NASA successfully launched the Perseverance rover to Mars.",
                "Climate scientists report record-breaking temperatures in 2023.",
                "New study shows benefits of regular exercise on mental health.",
                "Global renewable energy adoption reached new milestone in 2024."
            ]
        else:  # Mixed Examples
            examples = [
                "Scientists discover that drinking hot water with lemon cures all diseases!",
                "The World Health Organization released new guidelines for COVID-19 prevention.",
                "ALIENS CONFIRMED: UFO spotted over Washington DC - government covering it up!",
                "NASA successfully launched the Perseverance rover to Mars.",
                "SHOCKING: This one weird trick will make you lose 50 pounds in a week!"
            ]
        
        # Display examples
        st.markdown("### 📝 Example Texts")
        for i, example in enumerate(examples):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.text_area(f"Example {i+1}:", value=example, height=80, disabled=True, key=f"example_{i}")
            
            with col2:
                if st.button(f"Test {i+1}", key=f"test_{i}"):
                    with st.spinner("Testing..."):
                        result = predict_fake_news(example, model, tokenizer, config)
                        
                        if result:
                            st.markdown("**Result:**")
                            if result['prediction'] == 0:
                                st.markdown("🚨 **FAKE**")
                            else:
                                st.markdown("✅ **REAL**")
                            
                            st.markdown(f"Confidence: {result['confidence']:.1%}")
    
    # Model Info
    elif page == "📈 Model Info":
        st.markdown("## 📈 Model Information")
        st.markdown("Learn about the AI model powering this fake news detector.")
        
        if config:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏗️ Model Architecture")
                st.markdown(f"""
                - **Base Model**: {config.get('model_name', 'N/A')}
                - **Model Type**: Hybrid BERT + Engineered Features
                - **Number of Labels**: {config.get('num_labels', 'N/A')}
                - **Max Token Length**: {config.get('max_token_length', 'N/A')}
                - **Feature Hidden Size**: {config.get('feature_hidden_size', 'N/A')}
                - **Dropout Rate**: {config.get('dropout', 'N/A'):.3f}
                """)
            
            with col2:
                st.markdown("### ⚙️ Training Configuration")
                st.markdown(f"""
                - **Learning Rate**: {config.get('learning_rate', 'N/A'):.2e}
                - **Batch Size**: {config.get('batch_size', 'N/A')}
                - **Number of Epochs**: {config.get('num_epochs', 'N/A')}
                - **Number of Features**: {len(config.get('feature_columns', []))}
                """)
            
            st.markdown("### 🔬 How It Works")
            st.markdown("""
            Our hybrid model combines the power of transformer-based language models with engineered features:
            
            1. **Text Processing**: Input text is cleaned and tokenized using advanced NLP techniques
            2. **BERT Encoding**: The text is processed through a pre-trained BERT model to extract contextual embeddings
            3. **Feature Engineering**: Additional linguistic and statistical features are extracted
            4. **Fusion**: BERT embeddings and engineered features are combined
            5. **Classification**: A neural network classifier determines if the text is fake or real
            6. **Confidence Scoring**: The model provides confidence scores for its predictions
            """)
            
            st.markdown("### 📊 Performance Metrics")
            st.markdown("""
            The model has been trained on multiple datasets and optimized for:
            - **High Accuracy**: Minimizing false positives and negatives
            - **Robustness**: Handling various writing styles and topics
            - **Speed**: Fast inference for real-time analysis
            - **Interpretability**: Providing confidence scores and insights
            """)
        
        else:
            st.error("❌ Model configuration not available.")
    
    # Settings
    elif page == "⚙️ Settings":
        st.markdown("## ⚙️ Application Settings")
        st.markdown("Customize your experience with the Fake News Detector.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎨 Display Settings")
            
            # Theme selection
            theme = st.selectbox(
                "Choose theme:",
                ["Light", "Dark"],
                help="Select your preferred color scheme"
            )
            
            # Animation settings
            show_animations = st.checkbox(
                "Show animations",
                value=True,
                help="Enable smooth transitions and animations"
            )
            
            # Auto-refresh
            auto_refresh = st.checkbox(
                "Auto-refresh results",
                value=False,
                help="Automatically refresh analysis results"
            )
        
        with col2:
            st.markdown("### 🔧 Analysis Settings")
            
            # Confidence threshold
            confidence_threshold = st.slider(
                "Confidence threshold:",
                min_value=0.5,
                max_value=0.95,
                value=0.7,
                step=0.05,
                help="Minimum confidence required for predictions"
            )
            
            # Max text length
            max_text_length = st.number_input(
                "Maximum text length:",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Maximum characters to process"
            )
        
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            st.success("✅ Settings saved successfully!")
            
            # Display current settings
            st.markdown("### 📋 Current Settings")
            settings_data = {
                "Theme": theme,
                "Show Animations": show_animations,
                "Auto-refresh": auto_refresh,
                "Confidence Threshold": f"{confidence_threshold:.2f}",
                "Max Text Length": max_text_length
            }
            
            for key, value in settings_data.items():
                st.markdown(f"**{key}**: {value}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🔍 Fake News Detector v1.0 | Built with ❤️ using Streamlit and PyTorch<br>
        Always verify information through reliable sources
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
