import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.features.build_features import preprocess_text

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        border: none;
    }
    .prediction-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .fake-news {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        border-left: 5px solid #d32f2f;
        color: white;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .real-news {
        background: linear-gradient(135deg, #4caf50, #45a049);
        border-left: 5px solid #2e7d32;
        color: white;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    .confidence-bar {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3.2rem;
        font-size: 1.1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    .stSelectbox > div > div > div {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'fake-news-detector', 'final_model')
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Please ensure the model is trained and saved.")
            return None, None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict_news(text, model, tokenizer, device):
    """Predict whether the given text is fake or real news"""
    if not text.strip():
        return None
    
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize
    inputs = tokenizer(
        preprocessed_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Map prediction to label using model config (0: Fake, 1: Real)
    id2label = getattr(model.config, 'id2label', {0: 'Fake', 1: 'Real'})
    label2id = getattr(model.config, 'label2id', {'Fake': 0, 'Real': 1})
    predicted_label = id2label.get(predicted_class, str(predicted_class))
    
    return {
        "text": text,
        "preprocessed_text": preprocessed_text,
        "prediction": predicted_label,
        "confidence": confidence,
        "probabilities": {
            "Fake": probabilities[0][label2id['Fake']].item(),
            "Real": probabilities[0][label2id['Real']].item()
        }
    }

def create_confidence_chart(probabilities):
    """Create a bar chart showing prediction probabilities"""
    df = pd.DataFrame([
        {"Category": "Real News", "Probability": probabilities["Real"]},
        {"Category": "Fake News", "Probability": probabilities["Fake"]}
    ])
    
    fig = px.bar(
        df, 
        x="Category", 
        y="Probability",
        color="Category",
        color_discrete_map={"Real News": "#4caf50", "Fake News": "#f44336"},
        title="Prediction Confidence",
        height=300
    )
    fig.update_layout(
        yaxis_title="Probability",
        showlegend=False,
        yaxis=dict(range=[0, 1])
    )
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered tool to detect fake news using advanced NLP techniques</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading the AI model..."):
        model, tokenizer, device = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model files exist.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Model Information")
    st.sidebar.info(f"**Device:** {device.upper()}")
    st.sidebar.info("**Model:** BERT-based Transformer")
    st.sidebar.info("**Dataset:** ISOT Fake News Dataset")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Single Text Analysis", "📊 Batch Analysis", "🧪 Test Examples"])
    
    with tab1:
        st.header("Analyze Single News Text")
        
        # Text input
        text_input = st.text_area(
            "Enter the news text you want to analyze:",
            height=150,
            placeholder="Paste your news article or headline here..."
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔍 Analyze Text", type="primary"):
                if text_input.strip():
                    with st.spinner("Analyzing..."):
                        result = predict_news(text_input, model, tokenizer, device)
                    
                    if result:
                        # Display prediction
                        prediction_class = "fake-news" if result["prediction"] == "Fake" else "real-news"
                        st.markdown(f"""
                        <div class="prediction-box {prediction_class}">
                            <h3>Prediction: {result['prediction']}</h3>
                            <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence chart
                        fig = create_confidence_chart(result["probabilities"])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Preprocessed text
                        with st.expander("🔧 View Preprocessed Text"):
                            st.text(result["preprocessed_text"])
                else:
                    st.warning("Please enter some text to analyze.")
        
        with col2:
            st.info("""
            **How it works:**
            1. Enter news text or headline
            2. Click "Analyze Text"
            3. Get instant prediction with confidence score
            4. View detailed probabilities
            """)
    
    with tab2:
        st.header("Batch Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a CSV file with 'text' column:",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column.")
                else:
                    st.success(f"Loaded {len(df)} texts for analysis.")
                    
                    if st.button("🔍 Analyze All Texts", type="primary"):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(df['text']):
                            if pd.notna(text) and str(text).strip():
                                result = predict_news(str(text), model, tokenizer, device)
                                if result:
                                    results.append(result)
                            progress_bar.progress((i + 1) / len(df))
                        
                        if results:
                            # Create results dataframe
                            results_df = pd.DataFrame([
                                {
                                    'Original Text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                                    'Prediction': r['prediction'],
                                    'Confidence': f"{r['confidence']:.1%}",
                                    'Real Probability': f"{r['probabilities']['Real']:.1%}",
                                    'Fake Probability': f"{r['probabilities']['Fake']:.1%}"
                                }
                                for r in results
                            ])
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Texts", len(results))
                            with col2:
                                fake_count = sum(1 for r in results if r['prediction'] == 'Fake')
                                st.metric("Predicted Fake", fake_count)
                            with col3:
                                real_count = sum(1 for r in results if r['prediction'] == 'Real')
                                st.metric("Predicted Real", real_count)
                        
                        progress_bar.empty()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("Test with Example Texts")
        
        # Example texts
        examples = {
            "Fake News Examples": [
                "BREAKING: Scientists discover that drinking hot water with lemon cures all diseases instantly! This revolutionary treatment has been hidden by big pharma for decades.",
                "ALIENS CONFIRMED: NASA admits to covering up evidence of extraterrestrial life on Mars. The truth is finally revealed!",
                "SHOCKING: 5G towers are actually mind control devices installed by the government to control our thoughts!",
                "MIRACLE CURE: This one simple trick will make you lose 50 pounds in a week without diet or exercise!"
            ],
            "Real News Examples": [
                "The World Health Organization released new guidelines for COVID-19 prevention measures based on recent scientific studies.",
                "NASA's Perseverance rover successfully landed on Mars and began its mission to search for signs of ancient life.",
                "Researchers at MIT developed a new algorithm that improves machine learning model efficiency by 40%.",
                "The Federal Reserve announced a 0.25% increase in interest rates following recent economic indicators."
            ]
        }
        
        selected_category = st.selectbox("Choose example category:", list(examples.keys()))
        selected_text = st.selectbox("Choose example text:", examples[selected_category])
        
        if st.button("🔍 Test This Example", type="primary"):
            with st.spinner("Testing..."):
                result = predict_news(selected_text, model, tokenizer, device)
            
            if result:
                # Display prediction
                prediction_class = "fake-news" if result["prediction"] == "Fake" else "real-news"
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    <h3>Prediction: {result['prediction']}</h3>
                    <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence chart
                fig = create_confidence_chart(result["probabilities"])
                st.plotly_chart(fig, use_container_width=True)
                
                # Preprocessed text
                with st.expander("🔧 View Preprocessed Text"):
                    st.text(result["preprocessed_text"])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit and Transformers</p>
        <p>Model trained on ISOT Fake News Dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 