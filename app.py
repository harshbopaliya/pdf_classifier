import os
import streamlit as st
import tempfile
import shutil
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import json
import logging

# Import the improved classifier
from classifier import DocumentClassifier, set_seed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Page Configuration ===
st.set_page_config(
    page_title="📄 PDF Document Classifier",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown("""
<div class="main-header">
    <h1>📑 AI Document Classifier</h1>
    <p>Advanced PDF classification using LayoutLMv3</p>
    <p><em>Classify documents into: Binder • Contract • Quotes • Policy</em></p>
</div>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_path = st.text_input("Model Path", value="./results", help="Path to the trained model")
    
    # Confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    
    # Statistics section
    st.header("📊 Session Statistics")
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = 0
        st.session_state.prediction_history = []
    
    st.metric("Predictions Made", st.session_state.predictions_made)
    
    if st.session_state.prediction_history:
        avg_confidence = sum([p['confidence'] for p in st.session_state.prediction_history]) / len(st.session_state.prediction_history)
        st.metric("Average Confidence", f"{avg_confidence:.2%}")
    
    st.markdown("---")
    
    # Quick actions
    st.header("🔧 Quick Actions")
    if st.button("Clear History"):
        st.session_state.predictions_made = 0
        st.session_state.prediction_history = []
        st.success("History cleared!")
    
    if st.button("Download History"):
        if st.session_state.prediction_history:
            df = pd.DataFrame(st.session_state.prediction_history)
            st.download_button(
                label="📥 Download CSV",
                data=df.to_csv(index=False),
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# === Model Loading ===
@st.cache_resource
def load_classifier():
    """Load the classifier with error handling"""
    try:
        set_seed(42)  # Ensure reproducibility
        classifier = DocumentClassifier()
        return classifier, None
    except Exception as e:
        return None, str(e)

classifier, load_error = load_classifier()

if load_error:
    st.error(f"❌ Failed to load classifier: {load_error}")
    st.stop()

# === Model Status Check ===
def check_model_status(model_path):
    required_files = [
        'config.json',
        'label_mappings.json',
        'model.safetensors'  # Accept safetensors as valid
    ]
    return all(os.path.exists(os.path.join(model_path, f)) for f in required_files)


model_exists = check_model_status(model_path)

if not model_exists:
    st.warning("⚠️ No trained model found. Please train a model first or check the model path.")
    st.info("💡 Use the training section below to train a new model.")

# === Main Content Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict Documents", "📈 Batch Processing", "🏋️ Train Model", "📊 Analytics"])

# === TAB 1: Single Document Prediction ===
with tab1:
    st.header("📄 Single Document Classification")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a PDF file to classify it automatically"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"✅ PDF uploaded: {uploaded_file.name}")
            file_size = len(uploaded_file.read()) / 1024  # KB
            uploaded_file.seek(0)  # Reset file pointer
            st.info(f"File size: {file_size:.1f} KB")
        
        with col2:
            if st.button("🚀 Classify Document", type="primary"):
                if model_exists:
                    with st.spinner("🔄 Processing document..."):
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            temp_path = tmp_file.name
                        
                        try:
                            # Make prediction
                            result = classifier.predict(temp_path, model_path)
                            
                            # Clean up
                            os.unlink(temp_path)
                            
                            if "error" not in result:
                                # Display results
                                predicted_class = result['predicted_class']
                                confidence = result['confidence']
                                
                                # Determine confidence level
                                if confidence >= 0.8:
                                    conf_class = "confidence-high"
                                    conf_emoji = "🟢"
                                elif confidence >= 0.5:
                                    conf_class = "confidence-medium"
                                    conf_emoji = "🟡"
                                else:
                                    conf_class = "confidence-low"
                                    conf_emoji = "🔴"
                                
                                # Main prediction display
                                st.markdown(f"""
                                <div class="prediction-box">
                                    <h2>🎯 Prediction Result</h2>
                                    <h1>{predicted_class.upper()}</h1>
                                    <p>Confidence: <span class="{conf_class}">{confidence:.1%}</span> {conf_emoji}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Detailed scores
                                st.subheader("📊 Detailed Confidence Scores")
                                
                                # Create interactive bar chart
                                scores_df = pd.DataFrame([
                                    {"Document Type": k.capitalize(), "Confidence": v}
                                    for k, v in result['all_scores'].items()
                                ])
                                
                                fig = px.bar(
                                    scores_df,
                                    x="Document Type",
                                    y="Confidence",
                                    color="Confidence",
                                    color_continuous_scale="viridis",
                                    title="Classification Confidence by Document Type"
                                )
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Additional info
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Words Extracted", result.get('words_extracted', 'N/A'))
                                with col2:
                                    st.metric("Processing Time", "< 1s")
                                with col3:
                                    above_threshold = confidence >= confidence_threshold
                                    st.metric("Above Threshold", "✅ Yes" if above_threshold else "❌ No")
                                
                                # Save to history
                                st.session_state.predictions_made += 1
                                st.session_state.prediction_history.append({
                                    'filename': uploaded_file.name,
                                    'predicted_class': predicted_class,
                                    'confidence': confidence,
                                    'timestamp': datetime.now().isoformat(),
                                    'words_extracted': result.get('words_extracted', 0)
                                })
                                
                                # Warning for low confidence
                                if confidence < confidence_threshold:
                                    st.warning(f"⚠️ Low confidence prediction. Consider manual review.")
                                
                            else:
                                st.error(f"❌ Prediction failed: {result['error']}")
                                
                        except Exception as e:
                            st.error(f"❌ An error occurred: {str(e)}")
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                else:
                    st.error("❌ No trained model available for prediction.")

# === TAB 2: Batch Processing ===
with tab2:
    st.header("📁 Batch Document Processing")
    
    batch_folder = st.text_input("Folder Path", value="./batch_pdfs", help="Path to folder containing PDF files")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Scan Folder"):
            if os.path.exists(batch_folder):
                pdf_files = [f for f in os.listdir(batch_folder) if f.lower().endswith('.pdf')]
                st.success(f"Found {len(pdf_files)} PDF files")
                st.session_state.batch_files = pdf_files
                
                if pdf_files:
                    with st.expander("📋 File List"):
                        for i, file in enumerate(pdf_files, 1):
                            st.write(f"{i}. {file}")
            else:
                st.error("❌ Folder does not exist")
    
    with col2:
        if st.button("⚡ Process All Files") and model_exists:
            if hasattr(st.session_state, 'batch_files') and st.session_state.batch_files:
                progress_bar = st.progress(0)
                results = []
                
                for i, filename in enumerate(st.session_state.batch_files):
                    file_path = os.path.join(batch_folder, filename)
                    
                    try:
                        result = classifier.predict(file_path, model_path)
                        if "error" not in result:
                            results.append({
                                'Filename': filename,
                                'Predicted Class': result['predicted_class'],
                                'Confidence': f"{result['confidence']:.2%}",
                                'Status': '✅ Success'
                            })
                        else:
                            results.append({
                                'Filename': filename,
                                'Predicted Class': 'Error',
                                'Confidence': '0%',
                                'Status': f"❌ {result['error']}"
                            })
                    except Exception as e:
                        results.append({
                            'Filename': filename,
                            'Predicted Class': 'Error',
                            'Confidence': '0%',
                            'Status': f"❌ {str(e)}"
                        })
                    
                    progress_bar.progress((i + 1) / len(st.session_state.batch_files))
                
                # Display results
                if results:
                    st.subheader("📊 Batch Processing Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_data,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    success_count = len([r for r in results if '✅' in r['Status']])
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Files", len(results))
                    with col2:
                        st.metric("Successful", success_count)
                    with col3:
                        st.metric("Success Rate", f"{success_count/len(results):.1%}")

# === TAB 3: Model Training ===
with tab3:
    st.header("🏋️ Train New Model")
    
    training_folder = st.text_input("Training Data Folder", value="./pdfs", help="Folder containing subfolders for each document class")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **Training Data Structure:**
        ```
        ./pdfs/
        ├── binder/
        │   ├── doc1.pdf
        │   └── doc2.pdf
        ├── contract/
        │   ├── doc3.pdf
        │   └── doc4.pdf
        ├── quotes/
        │   └── doc5.pdf
        └── policy/
            └── doc6.pdf
        ```
        """)
    
    with col2:
        epochs = st.number_input("Training Epochs", min_value=1, max_value=20, value=8)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=8, value=2)
    
    if st.button("🚀 Start Training", type="primary"):
        if os.path.exists(training_folder):
            with st.spinner("🔄 Preparing training data..."):
                try:
                    data = classifier.prepare_data_from_pdfs(training_folder)
                    
                    if len(data) < 20:
                        st.error("❌ Not enough data. Need at least 20 labeled documents for reliable training.")
                    else:
                        st.success(f"✅ Prepared {len(data)} training samples")
                        
                        # Show data distribution
                        class_counts = {}
                        for item in data:
                            class_name = classifier.id2label[item['label']]
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        st.subheader("📊 Data Distribution")
                        dist_df = pd.DataFrame([
                            {"Class": k, "Count": v} for k, v in class_counts.items()
                        ])
                        
                        fig = px.pie(dist_df, values="Count", names="Class", title="Training Data Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Split and train
                        from sklearn.model_selection import train_test_split
                        
                        train_data, test_data = train_test_split(
                            data, test_size=0.2, random_state=42,
                            stratify=[item['label'] for item in data]
                        )
                        
                        train_data, val_data = train_test_split(
                            train_data, test_size=0.2, random_state=42,
                            stratify=[item['label'] for item in train_data]
                        )
                        
                        st.info(f"📊 Split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
                        
                        # Training progress
                        training_placeholder = st.empty()
                        
                        with st.spinner("🔄 Training model... This may take several minutes."):
                            trainer = classifier.train(
                                train_data, 
                                val_data, 
                                output_dir=model_path,
                                epochs=epochs,
                                batch_size=batch_size
                            )
                            
                            # Evaluation
                            eval_results = trainer.evaluate()
                            
                        st.success("🎉 Training completed!")
                        
                        # Display results
                        st.subheader("📈 Training Results")
                        metrics_cols = st.columns(4)
                        
                        metrics = ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall']
                        metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                        
                        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                            if metric in eval_results:
                                with metrics_cols[i]:
                                    st.metric(name, f"{eval_results[metric]:.3f}")
                        
                        st.success("✅ Model saved and ready for predictions!")
                        
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    logger.error(f"Training error: {e}")
        else:
            st.error("❌ Training folder does not exist")

# === TAB 4: Analytics Dashboard ===
with tab4:
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.prediction_history:
        df = pd.DataFrame(st.session_state.prediction_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time series of predictions
        st.subheader("📈 Predictions Over Time")
        hourly_counts = df.set_index('timestamp').resample('H').size()
        fig = px.line(x=hourly_counts.index, y=hourly_counts.values, title="Predictions per Hour")
        st.plotly_chart(fig, use_container_width=True)
        
        # Class distribution
        st.subheader("🥧 Prediction Distribution")
        class_counts = df['predicted_class'].value_counts()
        fig = px.pie(values=class_counts.values, names=class_counts.index, title="Document Types Predicted")
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence analysis
        st.subheader("📊 Confidence Analysis")
        fig = px.histogram(df, x='confidence', nbins=20, title="Confidence Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent predictions table
        st.subheader("🕒 Recent Predictions")
        recent_df = df.sort_values('timestamp', ascending=False).head(10)
        st.dataframe(recent_df[['filename', 'predicted_class', 'confidence', 'timestamp']], use_container_width=True)
        
    else:
        st.info("📭 No predictions made yet. Upload some documents to see analytics!")

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 Powered by LayoutLMv3 • Built with Streamlit</p>
    <p><em>Advanced document classification with deep learning</em></p>
</div>
""", unsafe_allow_html=True)