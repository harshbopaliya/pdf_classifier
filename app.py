import os
import streamlit as st
from classifier import DocumentClassifier  # make sure your original code is saved as classifier.py

# === Setup ===
st.set_page_config(page_title="📄 PDF Document Classifier", layout="centered")
st.title("📑 Document Type Classifier")
st.markdown("Upload a PDF file to classify it into one of: **binder**, **contract**, **quotes**, or **policy**.")

# === Instantiate Model ===
@st.cache_resource
def load_classifier():
    return DocumentClassifier()

classifier = load_classifier()

# === File Upload ===
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file to temp directory
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully.")
    st.write("Running prediction...")

    result = classifier.predict(temp_path)

    if isinstance(result, dict):
        st.subheader("🧠 Prediction Result")
        st.write(f"**Predicted Class:** {result['predicted_class']}")
        st.write(f"**Confidence:** {result['confidence']:.2f}")
        st.bar_chart(result['all_scores'])
    else:
        st.error(result)

# === Optional: Trigger Training ===
with st.expander("⚙️ Train Model (for developers)"):
    if st.button("Train on ./pdfs folder"):
        st.warning("Training might take a long time depending on dataset size and system performance.")
        pdf_dir = "./pdfs"
        st.write("Preparing training data...")
        data = classifier.prepare_data_from_pdfs(pdf_dir)

        if len(data) < 10:
            st.error("At least 10 labeled documents are required for training.")
        else:
            from sklearn.model_selection import train_test_split
            train_data, val_data = train_test_split(
                data, test_size=0.2, random_state=42,
                stratify=[item['label'] for item in data]
            )

            trainer = classifier.train(train_data, val_data, epochs=5)
            eval_results = trainer.evaluate()
            st.success(f"Validation Accuracy: {eval_results['eval_accuracy']:.3f}")
