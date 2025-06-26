# Document Classification with LayoutLMv3

A machine learning project that uses Microsoft's LayoutLMv3 model to classify PDF documents into different categories based on their visual layout and textual content.

## Overview

This project implements an intelligent document classifier that can automatically categorize PDF documents into predefined classes: **binder**, **contract**, **quotes**, and **policy**. The system leverages LayoutLMv3, a state-of-the-art multimodal transformer that understands both text and visual layout information.

## Features

- **PDF Processing**: Converts PDF pages to images and extracts text with OCR
- **Multimodal Classification**: Uses both text content and visual layout for classification
- **Web Interface**: User-friendly Streamlit app for easy document upload and classification
- **Custom Training**: Train on your own labeled document dataset
- **Batch Prediction**: Classify multiple documents efficiently
- **Confidence Scoring**: Get prediction confidence scores for each classification
- **Visual Results**: Interactive charts showing classification confidence
- **Early Stopping**: Prevents overfitting during training
- **Developer Tools**: Built-in training interface in the web app

## Requirements

### Dependencies

```bash
torch
transformers
pandas
pillow
pytesseract
PyMuPDF
scikit-learn
numpy
tqdm
streamlit
```

### System Requirements

- **Tesseract OCR**: Must be installed on your system
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/harshbopaliya/pdf_classifier.git
cd pdf-classifier
```

2. Install Python dependencies:
```bash
pip install -r requiremets.txt
```

3. Install Tesseract OCR (see system requirements above)

## Usage

### Web Interface (Streamlit App)

The project includes a user-friendly web interface built with Streamlit for easy document classification.

#### Running the Web App

1. Save your main classifier code as `classifier.py`
2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

#### Web App Features

- **File Upload**: Drag and drop PDF files for instant classification
- **Real-time Prediction**: Get immediate results with confidence scores
- **Visual Results**: Interactive bar chart showing confidence for all classes
- **Developer Tools**: Built-in training interface for model updates

### Data Preparation

Organize your PDF documents in the following directory structure:

```
pdfs/
├── binder/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── contract/
│   ├── contract1.pdf
│   ├── contract2.pdf
│   └── ...
├── quotes/
│   ├── quote1.pdf
│   └── ...
└── policy/
    ├── policy1.pdf
    └── ...
```

### Command Line Training

Run the main script to train the classifier:

```bash
python document_classifier.py
```

The training process will:
1. Extract text and layout information from all PDFs
2. Split data into training (80%) and validation (20%) sets
3. Fine-tune the LayoutLMv3 model
4. Save the trained model to `./results/`
5. Evaluate performance on validation data
6. Test predictions on sample documents

### Web Interface Training

Alternatively, use the Streamlit interface for training:
1. Run `streamlit run app.py`
2. Expand the "Train Model (for developers)" section
3. Click "Train on ./pdfs folder" button
4. Monitor training progress in the web interface

### Making Predictions

#### Using the Web Interface
1. Run the Streamlit app: `streamlit run app.py`
2. Upload a PDF file through the web interface
3. View results instantly with confidence scores and visualizations

#### Programmatic Usage

```python
from classifier import DocumentClassifier

# Initialize classifier
classifier = DocumentClassifier()

# Predict on a single PDF
result = classifier.predict("path/to/document.pdf", model_path="./results")

print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"All scores: {result['all_scores']}")
```

## Model Architecture

The system uses **LayoutLMv3** from Microsoft, which combines:
- **Text Understanding**: Processes document text content
- **Visual Understanding**: Analyzes document layout and formatting
- **Spatial Awareness**: Understands the relationship between text elements

### Key Components

1. **PDFProcessor**: Converts PDFs to images and extracts OCR data
2. **DocumentDataset**: Custom PyTorch dataset for training
3. **DocumentClassifier**: Main class handling training and inference

## Configuration

### Hyperparameters

- **Batch Size**: 4 (adjustable based on GPU memory)
- **Learning Rate**: Default from TrainingArguments
- **Max Sequence Length**: 512 tokens
- **Training Epochs**: 5 (with early stopping)
- **OCR Confidence Threshold**: 30

### Model Parameters

- **Base Model**: `microsoft/layoutlmv3-base`
- **Number of Classes**: 4 (binder, contract, quotes, policy)
- **Image DPI**: 150 for PDF conversion

## Performance

The model's performance depends on:
- Quality and variety of training data
- Document complexity and layout consistency
- OCR accuracy on the input documents

Monitor training progress through:
- Validation accuracy during training
- Confidence scores on predictions
- Confusion matrix analysis (can be added)

## File Structure

```
├── classifier.py                # Main classifier implementation
├── app.py                       # Streamlit web interface
├── README.md                    # This file
├── pdfs/                        # Training data directory
├── results/                     # Saved model outputs
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files...
├── temp/                        # Temporary uploaded files
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Ensure Tesseract is installed and in PATH
2. **CUDA out of memory**: Reduce batch size in TrainingArguments
3. **Low accuracy**: Increase training data or adjust hyperparameters
4. **PDF processing errors**: Check PDF file integrity and format
5. **Streamlit port issues**: Use `streamlit run app.py --server.port 8502` for different ports
6. **File upload issues**: Check temp directory permissions and disk space

### Error Messages

- `"Need at least 10 labeled documents for training!"`: Add more training samples
- `"Error processing PDF"`: Check if PDF is corrupted or password-protected
- `"Could not extract text from PDF"`: PDF might be image-only or low quality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project uses the MIT License. The LayoutLMv3 model is subject to Microsoft's licensing terms.

## Citation

If you use this project in research, please cite:

```bibtex
@article{huang2022layoutlmv3,
  title={LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking},
  author={Huang, Yupan and Lv, Tengchao and Cui, Lei and Lu, Yutong and Wei, Furu},
  journal={arXiv preprint arXiv:2204.08387},
  year={2022}
}
```

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the [LayoutLMv3 documentation](https://huggingface.co/microsoft/layoutlmv3-base)
- Open an issue in this repository