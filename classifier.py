import os
import torch
import numpy as np
import random
import warnings
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from io import BytesIO
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
import torch.nn.functional as F
from collections import Counter, defaultdict
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class DocumentDataset(Dataset):
    def __init__(self, data, processor, max_length=512):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle potential issues with image processing
        try:
            encoding = self.processor(
                item['image'],
                item['words'],
                boxes=item['boxes'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Ensure all tensors have the right shape
            encoding = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in encoding.items()}
            encoding['labels'] = torch.tensor(item['label'], dtype=torch.long)
            
            return encoding
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            # Return a dummy encoding in case of error
            return self._get_dummy_encoding(item['label'])
    
    def _get_dummy_encoding(self, label):
        """Create a dummy encoding for failed cases"""
        return {
            'input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'bbox': torch.zeros(self.max_length, 4, dtype=torch.long),
            'pixel_values': torch.zeros(3, 224, 224, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class PDFProcessor:
    def __init__(self, dpi=200, confidence_threshold=40):
        self.dpi = dpi  # Increased DPI for better OCR
        self.confidence_threshold = confidence_threshold
        
    def pdf_to_image_and_data(self, pdf_path, page_num=0):
        try:
            doc = fitz.open(pdf_path)
            
            # Handle multi-page documents by processing first page or specified page
            if page_num >= len(doc):
                page_num = 0
                
            page = doc[page_num]
            
            # Higher resolution for better OCR
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            image = Image.open(BytesIO(pix.tobytes("ppm"))).convert("RGB")
            
            # Preprocess image for better OCR
            image = self._preprocess_image(image)
            
            # Extract text with OCR
            ocr_data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume uniform block of text
            )
            
            words, boxes = self._extract_words_and_boxes(ocr_data, image)
            
            doc.close()
            return image, words, boxes
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return None, [], []
    
    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale and back to RGB for consistency
        from PIL import ImageEnhance
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        return image
    
    def _extract_words_and_boxes(self, ocr_data, image):
        """Extract words and bounding boxes with better filtering"""
        words, boxes = [], []
        
        for i in range(len(ocr_data['text'])):
            confidence = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != -1 else 0
            
            if confidence > self.confidence_threshold:
                text = ocr_data['text'][i].strip()
                
                # Filter out single characters and noise
                if len(text) > 1 and text.isalnum():
                    x, y, w, h = (
                        ocr_data['left'][i], 
                        ocr_data['top'][i], 
                        ocr_data['width'][i], 
                        ocr_data['height'][i]
                    )
                    
                    # Normalize coordinates to 0-1000 range
                    box = [
                        max(0, min(1000, int(1000 * x / image.width))),
                        max(0, min(1000, int(1000 * y / image.height))),
                        max(0, min(1000, int(1000 * (x + w) / image.width))),
                        max(0, min(1000, int(1000 * (y + h) / image.height)))
                    ]
                    
                    # Ensure box is valid
                    if box[2] > box[0] and box[3] > box[1]:
                        words.append(text)
                        boxes.append(box)
        
        return words, boxes

class DocumentClassifier:
    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        self.label_names = ['binder', 'contract', 'quotes', 'policy']
        self.label2id = {label: i for i, label in enumerate(self.label_names)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_names),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        self.pdf_processor = PDFProcessor()
        
    def prepare_data_from_pdfs(self, base_pdf_dir):
        """Prepare data with better balancing and validation"""
        data_by_label = defaultdict(list)
        
        for folder in os.listdir(base_pdf_dir):
            label_path = os.path.join(base_pdf_dir, folder)
            if not os.path.isdir(label_path):
                continue
                
            label = folder.strip().lower()
            if label not in self.label2id:
                logger.warning(f"Skipping unknown label folder: {label_path}")
                continue
                
            label_id = self.label2id[label]
            pdfs = [f for f in os.listdir(label_path) if f.lower().endswith('.pdf')]
            
            logger.info(f"Processing {len(pdfs)} PDFs in '{label}' folder...")
            
            for file in tqdm(pdfs, desc=f"Processing {label}"):
                full_path = os.path.join(label_path, file)
                image, words, boxes = self.pdf_processor.pdf_to_image_and_data(full_path)
                
                if image is not None and len(words) > 5:  # Minimum word threshold
                    data_by_label[label].append({
                        'image': image,
                        'words': words,
                        'boxes': boxes,
                        'label': label_id,
                        'filename': file
                    })
        
        # Better balancing strategy
        return self._balance_dataset(data_by_label)
    
    def _balance_dataset(self, data_by_label):
        """Implement SMOTE-like balancing for better distribution"""
        min_samples = min(len(samples) for samples in data_by_label.values())
        max_samples = max(len(samples) for samples in data_by_label.values())
        
        # Target somewhere between min and max to avoid over-duplication
        target_samples = min(max_samples, min_samples * 3)
        
        logger.info(f"Balancing dataset to {target_samples} samples per class")
        
        balanced_data = []
        for label, samples in data_by_label.items():
            if len(samples) < target_samples:
                # Oversample with some randomization
                balanced_samples = self._oversample_with_augmentation(samples, target_samples)
            else:
                # Undersample by random selection
                balanced_samples = random.sample(samples, target_samples)
            
            balanced_data.extend(balanced_samples)
            logger.info(f"   {label}: {len(balanced_samples)} samples")
        
        random.shuffle(balanced_data)
        return balanced_data
    
    def _oversample_with_augmentation(self, samples, target_count):
        """Oversample with slight variations to improve generalization"""
        result = samples.copy()
        
        while len(result) < target_count:
            sample = random.choice(samples)
            # Create a slightly modified version
            augmented_sample = sample.copy()
            # Could add image augmentation here in the future
            result.append(augmented_sample)
        
        return result[:target_count]
    
    def train(self, train_data, val_data, output_dir="./results", epochs=8, batch_size=2):
        """Enhanced training with better hyperparameters"""
        train_dataset = DocumentDataset(train_data, self.processor)
        val_dataset = DocumentDataset(val_data, self.processor)
        
        # Calculate steps based on dataset size
        steps_per_epoch = len(train_dataset) // batch_size
        eval_steps = max(50, steps_per_epoch // 4)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,  # Effective batch size = batch_size * 2
            warmup_steps=min(500, steps_per_epoch),
            weight_decay=0.01,
            learning_rate=2e-5,  # Lower learning rate for fine-tuning
            logging_dir=f'{output_dir}/logs',
            logging_steps=min(25, steps_per_epoch // 4),
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=None,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
        )
        
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=1)
            
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            accuracy = accuracy_score(labels, preds)
            
            return {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save model and processor
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        
        # Save label mappings
        with open(os.path.join(output_dir, 'label_mappings.json'), 'w') as f:
            json.dump({
                'id2label': self.id2label,
                'label2id': self.label2id
            }, f)
        
        return trainer
    
    def predict(self, pdf_path, model_path="./results"):
        """Enhanced prediction with confidence scoring"""
        try:
            # Load model and processor
            model = LayoutLMv3ForSequenceClassification.from_pretrained(model_path)
            processor = LayoutLMv3Processor.from_pretrained(model_path)
            
            # Load label mappings
            with open(os.path.join(model_path, 'label_mappings.json'), 'r') as f:
                mappings = json.load(f)
                id2label = {int(k): v for k, v in mappings['id2label'].items()}
            
            # Process PDF
            image, words, boxes = self.pdf_processor.pdf_to_image_and_data(pdf_path)
            
            if not words:
                return {"error": "Could not extract text from PDF"}
            
            # Prepare input
            encoding = processor(
                image,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                outputs = model(**encoding)
                probs = F.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred].item()
            
            # Prepare results
            all_scores = {id2label[i]: probs[0][i].item() for i in range(len(id2label))}
            
            return {
                "predicted_class": id2label[pred],
                "confidence": confidence,
                "all_scores": all_scores,
                "words_extracted": len(words)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}
    
    def evaluate_model(self, test_data, model_path="./results"):
        """Comprehensive model evaluation"""
        predictions = []
        actuals = []
        
        for item in tqdm(test_data, desc="Evaluating"):
            # Create a temporary file for prediction
            temp_path = f"/tmp/temp_{item['filename']}"
            item['image'].save(temp_path)
            
            pred_result = self.predict(temp_path, model_path)
            if 'error' not in pred_result:
                predictions.append(pred_result['predicted_class'])
                actuals.append(self.id2label[item['label']])
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Generate classification report
        if predictions and actuals:
            report = classification_report(actuals, predictions, target_names=self.label_names)
            accuracy = accuracy_score(actuals, predictions)
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'predictions': predictions,
                'actuals': actuals
            }
        
        return None

def main():
    """Main training and evaluation pipeline"""
    set_seed(42)
    
    classifier = DocumentClassifier()
    pdf_dir = "./pdfs"
    
    logger.info("Preparing data from PDFs...")
    data = classifier.prepare_data_from_pdfs(pdf_dir)
    
    if len(data) < 20:
        logger.warning("Not enough data. Need at least 20 samples for reliable training.")
        return
    
    # Split data with stratification
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        stratify=[d['label'] for d in data],
        random_state=42
    )
    
    # Further split training data for validation
    train_data, val_data = train_test_split(
        train_data,
        test_size=0.2,
        stratify=[d['label'] for d in train_data],
        random_state=42
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Train model
    trainer = classifier.train(train_data, val_data, epochs=8, batch_size=2)
    
    # Final evaluation
    logger.info("Final Evaluation on Validation Set:")
    result = trainer.evaluate()
    for key, value in result.items():
        if key.startswith('eval_'):
            logger.info(f"{key}: {value:.4f}")
    
    # Test on holdout set
    logger.info("Evaluating on test set...")
    test_results = classifier.evaluate_model(test_data)
    if test_results:
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info("Classification Report:")
        logger.info(test_results['classification_report'])
    
    # Sample predictions
    logger.info("Sample Predictions:")
    for i, sample in enumerate(random.sample(val_data, min(3, len(val_data)))):
        actual = classifier.id2label[sample["label"]]
        
        # Save image temporarily for prediction
        temp_path = f"/tmp/sample_{i}.png"
        sample['image'].save(temp_path)
        
        pred = classifier.predict(temp_path)
        
        logger.info(f"Sample {i+1}: {sample['filename']}")
        if 'error' not in pred:
            logger.info(f"   Predicted: {pred['predicted_class']} ({pred['confidence']:.3f})")
            logger.info(f"   Actual: {actual}")
        else:
            logger.info(f"   Error: {pred['error']}")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()