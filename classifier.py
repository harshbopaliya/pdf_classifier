import os
import json
import torch
import pandas as pd
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings('ignore')

class DocumentDataset(Dataset):
    def __init__(self, data, processor, max_length=512):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        words = item['words']
        boxes = item['boxes']
        label = item['label']

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        for key in encoding.keys():
            encoding[key] = encoding[key].squeeze()
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        return encoding

class PDFProcessor:
    def __init__(self):
        self.dpi = 150

    def pdf_to_image_and_data(self, pdf_path, page_num=0):
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            from io import BytesIO
            image = Image.open(BytesIO(img_data)).convert('RGB')

            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            words, boxes = [], []

            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > 30:
                    text = ocr_data['text'][i].strip()
                    if text:
                        words.append(text)
                        x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                                      ocr_data['width'][i], ocr_data['height'][i])
                        box = [
                            int(1000 * x / image.width),
                            int(1000 * y / image.height),
                            int(1000 * (x + w) / image.width),
                            int(1000 * (y + h) / image.height)
                        ]
                        boxes.append(box)

            doc.close()
            return image, words, boxes
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return None, [], []

class DocumentClassifier:
    def __init__(self):
        self.label_names = ['binder', 'contract', 'quotes', 'policy']
        self.label2id = {label: i for i, label in enumerate(self.label_names)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(self.label_names)

        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.pdf_processor = PDFProcessor()

    def prepare_data_from_pdfs(self, base_pdf_dir):
        data = []
        for label_folder in os.listdir(base_pdf_dir):
            label_path = os.path.join(base_pdf_dir, label_folder)
            if not os.path.isdir(label_path):
                continue

            folder_name = label_folder.strip().lower()
            if folder_name in self.label2id:
                label_name = folder_name
            else:
                label_name = 'policy'

            label_id = self.label2id[label_name]
            pdf_files = [f for f in os.listdir(label_path) if f.lower().endswith('.pdf')]
            print(f"Processing {len(pdf_files)} PDFs in '{label_folder}' folder as label '{label_name}'...")

            for pdf_file in tqdm(pdf_files):
                pdf_path = os.path.join(label_path, pdf_file)
                image, words, boxes = self.pdf_processor.pdf_to_image_and_data(pdf_path)
                if image is not None and words:
                    data.append({
                        'image': image,
                        'words': words,
                        'boxes': boxes,
                        'label': label_id,
                        'filename': pdf_file
                    })
        return data

    def train(self, train_data, val_data, output_dir='./results', epochs=5):
        train_dataset = DocumentDataset(train_data, self.processor)
        val_dataset = DocumentDataset(val_data, self.processor)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=None,
            dataloader_num_workers=0,
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'accuracy': accuracy_score(labels, predictions)}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print("Starting training...")
        trainer.train()
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        return trainer

    def predict(self, pdf_path, model_path='./results'):
        model = LayoutLMv3ForSequenceClassification.from_pretrained(model_path)
        processor = LayoutLMv3Processor.from_pretrained(model_path)
        image, words, boxes = self.pdf_processor.pdf_to_image_and_data(pdf_path)

        if not words:
            return "Error: Could not extract text from PDF"

        encoding = processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()

        return {
            'predicted_class': self.id2label[predicted_class],
            'confidence': confidence,
            'all_scores': {self.id2label[i]: predictions[0][i].item() for i in range(self.num_labels)}
        }

# def main():
#     classifier = DocumentClassifier()
#     pdf_directory = "./pdfs"

#     print("Preparing data from PDFs...")
#     data = classifier.prepare_data_from_pdfs(pdf_directory)

#     if len(data) < 10:
#         print("Need at least 10 labeled documents for training!")
#         return

#     train_data, val_data = train_test_split(data, test_size=0.2, random_state=42,
#                                             stratify=[item['label'] for item in data])

#     print(f"Training samples: {len(train_data)}")
#     print(f"Validation samples: {len(val_data)}")

#     trainer = classifier.train(train_data, val_data, epochs=5)

#     print("Evaluating model...")
#     eval_results = trainer.evaluate()
#     print(f"Validation Accuracy: {eval_results['eval_accuracy']:.3f}")

#     if len(data) > 0:
#         test_pdf = os.path.join(pdf_directory, classifier.label_names[data[0]['label']], data[0]['filename'])
#         result = classifier.predict(test_pdf)
#         print(f"\nTest prediction for {test_pdf}:")
#         print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:})")
def main():
    classifier = DocumentClassifier()
    pdf_directory = "./pdfs"

    print("Preparing data from PDFs...")
    data = classifier.prepare_data_from_pdfs(pdf_directory)

    if len(data) < 10:
        print("Need at least 10 labeled documents for training!")
        return

    # Split into training and validation
    train_data, val_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=[item['label'] for item in data]
    )

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Train model
    trainer = classifier.train(train_data, val_data, epochs=5)

    # Evaluate
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.3f}")

    # Predict on random validation samples
    print("\n🔍 Predicting on validation samples...")
    sample_indices = random.sample(range(len(val_data)), min(3, len(val_data)))

    for idx in sample_indices:
        sample = val_data[idx]
        actual_label = classifier.id2label[sample['label']]
        test_pdf = os.path.join(pdf_directory, actual_label, sample['filename'])

        if not os.path.exists(test_pdf):
            print(f"⚠️ File not found: {test_pdf}")
            continue

        result = classifier.predict(test_pdf)

        print(f"\n📄 File: {sample['filename']}")
        if isinstance(result, dict):
            print(f"  ✅ Predicted: {result['predicted_class']} (confidence: {result['confidence']:.2f})")
            print(f"  🏷️ Actual:    {actual_label}")
        else:
            print(f"  ❌ Prediction Error: {result}")


if __name__ == "__main__":
    main()
