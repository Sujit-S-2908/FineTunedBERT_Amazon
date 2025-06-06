import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import re
import random
import nltk
from nltk.corpus import stopwords
from collections import Counter
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Data loading and preprocessing
class AmazonReviewDataset:
    def __init__(self, kaggle_dataset_name="snap/amazon-fine-food-reviews", file_path="./Reviews.csv", sample_size=None):
        """
        Load and preprocess Amazon review dataset from Kaggle

        Args:
            kaggle_dataset_name: Name of the Kaggle dataset
            file_path: Path to the CSV file within the dataset
            sample_size: Number of samples to use (for development)
        """
        print(f"Loading dataset {kaggle_dataset_name}, file {file_path}...")

        # Load the dataset using kagglehub
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            kaggle_dataset_name,
            file_path
        )

        print(f"Original dataset shape: {df.shape}")

        # Use subset if sample_size provided
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)

        # Map the required columns to our standardized format
        # For the Amazon Fine Food Reviews dataset:
        # - 'Text' or 'review/text' contains the review text
        # - 'Score' or 'review/score' contains the rating (1-5)

        # Identify column names (they might vary)
        text_col = None
        score_col = None

        # Check for possible column names
        text_candidates = ['Text', 'review/text', 'reviewText', 'text']
        score_candidates = ['Score', 'review/score', 'rating', 'overall', 'stars']

        for col in text_candidates:
            if col in df.columns:
                text_col = col
                break

        for col in score_candidates:
            if col in df.columns:
                score_col = col
                break

        if text_col is None or score_col is None:
            print("Column names do not match expected patterns. Available columns:")
            print(df.columns)
            raise ValueError("Could not identify text and score columns")

        print(f"Using '{text_col}' for review text and '{score_col}' for ratings")

        # Create new dataframe with standardized column names
        self.df = pd.DataFrame({
            'text': df[text_col].astype(str),
            'label': df[score_col].astype(int) - 1  # Convert to 0-4 for model
        })

        # Filter out any rows with invalid ratings
        self.df = self.df[(self.df['label'] >= 0) & (self.df['label'] <= 4)]

        # Remove rows with empty text
        self.df = self.df[self.df['text'].str.strip().str.len() > 0]

        print(f"Loaded {len(self.df)} valid reviews")
        self._analyze_dataset()

    def _analyze_dataset(self):
        """Analyze dataset statistics"""
        print("Label distribution:")
        print(self.df['label'].value_counts().sort_index())

        # Calculate text lengths
        self.df['text_length'] = self.df['text'].apply(lambda x: len(x.split()))
        print(f"Average review length: {self.df['text_length'].mean():.2f} words")
        print(f"Max review length: {self.df['text_length'].max()} words")

        # Calculate class weights for imbalanced classes
        class_counts = Counter(self.df['label'])
        total = sum(class_counts.values())
        self.class_weights = {label: total / count for label, count in class_counts.items()}
        print("Class weights:", self.class_weights)

    def clean_text(self, apply_cleaning=True):
        """Apply text cleaning if requested"""
        if not apply_cleaning:
            return

        print("Cleaning review texts...")

        # Download stopwords if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        stop_words = set(stopwords.words('english'))

        def clean(text):
            # Convert to lowercase
            text = text.lower()
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Remove stopwords (optional - can be disabled as BERT handles this well)
            # words = [word for word in text.split() if word not in stop_words]
            # text = ' '.join(words)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        self.df['text'] = self.df['text'].apply(clean)

    def get_train_test_data(self, test_size=0.2, val_size=0.1):
        """Split into train, validation and test sets with stratification"""
        # First split: training + temp (validation + test)
        train_df, temp_df = train_test_split(
            self.df,
            test_size=test_size+val_size,
            random_state=42,
            stratify=self.df['label']
        )

        # Second split: validation + test from temp
        val_size_adjusted = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1-val_size_adjusted,
            random_state=42,
            stratify=temp_df['label']
        )

        print(f"Train set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")

        return train_df, val_df, test_df

# Custom Dataset for BERT
class AmazonBERTDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length=512):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        target = self.targets[idx]

        # Tokenize the text with BERT tokenizer
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

# Handling long reviews (sliding window approach)
class SlidingWindowBERTDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length=512, stride=256):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # Pre-process all reviews to create chunks
        self.chunks = []
        self.chunk_targets = []
        self.review_to_chunks = {}  # Maps review index to its chunk indices

        for idx, (review, target) in enumerate(zip(reviews, targets)):
            # Tokenize without padding or truncation to get actual tokens
            tokens = tokenizer.encode(review, add_special_tokens=True)

            if len(tokens) <= max_length:
                # If review fits in one chunk, just add it
                self.chunks.append(tokens)
                self.chunk_targets.append(target)
                self.review_to_chunks[idx] = [len(self.chunks) - 1]
            else:
                # Use sliding window for long reviews
                chunk_indices = []
                for i in range(0, len(tokens), stride):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) < 100:  # Skip very small chunks at the end
                        continue
                    self.chunks.append(chunk)
                    self.chunk_targets.append(target)
                    chunk_indices.append(len(self.chunks) - 1)
                self.review_to_chunks[idx] = chunk_indices

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        tokens = self.chunks[idx]
        target = self.chunk_targets[idx]

        # Pad if needed
        padding_length = self.max_length - len(tokens)
        if padding_length > 0:
            tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
        else:
            tokens = tokens[:self.max_length]

        attention_mask = [1] * min(len(tokens), self.max_length)
        padding_length = self.max_length - len(attention_mask)
        if padding_length > 0:
            attention_mask = attention_mask + [0] * padding_length

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }

# Model training and evaluation
class BERTSentimentClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=5, max_length=512, batch_size=16):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        self.model.to(device)

    def prepare_dataloader(self, df, is_training=True, use_sliding_window=False):
        """Create dataloader from DataFrame"""
        reviews = df['text'].values
        targets = df['label'].values

        if use_sliding_window:
            dataset = SlidingWindowBERTDataset(
                reviews=reviews,
                targets=targets,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                stride=self.max_length // 2
            )
        else:
            dataset = AmazonBERTDataset(
                reviews=reviews,
                targets=targets,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )

        # Use different samplers for training and validation/testing
        if is_training:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size
        )

        return dataloader

    def train(self, train_dataloader, val_dataloader, class_weights=None, epochs=4, learning_rate=2e-5):
        """Train the model"""
        # Define optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)

        # Total steps for scheduler
        total_steps = len(train_dataloader) * epochs

        # Create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Create loss weights if class imbalance is present
        if class_weights:
            # Convert dict to tensor
            weights = torch.tensor([class_weights[i] for i in range(self.num_labels)]).to(device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        # Training loop
        best_val_f1 = 0
        training_stats = []

        for epoch in range(epochs):
            print(f"\n{'=' * 20} Epoch {epoch+1}/{epochs} {'=' * 20}")

            # Training
            self.model.train()
            total_train_loss = 0

            for batch in tqdm(train_dataloader, desc="Training"):
                # Clear gradients
                self.model.zero_grad()

                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets
                )

                loss = outputs.loss

                # Backward pass and update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss:.4f}")

            # Validation
            val_loss, val_accuracy, val_f1_macro = self.evaluate(val_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation Macro F1: {val_f1_macro:.4f}")

            # Save stats
            training_stats.append({
                'epoch': epoch + 1,
                'training_loss': avg_train_loss,
                'valid_loss': val_loss,
                'valid_accuracy': val_accuracy,
                'valid_f1_macro': val_f1_macro
            })

            # Save best model
            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                # Save model
                model_path = f'bert_sentiment_model_epoch_{epoch+1}'
                self.save_model(model_path)
                print(f"Best model saved to {model_path}")

        return training_stats

    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()

        total_eval_loss = 0
        all_preds = []
        all_labels = []

        # No gradient calculation needed for evaluation
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets
                )

                loss = outputs.loss
                logits = outputs.logits

                total_eval_loss += loss.item()

                # Move predictions and labels to CPU
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = targets.cpu().numpy()

                # Extend lists
                all_preds.extend(preds)
                all_labels.extend(labels)

        # Calculate metrics
        avg_eval_loss = total_eval_loss / len(dataloader)
        accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        f1_macro = f1_score(all_labels, all_preds, average='macro')

        return avg_eval_loss, accuracy, f1_macro

    def predict_review(self, review_text):
        """Predict sentiment for a single review"""
        self.model.eval()

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            review_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        # Convert back to 1-5 rating
        return prediction + 1

    def predict_long_review(self, review_text, stride=256):
        """Predict sentiment for a very long review using sliding window"""
        self.model.eval()

        # Tokenize without padding or truncation
        tokens = self.tokenizer.encode(review_text, add_special_tokens=True)

        if len(tokens) <= self.max_length:
            # If review fits in one chunk, use normal prediction
            return self.predict_review(review_text)

        # Use sliding window for long reviews
        all_logits = []

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + self.max_length]
            if len(chunk_tokens) < 100:  # Skip very small chunks
                continue

            # Pad if needed
            attention_mask = [1] * len(chunk_tokens)
            if len(chunk_tokens) < self.max_length:
                padding_length = self.max_length - len(chunk_tokens)
                chunk_tokens = chunk_tokens + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            # Truncate if needed
            chunk_tokens = chunk_tokens[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

            # Convert to tensors
            input_ids = torch.tensor(chunk_tokens).unsqueeze(0).to(device)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

            # Get logits
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits)

        # Average logits from all chunks
        if all_logits:
            avg_logits = torch.mean(torch.cat(all_logits, dim=0), dim=0).unsqueeze(0)
            prediction = torch.argmax(avg_logits, dim=1).item()
            return prediction + 1
        else:
            # Fallback if something went wrong
            return self.predict_review(review_text)

    def save_model(self, path):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model and tokenizer"""
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.to(device)
        print(f"Model loaded from {path}")

    def confusion_matrix_analysis(self, dataloader):
        """Generate confusion matrix"""
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating confusion matrix"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get predictions
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                labels = targets.cpu().numpy()

                predictions.extend(preds)
                true_labels.extend(labels)

        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(1, 6),  # Convert back to 1-5 for display
                    yticklabels=range(1, 6))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Generate classification report
        report = classification_report(
            true_labels,
            predictions,
            target_names=[f"{i+1} stars" for i in range(self.num_labels)],
            digits=4
        )
        print("\nClassification Report:")
        print(report)

        return cm, report

    def error_analysis(self, test_df, predictions):
        """Analyze prediction errors"""
        test_df = test_df.copy()
        test_df['predicted'] = predictions
        test_df['correct'] = test_df['label'] == test_df['predicted']

        # Find most confused pairs
        errors = test_df[~test_df['correct']]
        error_counts = errors.groupby(['label', 'predicted']).size().reset_index(name='count')
        error_counts = error_counts.sort_values('count', ascending=False)

        print("\nMost frequent prediction errors:")
        for _, row in error_counts.head(5).iterrows():
            true_label = row['label'] + 1  # Convert back to 1-5
            pred_label = row['predicted'] + 1  # Convert back to 1-5
            count = row['count']
            print(f"True: {true_label} stars, Predicted: {pred_label} stars - {count} instances")

        # Sample errors for qualitative analysis
        print("\nSample misclassifications:")
        for _, row in error_counts.head(3).iterrows():
            true_label = row['label']
            pred_label = row['predicted']

            # Get some examples
            examples = errors[(errors['label'] == true_label) & (errors['predicted'] == pred_label)].head(2)

            for _, example in examples.iterrows():
                print(f"\nTrue: {true_label+1} stars, Predicted: {pred_label+1} stars")
                print(f"Review: {example['text'][:200]}...")

        return errors

# Main execution function
def main():
    # Load and preprocess data using kagglehub
    print("Loading dataset from Kaggle...")
    amazon_data = AmazonReviewDataset(
        kaggle_dataset_name="snap/amazon-fine-food-reviews",
        file_path="./Reviews.csv",
        sample_size=100000  # Adjust based on available resources
    )

    # Clean text data
    amazon_data.clean_text(apply_cleaning=True)

    # Split data
    train_df, val_df, test_df = amazon_data.get_train_test_data()

    # Initialize model
    print("Initializing BERT model...")
    classifier = BERTSentimentClassifier(
        model_name='bert-base-uncased',
        num_labels=5,
        max_length=512,
        batch_size=16  # Adjust based on available GPU memory
    )

    # Decision: use sliding window for long reviews?
    use_sliding_window = True  # Set to True if many reviews exceed 512 tokens

    # Prepare dataloaders
    print("Preparing dataloaders...")
    train_dataloader = classifier.prepare_dataloader(
        train_df,
        is_training=True,
        use_sliding_window=use_sliding_window
    )
    val_dataloader = classifier.prepare_dataloader(
        val_df,
        is_training=False,
        use_sliding_window=use_sliding_window
    )
    test_dataloader = classifier.prepare_dataloader(
        test_df,
        is_training=False,
        use_sliding_window=use_sliding_window
    )

    # Train model
    print("Training model...")
    training_stats = classifier.train(
        train_dataloader,
        val_dataloader,
        class_weights=amazon_data.class_weights,  # Use class weights for imbalance
        epochs=4,
        learning_rate=2e-5
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_f1_macro = classifier.evaluate(test_dataloader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_f1_macro:.4f}")

    # Generate confusion matrix and classification report
    print("\nGenerating confusion matrix and classification report...")
    cm, report = classifier.confusion_matrix_analysis(test_dataloader)

    # Predict on entire test set for error analysis
    print("\nRunning error analysis...")
    all_predictions = []

    # Get predictions for each test sample
    for _, row in tqdm(test_df.iterrows(), desc="Getting predictions", total=len(test_df)):
        if use_sliding_window and len(row['text'].split()) > 400:  # Approximate token count
            pred = classifier.predict_long_review(row['text'])
        else:
            pred = classifier.predict_review(row['text'])
        all_predictions.append(pred - 1)  # Convert back to 0-4 for analysis

    # Run error analysis
    errors = classifier.error_analysis(test_df, all_predictions)

    # Sample predictions
    print("\nSample predictions:")
    sample_reviews = test_df.sample(5)
    for _, row in sample_reviews.iterrows():
        review_text = row['text']
        true_rating = row['label'] + 1  # Convert to 1-5

        if use_sliding_window and len(review_text.split()) > 400:
            pred_rating = classifier.predict_long_review(review_text)
        else:
            pred_rating = classifier.predict_review(review_text)

        print(f"Review: {review_text[:100]}...")
        print(f"True rating: {true_rating} stars")
        print(f"Predicted rating: {pred_rating} stars")
        print("---")

if __name__ == "__main__":
    main()

