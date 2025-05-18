# Fine-Tuned BERT for Amazon Fine Food Reviews

This project implements a fine-tuned BERT model for sentiment analysis on the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset. The model predicts review ratings (1-5 stars) using advanced NLP techniques and handles long reviews with a sliding window approach.

## Features

-   Loads and preprocesses the Amazon Fine Food Reviews dataset using `kagglehub`.
-   Cleans and standardizes review text.
-   Handles class imbalance with computed class weights.
-   Supports both standard and sliding window BERT tokenization for long reviews.
-   Trains a BERT-based classifier for 5-class sentiment prediction.
-   Provides evaluation metrics: accuracy, macro F1, confusion matrix, and classification report.
-   Includes error analysis and qualitative review of misclassifications.
-   Saves and loads fine-tuned models.

## Project Structure

-   `main.py`: Main script for data loading, preprocessing, model training, evaluation, and analysis.
-   `README.md`: Project documentation (this file).
-   `requirements.txt`: List of required Python packages.

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd FineTunedBERT_Amazon
```

### 2. Install Python Dependencies

It is recommended to use a virtual environment:

```sh
python -m venv venv
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

### 3. Kaggle API Setup

-   Create a Kaggle account and generate an API token from your [Kaggle account settings](https://www.kaggle.com/settings/account).
-   Place the downloaded `kaggle.json` file in the appropriate location (see [kagglehub documentation](https://github.com/Kaggle/kagglehub)).

### 4. Run the Script

```sh
python main.py
```

## Usage

-   The script will automatically download and preprocess the dataset, train the BERT model, evaluate its performance, and output results including confusion matrix and error analysis.
-   You can adjust parameters such as `sample_size`, `batch_size`, `epochs`, and whether to use the sliding window approach in `main.py`.

## Requirements

See `requirements.txt` for the full list. Key packages include:

-   torch
-   transformers
-   pandas
-   numpy
-   scikit-learn
-   matplotlib
-   seaborn
-   tqdm
-   nltk
-   kagglehub

## Notes

-   Training on the full dataset may require a GPU with sufficient memory. Adjust `sample_size` and `batch_size` as needed.
-   The script saves the best model during training and outputs evaluation plots and reports.

## References

-   [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
-   [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
-   [HuggingFace Transformers](https://huggingface.co/transformers/)
-   [kagglehub](https://github.com/Kaggle/kagglehub)

## License

This project is for educational and research purposes. Please check the dataset and model licenses for commercial use.
