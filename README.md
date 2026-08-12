# SMS Spam Classification using SVM

A machine learning project that classifies SMS messages as **Spam** or **Ham** (normal messages) using a **Support Vector Machine (SVM)** classifier, with comparison against Naive Bayes and Logistic Regression.

## Overview

SMS spam is a common problem, and automatically filtering it out requires a model that can understand text patterns. This project builds a full text-classification pipeline: cleaning raw SMS text, converting it into numerical features with TF-IDF, and training a **LinearSVC (SVM)** model to detect spam.

## Dataset

- **Source:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (Kaggle)
- **Size:** 5,572 SMS messages labeled as `spam` or `ham`
- **Format:** CSV file with two relevant columns — message label and message text

## Project Workflow

1. **Data Loading & Cleaning** — Load `spam.csv`, drop unnecessary columns, check for nulls/duplicates
2. **Exploratory Data Analysis (EDA)** — Class distribution, message length analysis, word clouds
3. **Text Preprocessing** — Lowercasing, removing URLs/emails/numbers/punctuation, stopword removal, lemmatization
4. **Train/Test Split** — 80/20 split
5. **Model Training** — TF-IDF vectorization + LinearSVC pipeline
6. **Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC Curve
7. **Model Comparison** — SVM vs Naive Bayes vs Logistic Regression
8. **Hyperparameter Tuning** — GridSearchCV to find the best `C` value
9. **Cross Validation** — 5-fold CV to check model stability
10. **Error Analysis** — Inspecting misclassified messages
11. **Model Saving** — Trained model exported with `joblib` for reuse
12. **Prediction Function** — Simple function to classify a new SMS message

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| SVM (LinearSVC) | *fill in after running* | | | |
| Naive Bayes | *fill in after running* | | | |
| Logistic Regression | *fill in after running* | | | |

> Run the notebook and paste your actual results here — this shows the reader real evidence instead of placeholder text.

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (TF-IDF, LinearSVC, GridSearchCV)
- nltk (stopwords, lemmatization)
- matplotlib, seaborn, wordcloud (visualization)
- joblib (model persistence)

## How to Run

1. Clone this repository
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Download `spam.csv` from the [Kaggle dataset link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it in the project folder
4. Open and run `SMS_Spam_SVM_refined.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab

## Project Structure

```
├── SMS_Spam_SVM_refined.ipynb   # Main notebook
├── spam.csv                     # Dataset (not included — download separately)
├── sms_spam_svm_model.pkl       # Saved trained model (generated after running)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Author

**Afia Tasnia**
Bangladesh University of Professionals
Artificial Intelligence Laboratory

## Acknowledgements

Dataset originally compiled by the UCI Machine Learning Repository, sourced via Kaggle.
