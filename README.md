# Resume Screening with LSTM

## Project Overview

An NLP-based deep learning project that automatically classifies resumes into job categories
using a Long Short-Term Memory (LSTM) neural network. The model takes raw resume text as input,
cleans and vectorizes it using TF-IDF, and predicts which of 25 job categories the resume belongs to.

This project demonstrates a full ML pipeline from raw text preprocessing to model deployment-ready inference.

---

## Problem Statement

Manually screening resumes is time-consuming and prone to bias. This project automates the
categorization of resumes, enabling HR teams and recruiting platforms to instantly route
candidates to the right job category based on their resume content.

---

## Project Pipeline

1. **Data Loading** — Load the [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) containing resumes across 25 job categories
2. **EDA** — Visualize category distributions, word frequencies, and word clouds
3. **Text Cleaning** — Remove URLs, mentions, hashtags, punctuation, and stopwords using regex + NLTK
4. **Feature Extraction** — Vectorize cleaned resume text using TF-IDF (`sublinear_tf=True`)
5. **Label Encoding** — Encode the 25 job categories into numerical labels
6. **LSTM Model** — Build and train a Sequential LSTM model with 64 hidden units and softmax output
7. **Evaluation** — Assess performance using accuracy metrics and a confusion matrix
8. **Inference** — Predict the job category of any new raw resume text

---

## Model Architecture

```
Input (TF-IDF vectors)
       ↓
LSTM Layer (64 hidden units)
       ↓
Dense Layer (25 units)
       ↓
Softmax Activation → Predicted Category
```

---

## Tech Stack

- Python
- Keras / TensorFlow — LSTM model
- Scikit-learn — TF-IDF Vectorizer, Label Encoder, train/test split
- NLTK — Tokenization, stopword removal
- Pandas, NumPy
- Matplotlib, Seaborn — Visualizations
- WordCloud

---

## Dataset

- **Source:** [Kaggle — Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- **Size:** 962 resumes across **25 job categories**
- Categories include: Data Science, HR, Advocate, Arts, Web Designing, Mechanical Engineer, and more

---

## How to Run

```bash
# Install dependencies
pip install tensorflow scikit-learn nltk pandas matplotlib seaborn wordcloud

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Run the notebook
jupyter notebook resume-screening-lstm.ipynb
```
