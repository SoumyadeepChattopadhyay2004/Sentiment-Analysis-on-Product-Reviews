# 💬 Sentiment Analysis on Product Reviews

> An NLP project to classify product reviews as Positive, Negative, or Neutral using text preprocessing, feature engineering, and machine learning models.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [NLP Pipeline](#nlp-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## 🔍 Overview

Online product reviews are a goldmine of customer feedback, but analyzing them manually is impractical at scale. This project applies **Natural Language Processing (NLP)** and **machine learning** to automatically classify the sentiment expressed in product reviews — helping businesses understand customer satisfaction and make data-driven decisions.

Key goals:
- Clean and preprocess raw review text data
- Perform exploratory analysis to uncover sentiment patterns
- Build and compare multiple ML classifiers for sentiment prediction
- Evaluate model performance using standard NLP metrics

---

## 📂 Dataset

The dataset contains customer product reviews with associated **star ratings** that serve as the ground truth for sentiment labels. Ratings are mapped to sentiment classes as follows:

| Star Rating | Sentiment Label |
|---|---|
| 4 – 5 stars | ✅ Positive |
| 3 stars | 😐 Neutral |
| 1 – 2 stars | ❌ Negative |

**Key columns typically include:**

| Column | Description |
|---|---|
| `reviewText` | The raw review written by the customer |
| `overall` | Star rating (1–5) used to derive sentiment label |
| `summary` | Short review headline/title |
| `sentiment` | Derived target variable (Positive / Neutral / Negative) |

---

## 🗂️ Project Structure

```
Sentiment-Analysis-on-Product-Reviews/
│
├── analysis.ipynb       # Main Jupyter notebook (full pipeline)
├── summary.pdf          # Project summary and findings report
├── README.md            # Project documentation
│
├── Chart 1.png          # EDA / sentiment distribution visualization
├── Chart 2.png          # Word frequency / token analysis
├── Chart 3.png          # Model performance comparison
└── Chart 4.png          # Additional analysis visualization
```

---

## 🔄 NLP Pipeline

```
Raw Review Text
      ↓
Text Cleaning
  - Lowercasing
  - Removing punctuation, special characters, HTML tags
  - Removing stopwords
  - Handling contractions
      ↓
Tokenization
      ↓
Stemming / Lemmatization
      ↓
Feature Extraction
  - Bag of Words (CountVectorizer)
  - TF-IDF Vectorizer
      ↓
Sentiment Labeling (from star ratings)
      ↓
Train / Test Split
      ↓
Model Training & Evaluation
      ↓
Results & Insights
```

---

## 📊 Exploratory Data Analysis

The EDA section covers:

- **Sentiment distribution** — class balance across Positive, Negative, and Neutral reviews
- **Review length analysis** — distribution of word counts per sentiment class
- **Most frequent words** — top tokens per sentiment category (word frequency plots)
- **Word clouds** — visual representation of dominant vocabulary per class
- **Rating distribution** — how star ratings map to sentiments
- **Review trends** — patterns in review text length and complexity

---

## 🧪 Methodology

**Text Preprocessing:**
- Converted text to lowercase
- Removed HTML tags, URLs, punctuation, and numbers
- Stripped English stopwords using NLTK
- Applied stemming (PorterStemmer) and/or lemmatization (WordNetLemmatizer)

**Feature Extraction:**

| Technique | Description |
|---|---|
| **Bag of Words** | Word frequency matrix using `CountVectorizer` |
| **TF-IDF** | Term Frequency–Inverse Document Frequency weighting |

**Label Engineering:**
- Star ratings mapped to 3-class sentiment labels (Positive / Neutral / Negative)
- Handled class imbalance where present

---

## 🤖 Models Used

| Model | Type |
|---|---|
| Logistic Regression | Linear classifier |
| Naive Bayes (MultinomialNB) | Probabilistic text classifier |
| Support Vector Machine (SVM) | Margin-based classifier |
| Random Forest | Ensemble tree-based classifier |
| Decision Tree | Single tree classifier |
| K-Nearest Neighbors | Distance-based classifier |

Each model was evaluated on **Accuracy**, **Precision**, **Recall**, and **F1-Score**, with confusion matrices generated for deeper analysis.

---

## 📈 Results

The models were benchmarked against each other, with **TF-IDF features** generally outperforming Bag-of-Words representations. Logistic Regression and SVM tend to perform strongly on text classification tasks, while Naive Bayes offers a fast and competitive baseline.

> For full model metrics, confusion matrices, and visualizations, refer to [`analysis.ipynb`](./analysis.ipynb) or [`summary.pdf`](./summary.pdf).

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/SoumyadeepChattopadhyay2004/Sentiment-Analysis-on-Product-Reviews.git
cd Sentiment-Analysis-on-Product-Reviews
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud jupyter
```

4. **Download required NLTK data**

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## 🚀 Usage

Launch Jupyter Notebook and open the main analysis file:

```bash
jupyter notebook analysis.ipynb
```

Run all cells sequentially to reproduce the complete pipeline — from raw text ingestion through preprocessing, EDA, model training, and evaluation.

---

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computing |
| `nltk` | Text preprocessing (tokenization, stopwords, stemming) |
| `scikit-learn` | Feature extraction, ML models, evaluation metrics |
| `matplotlib` & `seaborn` | Data visualization |
| `wordcloud` | Word cloud generation |
| `re` | Regular expression-based text cleaning |

---

## 👤 Author

**Soumyadeep Chattopadhyay**

[![GitHub](https://img.shields.io/badge/GitHub-SoumyadeepChattopadhyay2004-181717?logo=github)](https://github.com/SoumyadeepChattopadhyay2004)

---

## 📄 License

This project is open-source and available for educational and research purposes.

---

> ⭐ If you found this project useful, please consider giving it a star!
