# 🤖 Futuristic NLP Sentiment Analyzer

A cutting-edge NLP project that analyzes movie reviews and predicts their sentiment (positive/negative) using **TF-IDF + Logistic Regression**. It's packed with rich **3D visualizations**, **animated heatmaps**, and **ML explainability tools** — all wrapped in an interactive pipeline, topped off with a chatbot.

---

## 🧠 Table of Contents
- [📌 Project Highlights](#-project-highlights)
- [📂 Dataset](#-dataset)
- [⚙️ Tech Stack](#️-tech-stack)
- [🧹 NLP Pipeline](#-nlp-pipeline)
- [📈 Model Performance](#-model-performance)
- [📊 Visualizations](#-visualizations)
- [🤖 Chatbot Interface](#-chatbot-interface)
- [💡 How to Run](#-how-to-run)
- [📸 Output Gallery](#-output-gallery)
- [🙌 Credits](#-credits)

---

## 📌 Project Highlights

| Feature              | Description |
|----------------------|-------------|
| 🧹 Text Preprocessing | Lowercasing, punctuation removal, lemmatization, and stopword removal using **SpaCy + NLTK** |
| 🧠 Model             | **TF-IDF + Logistic Regression** with 88% accuracy |
| 🔮 Prediction        | Real-time review classification |
| 📊 3D Visuals        | PCA-based dimensionality reduction for interactive 3D plots |
| 🔥 Heatmaps          | Animated heatmaps showing sentiment evolution |
| ☁️ WordClouds        | Dual views for positive & negative vocabularies |
| 🤖 Chatbot           | Real-time sentiment prediction using custom text input |

---

## 📂 Dataset
- **Source**: Hugging Face [`imdb`](https://huggingface.co/datasets/imdb)
- **Size**: 25,000 training + 25,000 test reviews

```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

## ⚙️ Tech Stack

| Component      | Libraries Used                               |
|----------------|----------------------------------------------|
| NLP            | SpaCy, NLTK                                  |
| ML Modeling    | Scikit-learn                                 |
| Visualization  | Matplotlib, Seaborn, Plotly, WordCloud       |
| Data Handling  | Pandas, HuggingFace Datasets                 |

## 🧹 NLP Pipeline

1. Lowercase & punctuation removal
2. Tokenization & stopword filtering
3. Lemmatization using SpaCy
4. TF-IDF Vectorization

✅ Final TF-IDF Shape: (800, 5000)

## 📈 Model Performance

- Accuracy: 88%
- Precision: 0.93 (Neg) / 0.84 (Pos)
- Recall: 0.82 (Neg) / 0.94 (Pos)
- F1 Score: 0.88

## 📊 Visualizations

### 📌 Confusion Matrix
Clear separation of true positives and negatives.

![Confusion Matrix](https://github.com/user-attachments/assets/893e8ee1-f61a-4660-a3e7-e6d39fe214db)

### 📌 3D PCA Scatter Plot
Actual vs predicted samples in 3D sentiment space.

![3D PCA Plot](https://github.com/user-attachments/assets/e353a703-8345-4786-b417-7fada29668cd)

### 📌 Classification Report

![Classification Report](https://github.com/user-attachments/assets/f2fc190e-0d85-440c-946f-0d7db19dd7d8)

### 📌 Animated Heatmap
Visualizes sentiment confidence over time.

![Animated Heatmap](https://github.com/user-attachments/assets/8f08f13a-8cb9-450d-9856-973f0601d9d9)

### 📌 Feature Importance (Weights)
Shows most influential words contributing to classification.

![Feature Importance](https://github.com/user-attachments/assets/a241079b-6732-4991-a5a8-01c2641f3cda)

### 📌 Dual WordClouds
Common words in Positive vs Negative reviews.

## 🤖 Chatbot Interface

```python
predict_sentiment("I absolutely loved the movie! The acting was amazing.")
```

Sample Results:
- "This movie was so boring and predictable" → Negative
- "One of the best performances I've ever seen!" → Positive

## 💡 How to Run

1. Clone this repo or open NLP.ipynb in Google Colab
2. Install dependencies:
```bash
pip install -q spacy wordcloud scikit-learn plotly nltk
python -m spacy download en_core_web_sm
```
3. Run all cells.
4. Use the chatbot section to test real-time inputs!

## 📸 Output Gallery

- ✅ Confusion Matrix
- ✅ PCA 3D Plot
- ✅ Heatmap Animation
- ✅ Feature Importance
- ✅ WordClouds
- ✅ Chatbot Predictions

📂 All included in the Visualizations Section

## 🙌 Credits

- Dataset: IMDb Dataset from Hugging Face
- ML Engineered & Visualized by: Mayank Shekhar

🚀 This project is part of my AI journey — aiming to combine technical skill, visual storytelling, and intelligent interactivity to build real-world ML solutions.
