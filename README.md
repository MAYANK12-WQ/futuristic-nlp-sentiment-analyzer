# 🤖 Futuristic NLP Sentiment Analyzer

A cutting-edge NLP project that analyzes movie reviews and predicts their sentiment (positive/negative) using **TF-IDF + Logistic Regression**. It’s packed with rich **3D visualizations**, **animated heatmaps**, and **ML explainability tools** — all wrapped in an interactive pipeline, topped off with a chatbot.

---

## 🧠 Table of Contents
- [📌 Project Highlights](#-project-highlights)
- [📂 Dataset](#-dataset)
- [⚙️ Tech Stack](#️-tech-stack)
- [🧹 NLP Pipeline](#-nlp-pipeline)
- [📈 Model Performance](#-model-performance)
- [📊 Visualizations](#-visualizations)
- [🤖 Chatbot](#-chatbot)
- [💡 How to Run](#-how-to-run)
- [📸 Output Gallery](#-output-gallery)
- [🙌 Credits](#-credits)

---

## 📌 Project Highlights
| Feature | Description |
|--------|-------------|
| 🧹 Text Preprocessing | Lowercasing, punctuation removal, lemmatization, and stopword removal using SpaCy + NLTK |
| 🧠 Model | TF-IDF + Logistic Regression with 88% accuracy |
| 🔮 Prediction | Real-time review classification |
| 📊 3D Visuals | PCA-based dimensionality reduction for interactive 3D plots |
| 🔥 Heatmaps | Animated heatmaps showing sentiment evolution |
| ☁️ WordClouds | Dual views for positive & negative vocabularies |
| 🤖 Chatbot | Text-based prediction assistant built right into the notebook |

---

## 📂 Dataset
- **Source**: Hugging Face `imdb` dataset
- **Size**: 25,000 training + 25,000 test reviews

```python
from datasets import load_dataset
dataset = load_dataset("imdb")

## 📂 Tech Stack
Component	Libraries Used
NLP	SpaCy, NLTK
ML Modeling	Scikit-learn
Visualization	Matplotlib, Seaborn, Plotly, WordCloud
Data Handling	Pandas, HuggingFace Datasets

## 📂 NLP Pipeline

# Steps:
1. Lowercase and remove punctuation
2. Tokenize & remove stopwords
3. Lemmatize using SpaCy
4. Convert text to TF-IDF vectors

## 📂 Model Performance
Accuracy: 88%
Precision: 0.93 (Neg) / 0.84 (Pos)
Recall: 0.82 (Neg) / 0.94 (Pos)
F1 Score: 0.88

📊 Visualizations
![image](https://github.com/user-attachments/assets/46a6aef5-bd96-41b6-aa54-1439d41e6f07)
![newplot (4)](https://github.com/user-attachments/assets/e1afb449-6e5e-4d46-996f-3d1c5eceb91c)
![image](https://github.com/user-attachments/assets/2ab8da8d-8d4b-4c28-bf7d-006345ed924f)
![newplot (2)](https://github.com/user-attachments/assets/cc9c63b0-3a69-47d3-8a36-4cff97b4d9ad)
![newplot (3)](https://github.com/user-attachments/assets/997fa96f-e026-44f8-8427-8ced4adfab10)
![image](https://github.com/user-attachments/assets/52986efd-2130-4da6-8bf4-17fe826651d9)

🤖 Chatbot Interface
Interact directly with the model:

predict_sentiment("I absolutely loved the movie! The acting was amazing.")
Example Results:

"This movie was so boring and predictable" → Negative
"One of the best performances I've ever seen!" → Positive
💡 How to Run
Clone the repo or open the NLP.ipynb in Google Colab.
Install dependencies:

pip install -q spacy wordcloud scikit-learn plotly nltk
python -m spacy download en_core_web_sm
Run all cells and interact with the chatbot near the end!

🙌 Credits
Dataset: IMDb Dataset
Inspired by modern ML storytelling & dashboarding
Built by: Mayank Shekhar
🚀 This project is part of my AI journey — aiming to combine technical skill, visual storytelling, and intelligent interactivity to build real-world ML solutions.
