# 🎵 Music Genre Classification 🎧

A machine learning project to predict the genre of a music file using the GTZAN dataset and a deep learning model trained on audio signal features.

---

## 📚 Dataset

- **GTZAN Genre Collection**  
  - 10 genres: Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock  
  - 1000 audio files, 30s each, `.wav` format  
  - Source: [marsyas.info](http://marsyas.info/downloads/datasets.html)

---

## 🚀 Features

- 🎶 Accepts `.mp3` or `.wav` audio files
- 🧠 Uses a trained deep learning model (CNN) for classification
- 📊 Trained on Mel Spectrograms extracted from GTZAN dataset
- 🔌 REST API with FastAPI or Streamlit interface
- 🌍 Integration-ready for MERN stack frontend

---

## 🛠️ Tech Stack

- Python 3.9+
- TensorFlow / Keras
- Librosa (audio preprocessing)
- NumPy, Scikit-learn
- FastAPI (API server)
- Streamlit (optional frontend)

---



## 📈 Model Performance

- **Training Accuracy**: ~98%  
- **Validation Accuracy**: ~92%  
- Evaluated using 80/20 split on the GTZAN dataset  
- Audio converted into Mel spectrograms for input to CNN

