# 🌱 NomaAI

## AI-Assisted Tomato Plant Disease Detection System

NomaAI is an AI-powered application designed to assist users in identifying possible tomato plant diseases from tomato leaf images.

The system uses a deep-learning image classification model trained to recognize 10 different tomato leaf conditions.

---

## 🎯 Project Goal

The goal of NomaAI is to make AI-assisted tomato plant disease detection more accessible and easier to use.

NomaAI can assist:

- Farmers
- Agricultural learners
- Students
- Researchers
- Tomato growers
- Anyone interested in tomato plant health

---

## 🧠 NomaAI V3 Model

NomaAI currently uses **NomaAI V3**, a TensorFlow/Keras image classification model.

### Model specifications

| Specification | Value |
|---|---|
| Model | NomaAI V3 |
| Framework | TensorFlow / Keras |
| Input size | 128 × 128 pixels |
| Channels | RGB (3) |
| Number of classes | 10 |
| Test accuracy | 94.59% |
| Model file | `plant_disease_model_v3.keras` |

---

## 🍅 Supported Tomato Conditions

NomaAI V3 recognizes the following 10 classes:

1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Spider Mites
7. Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Healthy Tomato Leaf

---

## ✨ Features

### Disease Detection

Users can upload a tomato leaf image and receive an AI prediction.

### Image Quality Check

NomaAI performs basic checks to identify images that may be too small, too dark, too bright, or otherwise unsuitable for analysis.

### Confidence Score

The application displays the model's confidence for its top prediction.

### Top 3 Predictions

NomaAI also displays the three highest-scoring predictions from the model.

### Prediction History

Previous predictions can be saved and reviewed through the Prediction History page.

### About NomaAI

The application includes information about the project and its purpose.

### Model Information

Users can view information about the NomaAI V3 model and its supported classes.

---

## 📊 Model Performance

NomaAI V3 was evaluated using a separate test dataset containing **2,902 images across 10 classes**.

### Overall test performance

- Test Accuracy: **94.59%**
- Test Loss: **0.1520**
- Weighted F1 Score: **0.9461**
- Macro F1 Score: **0.9325**

These results indicate strong classification performance on the test dataset.

---

## 💻 Technology Stack

NomaAI is built using:

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow
- Scikit-learn

---

## 📁 Project Structure

```text
NomaAI/
│
├── app/
│   └── app.py
│
├── models/
│   ├── plant_disease_model.keras
│   ├── plant_disease_model_v2.keras
│   ├── plant_disease_model_v2_backup.keras
│   └── plant_disease_model_v3.keras
│
├── picture/
│   └── founder.jpg
│
├── data/
│
├── history/
│
├── historydir/
│
├── plantvillage/
│
├── training_data/
│
├── test_data/
│
├── Tomato/
│
├── create_training_split.py
├── create_test_split.py
├── evaluate_model.py
├── evaluate_v3.py
├── requirements.txt
└── README.md