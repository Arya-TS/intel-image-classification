# Intel Image Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify natural scene images into six categories using the Intel Image Classification dataset.  
A **Streamlit web application** is built on top of the trained model to allow interactive image predictions.

---

## 📌 Problem Statement
Given an input image, classify it into one of the following scene categories:

- Buildings  
- Forest  
- Glacier  
- Mountain  
- Sea  
- Street  

This is a **multi-class image classification problem**.

---

## 🧠 Approach

- Built a **custom CNN architecture** using `tf.keras`
- Used **ImageDataGenerator** for efficient data loading and data augmentation
- Trained the model on the Intel Image Classification dataset
- Evaluated the model using accuracy and classification report
- Deployed the trained model using **Streamlit** for real-time predictions

---

## 🏗️ Model Architecture

- Input shape: `150 × 150 × 3`
- Convolution + ReLU + MaxPooling blocks
- Fully connected Dense layers
- Output layer with **Softmax activation** (6 classes)

Frameworks used:
- **TensorFlow** (backend computation engine)
- **Keras** (high-level deep learning API)

---

## 📊 Model Performance

- **Test Accuracy:** ~84%

Observations:
- Strong performance on `forest`, `street`, and `sea`
- Some confusion between visually similar classes such as `mountain` and `glacier`

---

## 🖥️ Streamlit Web Application

The Streamlit UI allows users to:
- Upload an image
- Run inference using the trained CNN model
- View the predicted class and confidence score

---

⚙️ Installation & Usage
1. Clone the repository

   git clone https://github.com/Arya-TS/intel-image-classification.git
   cd Intel-Image-Classification

2. Install dependencies

   pip install -r requirements.txt

3. Run the Streamlit app

   streamlit run app.py

🧪 Dataset

- Intel Image Classification Dataset
- Source: Kaggle
- Approximately 25,000 images across 6 scene categories

🚀 Future Improvements

- Apply transfer learning (ResNet, MobileNet, EfficientNet)
- Improve classification of visually similar classes
- Add Grad-CAM visualizations for interpretability
- Deploy the application on Streamlit Cloud or Hugging Face Spaces

🧑‍💻 Author

- This project was built to gain hands-on experience in:
- Convolutional Neural Networks (CNNs)
- TensorFlow & Keras
- End-to-end machine learning workflows
- Model deployment using Streamlit
