# Fish Classification Using ANN - Akbank Deep Learning Bootcamp

Welcome to my project for the **Akbank Deep Learning Bootcamp** organized in collaboration with **10million.ai**.  
This project aims to classify fish species based on images using an Artificial Neural Network (ANN) model.

## 📑 Project Overview
This repository contains the code and resources for building an ANN-based fish classifier.  
The dataset used is "**A Large Scale Fish Dataset**", which includes **9 different fish classes**.  
Since the bootcamp's requirement was to use **only ANN architectures**, this project solely leverages a multi-layer ANN model.

## 📂 Dataset
- **Name**: A Large Scale Fish Dataset  
- **Directory Path**: `/kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset`  
- **Classes**: 9 fish types, 1000 instances each fish class.
- **Size**: 3 GB

## 📦 Libraries Used
- **TensorFlow/Keras** for building the model
- **Pandas, NumPy** for data handling
- **Matplotlib, Seaborn** for visualizations
- **Scikit-Learn** for evaluation metrics

## 🛠 Model Architecture
The project follows a **simple ANN architecture**:
1. **Input Layer**: Flattened 3D image input
2. **Multiple Dense Layers**: 
   - 1024 → 512 → 256 → 128 neurons  
   - **ReLU activation** function used in each hidden layer
3. **Output Layer**: Softmax activation to classify into 9 classes

### 🔧 Regularization and Optimization
- **Normalization** RGB images normalized into (-1,1)
- **Batch Normalization** layers added to prevent overfitting
- **Adam Optimizer** used with **learning rate scheduling**
- **ModelCheckpoint** to get the best checkpoint

## 📝 Evaluation Metrics
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report**
- **Loss and accuracy graphs**

## 🔥 Results
The model achieved promising results with improved validation accuracy using **-1, 1 scaling** compared to [0, 1] normalization.
Finally, it climbed to ~0.92 accuracy points.

## 🚀 How to Inspect
**Click the link below**: https://www.kaggle.com/code/emiryilmazey/akbank-deep-learning-bootcamp
**Or inspect main.ipynb from github**
