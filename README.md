# Handwritten Digit Recognition (MNIST) 🤖✍️

## 📌 Project Overview
This project implements a complete Machine Learning pipeline to classify handwritten digits (0-9) using the MNIST dataset. The goal was to compare classical machine learning algorithms, implement a model from scratch, and optimize performance using advanced techniques like Dimensionality Reduction (PCA) and Ensemble Learning.

## ⚙️ Technical Architecture
The system follows a strict execution pipeline as visualized below. It processes raw CSV data, splits it for validation, and runs parallel modeling paths (Manual, Standard, and Optimized) before converging for evaluation.

![Project Workflow](Flow_Diagram.png)
*(See `Flow_Diagram.png` in the repository for the full execution pipeline)*

## 🚀 Key Features
* **Preprocessing:** Data normalization (0-255 scaled to 0-1) and Train/Test splitting (80/20).
* **Custom Implementation:** Built a **K-Nearest Neighbors (KNN)** classifier entirely from scratch using NumPy to understand the underlying distance logic.
* **Standard Models:** Implemented and tuned **SVM (RBF Kernel)**, **KNN (Scikit-Learn)**, and **Decision Tree** classifiers.
* **Advanced Optimization (Bonus):**
    * **PCA (Principal Component Analysis):** Reduced feature space from 784 pixels to ~153 components (95% variance) to speed up training and remove noise.
    * **Voting Ensemble:** Combined predictions from all models to improve stability.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:**
    * `NumPy`, `Pandas` (Data Manipulation)
    * `Matplotlib`, `Seaborn` (Visualization)
    * `Scikit-Learn` (Model Training & Evaluation)

## 📊 Model Performance Results
| Model | Accuracy | Notes |
|-------|----------|-------|
| **SVM + PCA** | **98.20%** | 🏆 **Best Performing Model** |
| SVM (RBF Kernel) | 97.99% | Excellent accuracy, slightly slower than PCA version |
| Voting Ensemble | 97.11% | Robust, but held back by the weaker Decision Tree |
| KNN (k=3) | 96.67% | Strong baseline performance |
| Decision Tree | 85.82% | Prone to overfitting on pixel noise |
| Custom KNN (Scratch)| 89.00% | Validated logic on a data subset |

## 📂 Project Structure
```text
MNIST-Handwritten-Digit-Recognition/
│
├── Lakshay_Malia_Assignment.ipynb    #
├── Lakshay_Malia_Assignment.pdf      # PDF Report of the Notebook
├── Flow_Diagram.png                  # Technical Execution Flowchart
├── README.md                         # Project Documentation
├── LICENSE                           # MIT License
└── Outputs/                          
