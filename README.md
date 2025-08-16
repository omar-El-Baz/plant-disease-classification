![GitHub top language](https://img.shields.io/github/languages/top/omar-El-Baz/plant-disease-classification?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/omar-El-Baz/plant-disease-classification?style=flat-square)

A machine learning project that demonstrates a hybrid approach to classifying plant diseases from leaf images. This system achieves **88.57% test accuracy** on the PlantVillage dataset by leveraging a pre-trained Convolutional Neural Network (MobileNetV2) for powerful feature extraction, followed by PCA for dimensionality reduction, and a K-Nearest Neighbors (KNN) classifier for the final prediction.

**Dataset:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) (15 classes, 20,638 images)

## 📋 Project Summary

The core challenge was to adapt the simple but effective KNN algorithm, which struggles with high-dimensional raw image data. This was solved by creating a robust feature engineering pipeline:

1.  **Feature Extraction:** A pre-trained **MobileNetV2** (without its top classification layer) was used to convert each 224x224 leaf image into a semantically rich 1280-dimensional feature vector.
2.  **Dimensionality Reduction:** **Principal Component Analysis (PCA)** was applied to the extracted features, reducing the dimensionality to 787 components while retaining ~95% of the variance. This improves KNN's efficiency and reduces noise.
3.  **Classification & Optimization:** A **K-Nearest Neighbors (KNN)** model was trained on these refined features. **GridSearchCV** was used to systematically find the optimal hyperparameters (`metric='cosine'`, `n_neighbors=7`, `weights='distance'`).
4.  **Prediction Pipeline:** A standalone Python script (`app.py`) demonstrates the end-to-end prediction process, loading the trained models to classify new, unseen images.

## 🚀 Key Results

| Metric                      | Value         |
| --------------------------- | ------------- |
| **Test Set Accuracy**       | **88.57%**    |
| Cross-Validation Accuracy   | 87.97%        |
| Optimal PCA Components      | 787           |
| Variance Explained by PCA   | 94.88%        |

## 🛠️ How to Run This Project

### 1. Prerequisites

- Python 3.8+
- An environment manager like `venv` or `conda`.
- [Git](https://git-scm.com/)

### 2. Setup & Installation

**Clone the repository:**
```bash
git clone https://github.com/omar-El-Baz/plant-disease-classification.git
cd plant-disease-classification
```

**Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
**Install the required dependencies**
```bash
pip install -r requirements.txt
```

## Download the dataset
1. Download the "PlantVillage" dataset from the Kaggle link above.
2. Create a directory named data at the root of the project.
3. Extract the downloaded archive and place the PlantVillage folder inside the data directory. The final path should look like this: ./data/PlantVillage/.

## 3. Running the Pipeline

### Step 1 — Train the Model
1. Open and run all cells in the `model_training.ipynb` Jupyter Notebook.
2. This will perform all data processing and save the trained model files (`.pkl`) into the `models/` directory.

### Step 2 — Make a Prediction
Run the prediction script from your terminal to classify a sample image:
```bash
python app.py
```
**Note:** You can modify the `image_path` variable in `app.py` to test your own images.

```text
Project Root
├── data/                     # Dataset directory
│   └── PlantVillage/         # <-- Place dataset here
├── models/                   # Saved model files
│   ├── best_knn_model.pkl
│   ├── label_encoder.pkl
│   ├── pca_model.pkl
│   └── scaler.pkl
├── notebooks/                # Jupyter notebooks
│   └── model_training.ipynb  # Main notebook for training & evaluation
├── app.py                    # Script for making predictions
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## 👥 Team Members
- **Moataz Ahmed Samir**
- **Malak Gehad**
- **Omar El-Sayed**
