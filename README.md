# Plant Disease Classification using CNN-Extracted Features and KNN

![GitHub top language](https://img.shields.io/github/languages/top/omar-El-Baz/plant-disease-classification?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/omar-El-Baz/plant-disease-classification?style=flat-square)

A machine learning project that demonstrates a hybrid approach to classifying plant diseases from leaf images. This system achieves **97.77% test accuracy** on the PlantVillage dataset by leveraging a pre-trained Convolutional Neural Network (ResNet-50) for powerful feature extraction, followed by Linear Discriminant Analysis (LDA) for dimensionality reduction, and a K-Nearest Neighbors (KNN) classifier for the final prediction.

**Dataset:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) (15 classes, 20,638 images)

## 🎯 Project Context & Objective

This project was developed to fulfill a university course requirement with a specific and challenging constraint: to utilize the K-Nearest Neighbors (KNN) algorithm for classification.

**Original Assignment Prompt:**

> "Use KNN to classify plant diseases based on leaf images or other features such as
temperature, humidity, and soil conditions. You have a plant leafs images or
environmental features, train a KNN model to classify plant diseases based on
input features. This can assist in agricultural settings where diagnosing plant
health quickly is crucial for preventing crop damage."

Recognizing that KNN is traditionally ill-suited for the high-dimensionality of raw image data, our team's primary engineering challenge was to design a sophisticated feature extraction and dimensionality reduction pipeline to make the KNN classifier effective. This project showcases a solution that works robustly within the given constraints while addressing the "curse of dimensionality."

## 📋 Technical Summary

Our solution transforms raw image data into a low-dimensional, feature-rich format ideal for KNN classification through the following pipeline:

1. **Feature Extraction:** A pre-trained **ResNet-50** (ImageNet weights, without its top classification layer) was used to convert each 224x224 leaf image into a semantically rich 2048-dimensional feature vector through global average pooling.

2. **Dimensionality Reduction:** **Linear Discriminant Analysis (LDA)** was applied to the extracted features, dramatically reducing the dimensionality from 2048 to just **14 components** (>99% reduction) while maximizing class separability for optimal KNN performance.

3. **Data Augmentation:** To address class imbalance in the dataset, we implemented comprehensive data augmentation including rotation, flipping, brightness adjustment, and zoom transformations to balance underrepresented classes.

4. **Classification & Optimization:** A **K-Nearest Neighbors (KNN)** model was trained on these refined features. **GridSearchCV with StratifiedKFold** cross-validation was used to systematically find the optimal hyperparameters (`metric='euclidean'`, `n_neighbors=7`, `weights='distance'`).

5. **Prediction Pipeline:** A standalone Python script (`app.py`) demonstrates the end-to-end prediction process, loading the trained models to classify new, unseen images.

## 🚀 Key Results

| Metric                      | Value         |
| --------------------------- | ------------- |
| **Test Set Accuracy**       | **97.77%**    |
| Cross-Validation Accuracy   | 99.39%        |
| Feature Dimensions (Original)| 2048         |
| Feature Dimensions (LDA)    | 14            |
| Dimensionality Reduction    | >99%          |
| Training Images            | 15,478        |
| Test Images                | 5,160         |

## 🔬 Technical Innovations

- **Hybrid Architecture:** Successfully combined deep learning feature extraction with traditional machine learning classification to overcome KNN's limitations with high-dimensional data
- **Extreme Dimensionality Reduction:** Achieved >99% feature reduction (2048 → 14) while maintaining exceptional accuracy through LDA's supervised approach
- **Class Imbalance Solution:** Implemented targeted data augmentation to balance classes ranging from 152 to 3,208 samples
- **Comprehensive Hyperparameter Optimization:** Evaluated 30 different parameter combinations across distance metrics, neighbor counts, and weighting schemes

## 🛠️ How to Run This Project

### 1. Prerequisites
- Python 3.8+
- An environment manager like `venv` or `conda`
- [Git](https://git-scm.com/)

### 2. Setup and Installation

**Clone the repository:**
```bash
git clone https://github.com/omar-El-Baz/plant-disease-classification.git
cd plant-disease-classification
```

**Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**Install the required dependencies:**
```bash
pip install -r requirements.txt
```

## Download the dataset
1. Download the "PlantVillage" dataset from the Kaggle link above
2. Create a directory named `data` at the root of the project
3. Extract the downloaded archive and place the PlantVillage folder inside the data directory. The final path should look like this: `./data/PlantVillage/`

## 3. Running the Pipeline

### Step 1 — Train the Model
1. Open and run all cells in the `model_training.ipynb` Jupyter Notebook
2. This will perform all data processing and save the trained model files (`.pkl`) into the `models/` directory

### Step 2 — Make a Prediction
Run the prediction script from your terminal to classify a sample image:
```bash
python app.py
```

**Note:** You can modify the `image_path` variable in `app.py` to test your own images.

## 📁 Project Structure

```text
Project Root
├── data/                     # Dataset directory
│   └── PlantVillage/         # <-- Place dataset here
├── models/                   # Saved model files
│   ├── best_knn_model.pkl    # Trained KNN classifier
│   ├── label_encoder.pkl     # Label encoding mappings
│   ├── lda_model.pkl         # LDA dimensionality reducer
│   └── scaler.pkl            # Feature standardization
├── notebooks/                # Jupyter notebooks
│   └── model_training.ipynb  # Main notebook for training & evaluation
├── app.py                    # Script for making predictions
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## 📊 Model Performance Details

The final model demonstrates exceptional performance across all 15 plant disease classes:

- **Precision:** 97.78% (macro average)
- **Recall:** 97.02% (macro average)
- **F1-Score:** 97.37% (macro average)

Classes include healthy and diseased variants of:
- Tomato (9 classes including healthy)
- Potato (3 classes including healthy)  
- Bell Pepper (2 classes including healthy)

## 🔧 Technical Dependencies

- **TensorFlow/Keras:** Deep learning framework for ResNet-50 feature extraction
- **Scikit-learn:** Traditional ML algorithms (KNN, LDA, preprocessing)
- **OpenCV & PIL:** Image processing and loading
- **NumPy & Pandas:** Data manipulation and numerical computing
- **Matplotlib & Seaborn:** Data visualization and analysis

## 👥 Team Members
- **Moataz Ahmed Samir**
- **Malak Gehad**
- **Omar El-Sayed**
