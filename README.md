# Plant Disease Classification using KNN and CNN Features

This project implements a system for classifying plant diseases from leaf images. It utilizes a K-Nearest Neighbors (KNN) classifier trained on features extracted by a pre-trained MobileNetV2 Convolutional Neural Network (CNN) and further refined by Principal Component Analysis (PCA).


**Dataset:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

## Project Overview

The primary goal is to accurately identify plant diseases based on leaf imagery to assist in agricultural settings. The system involves:
1.  **Feature Extraction:** Using a pre-trained MobileNetV2 to convert images into meaningful numerical feature vectors.
2.  **Dimensionality Reduction:** Applying PCA to the extracted features to reduce complexity and noise.
3.  **Classification:** Training a KNN model on the processed features.
4.  **Prediction:** A Python script (`app.py`) to predict diseases on new images using the trained models.

The main training and evaluation pipeline is detailed in the Jupyter Notebook: `model_training.ipynb`.


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/omar-El-Baz/plant-disease-classification.git
    cd plant-disease-classification
    ```

2.  **Create a Python Environment (Recommended):**
    It's good practice to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    You will need Python 3.x and the following major libraries. You can install them using pip:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python Pillow
    ```
    (Alternatively, if you create a `requirements.txt` file, users can run `pip install -r requirements.txt`)

4.  **Download the Dataset:**
    *   Go to [https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease).
    *   Download the dataset.
    *   Extract the contents and ensure you have a folder named `PlantVillage` containing subfolders for each class (e.g., `Pepper__bell___Bacterial_spot`, `Tomato___healthy`).
    *   Place the `PlantVillage` folder inside a `data` directory at the root of this project. The path used in the notebook is relative, e.g., `../data/PlantVillage` or `data/PlantVillage` depending on your notebook's specific path setup. Adjust the `dataset_path` variable in the notebook (`[Your_Notebook_Name].ipynb`) if your structure differs. **The notebook OCR shows `dataset_path = r"../data//PlantVillage"`, implying the notebook might be in a subdirectory and `data` is one level up.** Please ensure this matches your setup.

## Running the Project

### 1. Training the Model (Jupyter Notebook)

1.  Ensure your Python environment is activated and all dependencies are installed.
2.  Make sure the PlantVillage dataset is correctly placed in the `data/PlantVillage/` directory relative to the notebook's expectations.
3.  Open and run the `[Your_Notebook_Name].ipynb` Jupyter Notebook. This will:
    *   Load and preprocess the data.
    *   Extract features using MobileNetV2.
    *   Apply PCA.
    *   Train and tune the KNN model using GridSearchCV.
    *   Evaluate the model and display results (accuracy, classification report, confusion matrix).
    *   Save the `label_encoder.pkl`, `scaler.pkl`, `pca_model.pkl`, and `best_knn_model.pkl` files into the `models/` directory (the notebook OCR shows saving to `../models/`, ensure this directory exists or is created, and that `app.py` loads from the correct relative path).

    **Note:** The feature extraction and model training steps can be time-consuming.

### 2. Making Predictions (app.py)

1.  Once the notebook has been run and the `.pkl` model files are saved in the `models/` directory.
2.  You can run the `app.py` script to predict the disease of a sample image.
3.  Modify the `test_image` variable in the `if __name__ == "__main__":` block of `app.py` to point to an image you want to classify. The current example path is:
    `test_image = "data/PlantVillage/Pepper__bell___Bacterial_spot/f92689ca-b5db-4a0a-b865-a69ba215922f___JR_B.Spot 9040.JPG"`
    Ensure this path is valid relative to where you run `app.py`.
4.  Run the script from your terminal:
    ```bash
    python3 app.py
    ```
    Output will show the loaded models and the predicted disease for the test image.

## Results Summary

*   **CNN Feature Extractor:** MobileNetV2
*   **Dimensionality Reduction:** PCA (787 components, ~94.88% variance explained)
*   **Classifier:** K-Nearest Neighbors
*   **Best KNN Hyperparameters:** `{'metric': 'cosine', 'n_neighbors': 7, 'weights': 'distance'}`
*   **Cross-Validation Accuracy:** ~0.8797
*   **Test Set Accuracy:** ~0.8857

Detailed per-class metrics and confusion matrix can be found in the output of the Jupyter Notebook.

## Potential Future Work

*   Implement data augmentation techniques.
*   Explore advanced methods for handling class imbalance (e.g., SMOTE).
*   Experiment with other classifiers (SVM, Random Forest) or fine-tune the CNN.
*   Integrate environmental features if such data becomes available.
*   Develop a web application interface for easier use.
