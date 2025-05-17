import os
import pickle
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Define image dimensions
IMG_WIDTH, IMG_HEIGHT = 224, 224

def load_models(models_dir="models"):
    """
    Load all saved models from the specified directory
    """
    # Check if models directory exists
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Load label encoder
    with open(os.path.join(models_dir, 'label_encoder.pkl'), 'rb') as file:
        label_encoder = pickle.load(file)
    
    # Load scaler
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as file:
        scaler = pickle.load(file)
    
    # Load PCA model
    with open(os.path.join(models_dir, 'pca_model.pkl'), 'rb') as file:
        pca = pickle.load(file)
    
    # Load KNN model
    with open(os.path.join(models_dir, 'best_knn_model.pkl'), 'rb') as file:
        knn_model = pickle.load(file)
    
    print("All models loaded successfully!")
    return label_encoder, scaler, pca, knn_model

def load_feature_extractor():
    """
    Load and configure the MobileNetV2 model for feature extraction
    """
    # Load base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, 
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    
    # Add global pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Create feature extractor model
    feature_extractor = Model(inputs=base_model.input, outputs=x)
    print("Feature extractor loaded successfully!")
    return feature_extractor

def extract_features(image_path, feature_extractor):
    """
    Extract features from a single image using the CNN feature extractor
    """
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = mobilenet_preprocess_input(img_array_expanded)
        
        # Extract features
        features = feature_extractor.predict(img_preprocessed, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict_disease(image_path, models_dir="models"):
    """
    Main function to predict plant disease from a leaf image
    
    Args:
        image_path: Path to the image file
        models_dir: Directory containing the saved models
        
    Returns:
        predicted_disease: String with the predicted disease
        probability: Confidence score (if available)
    """
    # Load all models
    label_encoder, scaler, pca, knn_model = load_models(models_dir)
    
    # Load feature extractor
    feature_extractor = load_feature_extractor()
    
    # Extract features from the image
    features = extract_features(image_path, feature_extractor)
    
    if features is None:
        return "Error: Could not process the image", None
    
    # Reshape features for processing
    features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Apply PCA
    features_pca = pca.transform(features_scaled)
    
    # Predict using KNN model
    prediction = knn_model.predict(features_pca)[0]
    
    # Get probabilities if available (depends on KNN configuration)
    try:
        probabilities = knn_model.predict_proba(features_pca)[0]
        confidence = probabilities[prediction] * 100
    except:
        confidence = None
    
    # Convert numeric prediction back to disease name
    predicted_disease = label_encoder.inverse_transform([prediction])[0]
    
    return predicted_disease, confidence

def batch_predict(image_dir, models_dir="models"):
    """
    Predict diseases for all images in a directory
    """
    # Load all models
    label_encoder, scaler, pca, knn_model = load_models(models_dir)
    
    # Load feature extractor
    feature_extractor = load_feature_extractor()
    
    results = []
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        
        # Extract features
        features = extract_features(img_path, feature_extractor)
        
        if features is not None:
            # Process features
            features = features.reshape(1, -1)
            features_scaled = scaler.transform(features)
            features_pca = pca.transform(features_scaled)
            
            # Predict
            prediction = knn_model.predict(features_pca)[0]
            predicted_disease = label_encoder.inverse_transform([prediction])[0]
            
            results.append({
                'image': img_file,
                'predicted_disease': predicted_disease
            })
        else:
            results.append({
                'image': img_file,
                'predicted_disease': 'Error processing image'
            })
    
    return results

# Example usage
if __name__ == "__main__":
    # Single image prediction example
    test_image = "data/PlantVillage/Pepper__bell___Bacterial_spot/f92689ca-b5db-4a0a-b865-a69ba215922f___JR_B.Spot 9040.JPG"
    disease, confidence = predict_disease(test_image)
    
    if confidence:
        print(f"Predicted disease: {disease} (Confidence: {confidence:.2f}%)")
    else:
        print(f"Predicted disease: {disease}")