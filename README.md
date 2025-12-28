# Spam-Detection

## Overview

This repository contains a machine learning-based spam detection system that classifies emails as spam or not spam using Logistic Regression. The model is trained on email features and can be easily deployed for real-time predictions.

## Structure
    ├── SpamDetection.py          # Training script for the spam detection model
    ├── LoadSpamDetection.py      # Deployment script for making predictions
    ├── spamDetection.pkl         # Saved trained model (generated after training)
    ├── spamDetectionScaler.pkl   # Saved scaler for feature normalization (generated after training)
    └── spam_detection_dataset.csv # Dataset used for training (not included in repo)

## Features

- Machine Learning Model: Logistic Regression with class weighting to handle imbalanced data

- Feature Scaling: StandardScaler for normalizing input features

- Model Persistence: Save and load trained models using joblib

- Easy Deployment: Simple script for making real-time predictions

## Model Features

The model uses the following 5 features for spam detection:

1. Number of links - Integer count of links in the email
2. Number of words - Integer count of words in the email
3. Has offer - Binary (0 or 1) indicating if the email contains offers
4. Sender score - Float score representing sender reputation
5. All caps - Binary (0 or 1) indicating if email is in all capital letters

## Installation & Setup

1. Clone the repository

    git clone https://github.com/Nikk-hub-code/Spam-Detection.git
    cd Spam-Detection

2. Install required packages

    pip install pandas scikit-learn joblib

## Dataset Preparation

Prepare your dataset as a CSV file named `spam_detection_dataset.csv` with the following columns:

`num_link`: Number of links in the email
`num_words`: Number of words in the email
`has_offer`: Binary indicator (0 or 1) for offers
`sender_score`: Sender reputation score (float)
`all_caps`: Binary indicator (0 or 1) for all caps
`is_spam`: Target variable (0 = not spam, 1 = spam)

## Training the Model

    python SpamDetection.py

This will:

1. Load and preprocess the dataset
2. Scale the features using StandardScaler
3. Split data into training and testing sets (80/20)
4. Train a Logistic Regression model with class weights
5. Evaluate model performance
6. Save the trained model and scaler as `.pkl` files

## Model Configuration

- Algorithm: Logistic Regression
- Class weights: {0: 1, 1: 3} (penalizes misclassifying spam more heavily)
- Test size: 20% of data
- Random state: 42 for reproducibility

## Making Prediction

Run the prediction script:

    python LoadSpamDetection.py

You will be prompted to enter:

    Enter number of links: 
    Enter number of words: 
    Enter has offer (0 or 1): 
    Enter sender score: 
    Enter all caps (0 or 1):
The script will then output whether the email is classified as spam or not.

## Model Evaluation

The training script outputs:

- Accuracy Score: Overall prediction accuracy
- Confusion Matrix: True/False Positive/Negative counts
- Classification Report: Precision, recall, and F1-score for each class

## Dependencies

- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning algorithms and utilities
- `joblib`: Model serialization and persistence
