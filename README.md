**Speech Emotion Recognition using Ensemble Learning**
This project focuses on recognizing human emotions from speech using a machine learning-based approach. It uses feature extraction techniques and applies multiple classifiers with a Voting Ensemble Model to improve accuracy.

**Dataset Used**:- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
→ Contains 735 audio files in .wav format categorized by 8 emotions (happy, sad, angry, etc.).

**Objective**:- To build a Speech Emotion Recognition (SER) system using extracted audio features and train it using SVM, KNN, and Random Forest, combined through an ensemble Voting Classifier.

**Methodology**:-
1. **Feature Extraction**
Extracted key audio features using librosa:
MFCC (Mel-frequency cepstral coefficients)
ZCR (Zero Crossing Rate)
HNR (Harmonics to Noise Ratio)
Spectral Contrast

2. **Model Training & Evaluation**
Used GridSearchCV for hyperparameter tuning on:
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Random Forest Classifier
Combined best models using a Voting Classifier

3.**Testing**
Evaluated on unseen test data with accuracy, classification report, and confusion matrix.

**Results**
Ensemble model performed better than individual classifiers with accuracy of 92.53%.
High accuracy achieved on emotion classification


**Project Structure**
Speech_Emotion_Recognition/
│
├── data/                  Contains downloaded datasets (e.g., RAVDESS)
├── features/              Extracted features like MFCC, ZCR, HNR, Spectral Contrast
├── models/                Trained model files or model scripts
├── main.py                Main script to run training, testing, and evaluation
├── requirements.txt      
├── README.md            
└── .gitignore            

**Note:** The data/ folder is not included in the repository. Please download the dataset manually from the official source.
**License:** This project is private and not open for reuse.
