#!/usr/bin/env python
# coding: utf-8

# # Speech Emotion Recognition using ML

# **Introduction**                        
# This project explores Speech Emotion Recognition (SER), which aims to detect and classify human emotions based on audio signals. Understanding emotions from speech is essential for enhancing human-computer interaction, virtual assistants, and mental health monitoring systems. Here, the **RAVDESS**(Ryerson Audio-Visual Database of Emotional Speech and Song) dataset has been used, which contains high-quality, professionally acted recordings representing eight distinct emotions. Using supervised machine learning techniques, this study extracts acoustic features from the audio files and builds classification models to accurately recognize emotional states from speech.

# **Support Vector Machine (SVM)**

# In[1]:


get_ipython().system('pip install librosa numpy soundfile scikit-learn ')
get_ipython().system('pip install praat-parselmouth')

# Importing required Python libraries
import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import parselmouth
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Writing a function to extract features from an audio file
def extract_features(file_path):
    try:
        audio, sample_rate = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        audio = audio.astype(float)

        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zcr.T, axis=0)

        # Chroma
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        contrast_mean = np.mean(contrast.T, axis=0)

        # HNR
        try:
            snd = parselmouth.Sound(file_path)
            hnr = snd.to_harmonicity()
            hnr_values = hnr.values[hnr.values != -200]
            hnr_mean = np.mean(hnr_values) if hnr_values.size > 0 else 0
        except:
            hnr_mean = 0

        # Combining all features
        combined = np.hstack([
            mfccs_mean,
            zcr_mean,
            chroma_mean,
            contrast_mean,
            [hnr_mean]
        ])
        return combined

    except Exception as e:
        print("Error extracting:", file_path, "→", e)
        return None
    
dataset_path = r"C:\Users\muska\Downloads\ravdess"    # Dataset Path
X_features = []
y_labels = []

#  Looping through all actor folders and WAV files
for emotion_label in os.listdir(dataset_path):
    emotion_folder = os.path.join(dataset_path, emotion_label)
    if os.path.isdir(emotion_folder):
        for file_name in os.listdir(emotion_folder):
            if file_name.endswith(".wav"):
                file_path = os.path.join(emotion_folder, file_name)
                features = extract_features(file_path)
                if features is not None:
                    X_features.append(features)
                    y_labels.append(emotion_label)

# Converting the lists to NumPy arrays for model processing                    
X_features = np.array(X_features)
y_labels = np.array(y_labels)

#  Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Splitting data into training and testing sets (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_labels, test_size=0.4, random_state=42, stratify=y_labels
)

# Defining hyperparameter grid 
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Using GridSearchCV for finding best parameters
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)


# Printing best parameters
print("Best Parameters:", grid.best_params_)

# Using the best model for predictions
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluating model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy of SVM: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# **Random Forest**

# In[2]:


# Importing required Python libraries
import os
import numpy as np
import librosa
import soundfile as sf
import parselmouth
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Writing Function for Feature Extraction
def extract_features(file_path):
    try:
        audio, sample_rate = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        audio = audio.astype(float)
         
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zcr.T, axis=0)

        #Chroma
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        contrast_mean = np.mean(contrast.T, axis=0)

        # HNR
        try:
            snd = parselmouth.Sound(file_path)
            hnr = snd.to_harmonicity()
            hnr_values = hnr.values[hnr.values != -200]
            hnr_mean = np.mean(hnr_values) if hnr_values.size > 0 else 0
        except:
            hnr_mean = 0

        # Combing all features
        combined = np.hstack([
            mfccs_mean,
            zcr_mean,
            chroma_mean,
            contrast_mean,
            [hnr_mean]
        ])
        return combined
    except Exception as e:
        print("Error extracting:", file_path, "→", e)
        return None

# Dataset Path
dataset_path = r"C:\Users\muska\Downloads\ravdess"
X_features = []
y_labels = []

#  Looping through all actor folders and WAV files
for emotion_label in os.listdir(dataset_path):
    emotion_folder = os.path.join(dataset_path, emotion_label)
    if os.path.isdir(emotion_folder):
        for file_name in os.listdir(emotion_folder):
            if file_name.endswith(".wav"):
                file_path = os.path.join(emotion_folder, file_name)
                features = extract_features(file_path)
                if features is not None:
                    X_features.append(features)
                    y_labels.append(emotion_label)

# Preprocessing
X = np.array(X_features)
y = np.array(y_labels)

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42, stratify=y
)

# Defining hyperparameter grid 
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Using GridSearchCV for finding best parameters
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Printing Best Parameters
print(" Best Parameters:", grid.best_params_)

# Using the best model for predictions
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)

# Evaluating model
accuracy = accuracy_score(y_test, y_pred)
print(f" Test Accuracy (Random Forest): {accuracy * 100:.2f}%")

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# **k-nearest neighbor algorithm**

# In[3]:


# Importing required Python libraries
import os
import numpy as np
import librosa
import soundfile as sf
import parselmouth
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Writing Function for Feature Extraction
def extract_features(file_path):
    try:
        audio, sample_rate = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        audio = audio.astype(float)
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        #ZCR
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zcr.T, axis=0)

        #Chroma
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        contrast_mean = np.mean(contrast.T, axis=0)

        #HNR
        try:
            snd = parselmouth.Sound(file_path)
            hnr = snd.to_harmonicity()
            hnr_values = hnr.values[hnr.values != -200]
            hnr_mean = np.mean(hnr_values) if hnr_values.size > 0 else 0
        except:
            hnr_mean = 0
            
        # Combing all features together
        combined = np.hstack([
            mfccs_mean,
            zcr_mean,
            chroma_mean,
            contrast_mean,
            [hnr_mean]
        ])
        return combined
    except Exception as e:
        print("Error extracting:", file_path, "→", e)
        return None


# Dataset Path
dataset_path = r"C:\Users\muska\Downloads\ravdess"
X_features = []
y_labels = []


#  Looping through all actor folders and WAV files
for emotion_label in os.listdir(dataset_path):
    emotion_folder = os.path.join(dataset_path, emotion_label)
    if os.path.isdir(emotion_folder):
        for file_name in os.listdir(emotion_folder):
            if file_name.endswith(".wav"):
                file_path = os.path.join(emotion_folder, file_name)
                features = extract_features(file_path)
                if features is not None:
                    X_features.append(features)
                    y_labels.append(emotion_label)

# Preprocessing
X = np.array(X_features)
y = np.array(y_labels)


# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42, stratify=y
)

# Defining hyperparameter grid 
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Using GridSearchCV for finding best parameters
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Printing Best Parameters
print("Best Parameters:", grid.best_params_)

# Using the best model for predictions
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test)

# Evaluating model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (KNN + GridSearch): {accuracy * 100:.2f}%")

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("KNN Confusion Matrix (Tuned)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In this project, I have applied Ensemble Learning techniques to improve the model's accuracy and robustness. By combining multiple classifiers, the ensemble approach helps in better generalization and performance compared to using a single model.
# 
# Ensemble methods help reduce overfitting and boost prediction accuracy by combining the strengths of different models.

# In[4]:


# Importing required Python libraries For Ensemble Learning
from sklearn.ensemble import VotingClassifier

# Creating the Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('svm', best_model),        # from SVM GridSearch
        ('rf', best_rf),            # from Random Forest GridSearch
        ('knn', best_knn)           # from KNN GridSearch
    ],
    voting='hard'  # Hard Voting
)

# Fit the Voting Classifier
voting_clf.fit(X_train, y_train)

# Predictions & Evaluation
y_pred_ensemble = voting_clf.predict(X_test)

#  Accuracy Checking
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
print(f" Ensemble Test Accuracy (Hard Voting): {ensemble_acc * 100:.2f}%")

# Detailed Report for Ensemble Learning
print("\n Ensemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble))

# Plotting Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_ensemble), annot=True, fmt='d', cmap='Greens')
plt.title("Hard Voting Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# **Evaluating Model Performance Using Cross-Validation**    
# To assess the performance and generalizability of the ensemble model (voting_clf), 5-fold cross-validation is performed using cross_val_score from sklearn.model_selection.

# In[5]:


from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(voting_clf, X, y, cv=5)
print(f"Cross-validated accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")


# This generates a detailed classification report including precision, recall, F1-score, and support for each class.      
# It helps evaluate the performance of the ensemble model on the test data.

# In[6]:


print(classification_report(y_test, y_pred_ensemble))


# In[7]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generating the confusion matrix from actual vs predicted labels
cm = confusion_matrix(y_test, y_pred_ensemble)

# Set figure size for better visibility
plt.figure(figsize=(12, 8))

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Set axis labels and title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Ensemble Model')

# Display the heatmap
plt.show()


# In[ ]:





# In[ ]:




