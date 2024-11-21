import mlflow
from typing import Any
import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

def create_mlflow_experiment(
    experiment_name: str, artifact_location: str, tags: dict[str, Any]
) -> str:
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except:
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name=experiment_name)
    return experiment_id


# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
data = pd.read_csv('fraudTrain.csv').sample(n=50000, random_state=42)
synthetic_data = pd.read_csv('synthetic_data3_2.csv')

# Define target and feature columns
target = 'is_fraud'
n_columns = ['merchant', 'category', 'job']  
ml_columns = ['amt', 'lat', 'long', 'city_pop'] 

# Initialize NLP tools
stop_words = set(stopwords.words('english'))

# Preprocess text function
def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'[^A-Za-z\s]', '', text)  
    tokens = word_tokenize(text)  
    filtered_tokens = [word for word in tokens if word not in stop_words]  
    return ' '.join(filtered_tokens)  

# fill missing values with mean for numeric and frequency for category
data[ml_columns].fillna(data[ml_columns].mean(), inplace=True)
data[n_columns].fillna(data[n_columns].mode().iloc[0], inplace=True)
synthetic_data[ml_columns].fillna(data[ml_columns].mean(), inplace=True)
synthetic_data[n_columns].fillna(data[n_columns].mode().iloc[0], inplace=True)

# Combine NLP columns into one text feature for preprocessing
data['combined_text'] = data[n_columns].agg(' '.join, axis=1)
synthetic_data['combined_text'] = synthetic_data[n_columns].agg(' '.join, axis=1)

# Apply text preprocessing
data['combined_text'] = data['combined_text'].apply(preprocess_text)
synthetic_data['combined_text'] = synthetic_data['combined_text'].apply(preprocess_text)

# Set up the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(data['combined_text'])  

# Transform the combined text into TF-IDF features
nlp_features_train = tfidf_vectorizer.transform(data['combined_text']).toarray()
synthetic_nlp_features_test = tfidf_vectorizer.transform(synthetic_data['combined_text']).toarray()

# Scale numeric features
scaler = MinMaxScaler()
scaled_numeric_features = scaler.fit_transform(data[ml_columns])
synthetic_scaled_numeric_features_test = scaler.transform(synthetic_data[ml_columns])

# Combine numeric and NLP features
X = np.hstack((scaled_numeric_features, nlp_features_train))
y = data[target]

synthetic_x_test = np.hstack((synthetic_scaled_numeric_features_test, synthetic_nlp_features_test))
synthetic_y_test = synthetic_data[target]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

Name_of_experiment = "ML_model_2"
experiment_id = create_mlflow_experiment(
        experiment_name=Name_of_experiment,
        artifact_location="ML_model_2",
        tags={"env": "dev", "version": "1.0.2"},
        )
mlflow.set_experiment(Name_of_experiment)
with mlflow.start_run() as run:
    print(run.info.run_id)
    print(experiment_id)

    # Log number of samples and features for training data
    mlflow.log_param("num_rows_train", data.shape[0])  # Number of rows
    mlflow.log_param("num_columns_train", data.shape[1])  # Number of columns
    mlflow.log_param("num_features_train", X.shape[1])  # Number of features
    
    # Log the number of unique labels in the training data
    unique_labels_train = np.unique(y)
    mlflow.log_param("num_unique_labels_train", len(unique_labels_train))

    # Train Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_resampled, y_train_resampled)

    # Evaluate the model on the training data
    y_predict = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    print('Accuracy:', accuracy)

    # Log accuracy for training data
    mlflow.log_metric("accuracy_train", accuracy)

    # Log number of samples and features for synthetic data
    mlflow.log_param("num_rows_synthetic", synthetic_data.shape[0])  # Number of rows
    mlflow.log_param("num_columns_synthetic", synthetic_data.shape[1])  # Number of columns
    mlflow.log_param("num_features_synthetic", synthetic_x_test.shape[1])  # Number of features
    
    # Log the number of unique labels in the synthetic data
    unique_labels_synthetic = np.unique(synthetic_y_test)
    mlflow.log_param("num_unique_labels_synthetic", len(unique_labels_synthetic))

    # Evaluate the model on synthetic data
    y_synthetic_predict = rf_model.predict(synthetic_x_test)
    synthetic_accuracy = accuracy_score(synthetic_y_test, y_synthetic_predict)
    print('Synthetic Accuracy:', synthetic_accuracy)

    # Log accuracy for synthetic data
    mlflow.log_metric("accuracy_synthetic", synthetic_accuracy)

# Save the trained model, scaler, and TF-IDF Vectorizer
# joblib.dump(rf_model, 'rf_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')