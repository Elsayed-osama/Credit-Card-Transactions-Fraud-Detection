import streamlit as st
import numpy as np
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the trained Random Forest model, scaler, and TF-IDF vectorizer
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Set up the NLP preprocessing tools
stop_words = set(stopwords.words('english'))

# Define the text preprocessing function
def preprocessing_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = word_tokenize(text)  # Tokenize the text
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(filtered_tokens)  # Return as a cleaned string

# Define a function to preprocess and predict using the model
def preprocess_and_predict(input_data):
    # Apply MinMax Scaling on numerical columns
    scaled_cols = scaler.transform([[input_data['amt'], input_data['lat'], input_data['long'], input_data['city_pop']]])

    # Combine NLP columns into one text feature
    combined_text = " ".join([input_data.get(col, "") for col in ['merchant', 'category', 'job']])
    
    # NLP Preprocessing: Convert combined text to TF-IDF vector
    preprocessed_text = preprocessing_text(combined_text)
    nlp_features = tfidf_vectorizer.transform([preprocessed_text]).toarray()

    # Check if the TF-IDF vector is all zeros (i.e., no known words in the input text)
    if np.all(nlp_features == 0):
        st.warning("The input text contains no known terms, model may not have enough information.")
        return 0.5  # Return 0.5 as an uncertainty score

    # Combine numerical and NLP features
    combined_features = np.hstack((scaled_cols, nlp_features))

    # Use the Random Forest model to make a prediction
    prediction_prob = rf_model.predict_proba(combined_features)

    # Return the fraud probability (probability of class 1)
    fraud_prob = prediction_prob[0][1]
    return fraud_prob

# Set up the Streamlit app
st.title("Online Payment Fraud Detection")
st.write("Enter the details to check for potential fraud based on numerical and text data.")

# Input form
with st.form("fraud_detection_form"):
    # Collect user inputs
    merchant = st.text_input("Merchant Name", help="Enter the name of the merchant or service provider.")
    category = st.text_input("Category", help="Specify the transaction category (e.g., Grocery, Electronics).")
    job = st.text_input("Job", help="Input your job title or the job title associated with the transaction.")
    amt = st.number_input("Transaction Amount", min_value=0.0, step=0.01, help="Enter the amount of the transaction in currency.")
    lat = st.number_input("Latitude", step=0.0001, help="Provide the latitude of the transaction location.")
    long = st.number_input("Longitude", step=0.0001, help="Provide the longitude of the transaction location.")
    city_pop = st.number_input("City Population", min_value=0, step=1, help="Enter the population of the city where the transaction occurred.")

    # Submit button
    submitted = st.form_submit_button("Predict Fraud")

    if submitted:
        # Validate the inputs before proceeding
        if not merchant or not category or not job:
            st.error("Please fill out all text fields.")
        elif amt <= 0 or lat == 0 or long == 0 or city_pop <= 0:
            st.error("Please provide valid values for numerical inputs.")
        else:
            # Prepare the input data as a dictionary
            input_data = {
                'merchant': merchant,
                'category': category,
                'job': job,
                'amt': amt,
                'lat': lat,
                'long': long,
                'city_pop': city_pop
            }

            # Get the prediction
            fraud_prob = preprocess_and_predict(input_data)

            # Display the result based on fraud probability
            if fraud_prob > 0.5:  # Adjust the threshold as needed
                st.error(f"This transaction has a fraud probability of {fraud_prob:.2f}. High risk!")
            else:
                st.success(f"This transaction has a fraud probability of {fraud_prob:.2f}. Low risk.")
