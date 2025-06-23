# Chatbot Deva

This project implements a basic intent-based chatbot named Deva. It utilizes a pre-existing dataset of user inputs and bot responses to train a classification model that predicts the user's intent, and then provides an appropriate response.

-----

## Features

  * **Data Loading and Preprocessing:** Handles reading an Excel dataset, cleaning missing values, converting text to lowercase, and removing special characters and numbers.
  * **Text Tokenization and Stopword Removal:** Uses NLTK for tokenizing user input and removing common English stopwords to prepare text for vectorization.
  * **Intent Classification:** Employs a **TF-IDF Vectorizer** and a **Logistic Regression** model to classify user input into predefined intents.
  * **Model Persistence:** Saves the trained model and vectorizer using `joblib` for later use without retraining.
  * **Response Generation:** Retrieves a relevant bot response based on the predicted intent.

-----

## Setup and Installation

### Prerequisites

  * Python 3.x
  * Jupyter Notebook or Google Colab (for running the provided code)

### Libraries

You can install the necessary Python libraries using pip:

```bash
pip install pandas openpyxl nltk scikit-learn joblib
```

*Note: The original code includes commented-out sections for `transformers` and `torch`. These are not actively used in the final version of the code provided for intent classification, which instead relies on TF-IDF and Logistic Regression.*

### Data

The chatbot requires an Excel file named `nexyug_chatbot_dataset_3000.xlsx`. This file should be located in your Google Drive at `/content/drive/MyDrive/Colab Notebooks/ChatBot Deva/`. The dataset is expected to have columns: `User Input`, `Bot Response`, and `Intent`.

-----

## Usage

The provided code demonstrates the training and usage of the chatbot within a Python environment (e.g., Jupyter Notebook or Google Colab).

### 1\. Mount Google Drive (if using Colab)

The first step in the notebook is to mount your Google Drive to access the dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2\. Run the Notebook Cells

Execute the cells sequentially to:

  * Load and preprocess the dataset.
  * Perform text tokenization and stopword removal.
  * Train the TF-IDF vectorizer and Logistic Regression classifier.
  * Save the trained model and vectorizer.
  * Load the saved model and vectorizer.
  * Test the `get_response` function with sample user inputs.

### 3\. Predicting Intent and Getting Responses

The core logic for getting a response involves:

1.  **Cleaning and normalizing** the user's input.
2.  **Vectorizing** the input using the trained TF-IDF vectorizer.
3.  **Predicting the intent** using the loaded Logistic Regression model.
4.  **Retrieving a bot response** from the dataset based on the predicted intent.

A simplified `get_response` function is provided, which can be extended for more robust response generation:

```python
def get_response(user_input):
    # ... (code for cleaning, vectorizing, predicting intent) ...
    user_input_vectorized = vectorizer.transform([cleaned_input])
    predicted_intent = clf.predict(user_input_vectorized)[0]

    # Retrieve response based on predicted_intent
    # This example retrieves a random response for the predicted intent
    response = df[df['Intent'] == predicted_intent]['Bot Response'].sample(1).values[0]
    return response
```

-----

## Project Structure (Inferred)

```
.
├── nexyug_chatbot_dataset_3000.xlsx  (Your dataset file)
├── chatbot_deva_notebook.ipynb     (The Jupyter/Colab notebook with the code)
├── intent_model.pkl                (Saved Logistic Regression model)
└── vectorizer.pkl                  (Saved TF-IDF Vectorizer)
```
