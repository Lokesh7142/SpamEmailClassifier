## Spam Email Classifier using Machine Learning

## Short Description :
- This is a machine learning project that uses NLP to classify whether a given email or SMS is spam or not.
- It uses a Naive Bayes Classifier trained on a labeled SMS dataset.

## Features :
- Preprocesses and cleans message text using NLTK
- Converts text to features using CountVectorizer
- Trains a Naive Bayers Classifier
- Allows real - time user input for prediction
- Outputs whether the input message is Spam or Not Spam

## Technologies Used :
- Python
- Scikit-learn
- NLTK
- Pandas

## How to Run :
- Clone this repository or download the ".py" file
- Open terminal and install the required packages
- Run the Script
- Enter messages in the terminal to check if they are spam

## Sample output :
- Enter a message(or type 'exit' to quit): You have won a free gift card!
- Prediction : Spam

## Dataset :
- Source: SMS Spam Dataset
- Format: TSV (Tab-separated)
- Columns: label, message
 
## Future Improvements :
- Save trained model using joblib
- Build a web interface using Flask or Streamlit
- Use TF-IDF instead of CountVectorizer
