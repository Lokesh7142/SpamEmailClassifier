# Install required packages first (if not already installed)
# pip install pandas scikit-learn nltk

import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", names=["label", "message"])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocessing
stemmer = PorterStemmer()
default_stopwords = set(stopwords.words("english"))

# Retain key spam-related words
important_words = {"you", "your", "won", "click", "free", "now", "claim", "gift", "link"}
custom_stopwords = default_stopwords - important_words

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in custom_stopwords]
    return ' '.join(words)

df['cleaned_message'] = df['message'].apply(preprocess_text)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation (optional)
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function
def classify_message(message):
    cleaned = preprocess_text(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# ----------- USER INPUT SECTION ------------
while True:
    user_message = input("\nEnter a message (or type 'exit' to quit): ")
    if user_message.lower() == 'exit':
        break
    print("Prediction:", classify_message(user_message))
