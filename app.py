from flask import Flask, render_template, request
import joblib
import emoji
import numpy as np
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define your preprocessing functions here
def basic_pp(x, emoj="f"):
    emoj = emoj.lower()  
    if emoj == "t":
        x = emoji.demojize(x)
    x = re.sub("[\]\.\*'\-&!$%^,;?(0-9)_:#]", ' ', x)
    x = x.lower()
    x = re.sub('<.*?>',' ', x)
    x = re.sub('http[s]?://.+?\S+', ' ', x)
    x = re.sub('#\S+', ' ', x)
    x = re.sub('@\S+', ' ', x)
    return x
stp=stopwords.words('english')
stp.remove('not')
def stop_words(x):
    sent=[]
    for word in word_tokenize(x):
        if word in stp:
            pass
        else:
            sent.append(word)
    return ' '.join(sent)

def stem(x):
        sent=[]
        wl=WordNetLemmatizer()
        for word in word_tokenize(x):
            sent.append(wl.lemmatize(word,pos='n'))
        return " ".join(sent)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        # Get input review text from the form
        review_text = request.form.get('inputreview')
        
        # Perform preprocessing steps
        preprocessed_review = basic_pp(review_text, emoj='t')
        preprocessed_review = stop_words(preprocessed_review)
        preprocessed_review = stem(preprocessed_review)
        
        # Load the trained logistic regression model
        model = joblib.load("model/model1.pkl")
        
        # Vectorize the preprocessed review text
        # Assuming you have already fitted CountVectorizer on your training data
        # You should use the same CountVectorizer instance for consistency
        vectorizer = joblib.load("model/vocab.pkl")
        review_vector = vectorizer.transform([preprocessed_review])
        
        # Make prediction using the model
        prediction = model.predict(review_vector)
        
        return render_template('output.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)
