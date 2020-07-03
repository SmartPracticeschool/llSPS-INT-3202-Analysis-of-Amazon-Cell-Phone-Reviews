from flask import Flask,render_template,request,jsonify
from tensorflow.keras.models import load_model
import numpy as np
import random
import re
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle




app = Flask(__name__,template_folder = r'D:\ML\Analysis of Amazon Cell Phone Reviews\server\template')
model = load_model(r'D:\ML\Analysis of Amazon Cell Phone Reviews\model.h5')
cv = pickle.load(open(r'D:\ML\Analysis of Amazon Cell Phone Reviews\cv_vac.pickle','rb'))
def predictor(word):
    words = []
    for i in word:
        X = i
        ps = PorterStemmer()
        X = re.sub('[^a-zA-Z]',' ',X)
        X = X.lower()
        X = X.split()
        X = [ps.stem(w) for w in X if not w in stopwords.words('english')]
        X = ' '.join(X)
        words.append(X)
    X_in = cv.transform(word)
    pred = np.array(model.predict(X_in))
    if pred>0.5:
        my_pred = 'Positive review'
    else:
        my_pred = 'negetive review '
    return my_pred,pred

@app.route('/',methods=['POST'])
def pred():
    text = request.get_jason()['text']
    print(text)
    prediction = predictor(text)
    return jsonify({'sentiment':prediction[0],'confidance':prediction[1]})


if __name__ == '__main__':
    app.run(debug=False)
    
