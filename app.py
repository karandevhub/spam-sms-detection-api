from flask import Flask, request, jsonify
import pickle

import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps =PorterStemmer()

model = pickle.load(open('model.pkl','rb'))     
tfidf = pickle.load(open('vectorizer.pkl','rb'))

def Transform_text(text):
    text=text.lower()     
    text=nltk.word_tokenize(text)  
    y=[]                
    for i in text:       
        if i.isalnum():   
            y.append(i)
    text=y[:]             
    y.clear()            

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation: 
            y.append(i)                                                           
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))   
    return " ".join(y)     



app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get the input message from the request
        input_msg = request.json.get('message')
        # Transform the input message 
        transformed_text=Transform_text(input_msg)
       
        # vectorize the input message
        vector_input= tfidf.transform([transformed_text])
        #predict the input message
        prediction = int(model.predict(vector_input)[0])

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        # Handle any errors that may occur during the prediction
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
