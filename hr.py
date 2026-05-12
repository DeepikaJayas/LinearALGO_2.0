from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = [
        float(request.form['satisfaction']),
        float(request.form['evaluation']),
        int(request.form['projects']),
        int(request.form['hours']),
        int(request.form['years']),
        int(request.form['accident']),
        int(request.form['promotion']),
        int(request.form['department']),
        int(request.form['salary'])
    ]

    prediction = model.predict([features])

    result = "Employee Will Leave" if prediction[0] == 1 else "Employee Will Stay"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)