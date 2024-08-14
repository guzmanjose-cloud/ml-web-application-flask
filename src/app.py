from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('/workspaces/ml-web-application-flask/src/model.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    features = [float(x) for x in request.form.values()]
    features_array = np.array([features])
    
    # Make a prediction
    prediction = model.predict(features_array)
    
    # Determine the output
    output = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    
    return render_template('index.html', prediction_text=f'The person is likely {output}')

if __name__ == '__main__':
    app.run(debug=True)
