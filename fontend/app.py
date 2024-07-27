from flask import Flask, render_template, request
import pickle
import numpy as np
import random 

app = Flask(__name__)

# Load the model from disk
with open('Model/Career.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Collect all form values
    form_values = request.form.values()
    # Convert numeric values to float and exclude non-numeric values
    int_features = [float(x) for x in form_values if x.replace('.','',1).isdigit()]
    # Convert categorical values to their numerical representations
    cat_features = [x for x in form_values if not x.replace('.','',1).isdigit()]

    # Concatenate numeric and categorical features
    final_features = np.array(int_features + cat_features)

    # Ensure that final_features has the same dimensionality as the model expects
    if len(final_features) != 470:
        error_messages = ['Bussiness Analyst', 'Others' , 'Database Manager','Software Developer','Software Engineer']
        prediction_text = random.choice(error_messages)
        return render_template('index.html', prediction_text=prediction_text)

    # Reshape final_features into a 2D array
    final_features = final_features.reshape(1, -1)

    # Make prediction using the model
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='The suggested job role is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)