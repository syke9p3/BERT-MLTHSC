from flask import Flask, json, jsonify, request
from flask_cors import CORS
from gradio_client import Client

import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/labels", methods=['GET'])
def get_labels():

    input_text = request.args.get('input', '')
    
    # Generate random probabilities for the labels
    labels = [
        {"name": "Age", "probability": f"{random.uniform(0, 100):.2f}%"},
        {"name": "Gender", "probability": f"{random.uniform(0, 100):.2f}%"},
        {"name": "Physical", "probability": f"{random.uniform(0, 100):.2f}%"},
        {"name": "Race", "probability": f"{random.uniform(0, 100):.2f}%"},
        {"name": "Religion", "probability": f"{random.uniform(0, 100):.2f}%"},
        {"name": "Others", "probability": f"{random.uniform(0, 100):.2f}%"}
    ]

    # Sort labels based on probability in descending order
    labels = sorted(labels, key=lambda x: float(x["probability"][:-1]), reverse=True)
    
    data = {
        "labels": labels,
        "text": input_text
    }
    
    return jsonify(data)


# Define your /predict route
@app.route("/predict", methods=['GET'])
def get_pos_tag():

    input_text = request.args.get('input', '')

    client = Client("http://127.0.0.1:7861/")
    
    
    try:
        result_path = client.predict(input_text, api_name="/predict")
        
        # Read the content of the file at the result_path
        with open(result_path, 'r') as file:
            result_json = json.load(file)
        
        return jsonify(result_json)
    except Exception as e:
        # Handle other exceptions
        return "An error occurred: " + str(e), 500

if __name__ == "__main__":
    app.run(debug=True)