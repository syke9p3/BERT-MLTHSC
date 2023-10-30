from flask import Flask, json, jsonify, request
from flask_cors import CORS
from gradio_client import Client
import inference as MLTHSC

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/labels", methods=['GET'])
def get_labels():

    # http://127.0.0.1:5000/labels?input=di%20na%20natauhan%20tong%20mga%20animal%20na%20bakla

    input_text = request.args.get('input', '')
    
    labels = MLTHSC.get_predictions(input_text)
    
    data = {
        "text": input_text,
        "labels": labels
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