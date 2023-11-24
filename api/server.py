from flask import Flask, json, jsonify, request
from flask_cors import CORS
import classifier as MLTHSC
import random

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


@app.route("/random_labels", methods=['GET'])
def get_random_labels():

    input_text = request.args.get('input', '')

    labels = MLTHSC.get_predictions(input_text)

    sample_json = [
        {
            "labels": [
                {
                    "name": "Gender",
                    "probability": "95.15%"
                },
                {
                    "name": "Race",
                    "probability": "14.40%"
                },
                {
                    "name": "Physical",
                    "probability": "10.09%"
                },
                {
                    "name": "Age",
                    "probability": "8.11%"
                },
                {
                    "name": "Religion",
                    "probability": "7.61%"
                },
                {
                    "name": "Others",
                    "probability": "3.16%"
                }
            ],
            "text": "di na natauhan tong mga animal na bakla"
        }, 
        {
        "labels": [
            {
            "name": "Age",
            "probability": "90.39%"
            },
            {
            "name": "Physical",
            "probability": "21.30%"
            },
            {
            "name": "Others",
            "probability": "8.25%"
            },
            {
            "name": "Race",
            "probability": "6.87%"
            },
            {
            "name": "Gender",
            "probability": "6.16%"
            },
            {
            "name": "Religion",
            "probability": "4.03%"
            }
            ],
            "text": "Tanginang nga batang paslit na to ang babaho pota dikit pa ng dikit sakin"
            },
            {
            "labels": [
            {
            "name": "Others",
            "probability": "47.97%"
            },
            {
            "name": "Physical",
            "probability": "41.99%"
            },
            {
            "name": "Religion",
            "probability": "8.62%"
            },
            {
            "name": "Race",
            "probability": "8.55%"
            },
            {
            "name": "Gender",
            "probability": "2.61%"
            },
            {
            "name": "Age",
            "probability": "1.71%"
            }
            ],
            "text": "Bobo mo Daniella tanga mahulog ka sana"
            }

        ]

    rand = random.randint(0,3)  

    text = sample_json[rand]["text"]
    labels = sample_json[rand]["labels"]

    data = {
        "text": text,
        "labels": labels
    }

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)