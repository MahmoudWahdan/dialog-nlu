# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu import AutoNLU
from flask import Flask, jsonify, request
import argparse


# Create app
app = Flask(__name__)

def initialize(load_folder_path):
    global nlu_model
    nlu_model = AutoNLU.load(load_folder_path)    

@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'hello from NLU service'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    input_json = request.json
    utterance = input_json["utterance"]
    print(utterance)
    response = nlu_model.predict(utterance)
    return jsonify(response)


if __name__ == '__main__':    
    # read command-line parameters
    parser = argparse.ArgumentParser('Running JointNLU model basic service')
    parser.add_argument('--model', '-m', help = 'Path to joint NLU model', type=str, required=True)
    parser.add_argument('--port', '-p', help = 'port of the service', type=int, required=False, default=5000)
    
    args = parser.parse_args()
    load_folder_path = args.model
    port = args.port
    
    print(('Starting the Server'))
    initialize(load_folder_path)
    # Run app
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)