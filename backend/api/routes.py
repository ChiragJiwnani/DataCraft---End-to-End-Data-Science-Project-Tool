# backend/api/routes.py
from flask import Blueprint, jsonify, request
# from modelevaluation import plot_model_evaluation  # Import the plot function
from modelevaluation import generate_plots  # Import the plot function


api = Blueprint('api', __name__)

@api.route('/upload-data', methods=['POST'])
def upload_data():
    # Logic for handling data upload
    return jsonify({"message": "Data uploaded successfully"}), 200

@api.route('/data-cleaning', methods=['POST'])
def data_cleaning():
    # Logic for handling data cleaning
    return jsonify({"message": "Data cleaned successfully"}), 200

@api.route('/train-model', methods=['POST'])
def train_model():
    # Logic for handling model selection
    return jsonify({"message": "Model selected successfully"}), 200

@api.route('/evaluate', methods=['POST'])
def evaluate(model_name):
    """ Endpoint to evaluate the specified model and generate plots. """
    try:
        # Call the evaluation function and generate plots
        message = generate_plots(model_name)
        return jsonify({"status": "success", "message": message}), 200
    except FileNotFoundError:
        return jsonify({"status": "error", "message": f"Model evaluation results not found for {model_name}."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
