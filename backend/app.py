# app.py

from flask import Flask, request, jsonify, render_template, json, send_from_directory
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from datacleaning import clean_data
from modelselection import train_model
from modelevaluation import generate_plots
from api.routes import api
import os
import pandas as pd
import logging
from gan_augment import generate_synthetic_data 

logging.basicConfig(format='%(message)s')

app = Flask(__name__)
app.register_blueprint(api, url_prefix='/api')
CORS(app)

# Define the directory where uploaded files will be saved
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to validate if the file is a CSV
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

@app.route('/upload-data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("No file selected")
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            print(f"File saved successfully at {filepath}")
            return jsonify({"message": "File uploaded successfully", "filepath": filepath}), 200
        except Exception as e:
            print(f"Error saving the file: {str(e)}")
            return jsonify({"error": "Failed to save the file"}), 500
    else:
        print("Invalid file format. Only .csv files are allowed.")
        return jsonify({"error": "Invalid file format. Only .csv files are allowed."}), 400

@app.route('/augment-data', methods=['POST'])
def augment_data_route():
    try:
        data_path = request.json.get('filepath')
        
        if not data_path:
            return jsonify({"error": "No file provided"}), 400

        augmented_filepath, synthetic_df = generate_synthetic_data(data_path)

        if augmented_filepath is None:
            return jsonify({"error": synthetic_df}), 500
        
        return jsonify({"message": "Data augmented successfully", "augmented_filepath": augmented_filepath}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to clean the data
@app.route('/clean-data', methods=['POST'])
def clean_data_route():
    data_path = request.json.get('filepath')
    print("datapath:", data_path)
    
    # target_column = request.json.get('target_column', 'RainTomorrow')  # Use request.json instead of form for JSON input
    # print("target_column:", target_column)
    
    if not data_path:
        return jsonify({"error": "Filepath not provided"}), 400

    cleaned_filepath, scaled_data = clean_data(data_path)

    if cleaned_filepath is None:
        return jsonify({"error": scaled_data}), 500

    return jsonify({"message": "Data cleaned successfully", "cleaned_filepath": cleaned_filepath}), 200

# Route to train a model
@app.route('/train-model', methods=['POST'])
def train_model_route():
    try:
        logging.info("Received request to train a model")
        
        # Handle the uploaded cleaned file
        file = request.files['file']
        if file.filename == '':
            raise ValueError("No file selected")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        logging.info(f"Cleaned file uploaded successfully: {filepath}")

        # Read the CSV file
        df = pd.read_csv(filepath)
        logging.debug(f"CSV file content: {df.head()}")

        model_name = request.form.get('model')
        if model_name:
            model_name = model_name.strip().lower()
        else:
            raise ValueError("No model specified")
        
        # Get other parameters from the request
        model_name = request.form.get('model')
        test_size = float(request.form.get('test_size', 0.2))
        batch_size = int(request.form.get('batch_size', 32))
        epochs = int(request.form.get('epochs', 10))
        early_stopping = request.form.get('early_stopping') == 'true'
        target_column = request.form.get('target_column', 'RainTomorrow')  # Ensure the target column is retrieved correctly

        logging.info(f"Model: {model_name}, Test size: {test_size}, Batch size: {batch_size}, Epochs: {epochs}, Early stopping: {early_stopping}, Target column: {target_column}")

        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in the dataset")

        # Prepare features and target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        # Parameters to pass to the train_model function
        params = {
            'n_estimators': 200, 
            'batch_size': batch_size,
            'epochs': epochs,
            'early_stopping': early_stopping,
        }

        logging.debug(f"Params before calling train_model: {params}")

        # Call the model training function
        result = train_model(model_name, params, target_column=target_column)

        # Extract only serializable information
        # response = {
        #     "loss": result['Results'].get('loss'),
        #     "accuracy": result.get("accuracy"),  # Assuming train_model returns accuracy
        #     "train_mae": result['Results'].get('train_mae'),
        #     "train_rmse": result['Results'].get('train_rmse'),
        #     "test_mae": result['Results'].get('test_mae'),
        #     "test_rmse": result['Results'].get('test_rmse'),
        #     "message": "Model trained successfully"
        # }
    
        return jsonify({"status": "success", "result": result}), 200

    except Exception as e:
        logging.error(f"Error in /train-model route: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500    

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('results', filename)

@app.route('/upload-results', methods=['POST'])
def upload_results():
    try:
        model_name = request.form.get('model_name')
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save the file
        filepath = os.path.join('./static/results', f"{model_name}_evaluation.json")
        file.save(filepath)

        # Plot the evaluation results
        plot_model_evaluation(model_name)

        return jsonify({"message": f"Results uploaded and plots generated for {model_name}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Flask route to handle evaluation and plotting
@app.route('/evaluate', methods=['POST'])
def evaluate_model_route():
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join('./static/results', file.filename)
    file.save(filepath)

    # return jsonify({"status": "success", "message": result}), 200
    try:
        evaluation_results = generate_plots(file.filename)
        return jsonify(evaluation_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
