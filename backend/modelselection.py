# modelselection.py

import numpy as np
import pandas as pd
import logging
from datacleaning import clean_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR 
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import joblib
import os
import json
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import json

# Initialize logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# A dictionary to map model names to their respective functions
MODELS = {
    'random_forest': 'train_random_forest',
    'linear_regression': 'train_linear_regression',
    'logistic_regression': 'train_logistic_regression',
    'lstm': 'train_lstm',
    'svm': 'train_svm'
}

def train_random_forest(params, target_column, is_classification=False):
    logging.debug("Training Random Forest with params: %s", params)
    
    logging.info(f"Params: {params}, Target column: {target_column}")
    
    # Capture the output of clean_data function
    cleaned_filepath, scaled_df = clean_data('./static/uploads/cleaned_data.csv')
    
    # Log the outputs for verification
    logging.debug(f"cleaned_filepath: {cleaned_filepath}")
    logging.debug(f"scaled_df type: {type(scaled_df)}")
    logging.debug(f"scaled_df sample (first few rows):\n{scaled_df.head()}")  # Log first few rows for verification
    
    # Identify categorical columns
    categorical_cols = scaled_df.select_dtypes(include=['object']).columns.tolist()
    logging.info(f"Identified categorical columns: {categorical_cols}")
    
    # Convert categorical variables using One-Hot Encoding
    if categorical_cols:
        df_encoded = pd.get_dummies(scaled_df, columns=categorical_cols, drop_first=True)
        logging.info(f"One-Hot Encoding applied. New shape of the dataset: {df_encoded.shape}")
    else:
        df_encoded = scaled_df
        logging.info("No categorical columns found. Proceeding with the original dataset.")
    
    # Now separate features and target
    if target_column not in df_encoded.columns:
        logging.error(f"Target column '{target_column}' not found in the dataset.")
        raise ValueError(f"Target column '{target_column}' is missing.")

    X = df_encoded.drop(target_column, axis=1)  # Features
    y = df_encoded[target_column]  # Target
    
    # Log the shapes of X and y
    logging.debug(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Split data into training and testing sets (if not already done)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Log the split
    logging.info(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

    # Fit the Random Forest model
    n_estimators = params.get('n_estimators', 100)
    if is_classification:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        logging.info("Fitting the Random Forest Classifier model...")
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        logging.info("Fitting the Random Forest Regressor model...")
    
    model.fit(X_train, y_train)
    logging.info("Model fitting completed.")

    # Make predictions
    predictions = model.predict(X_test)
    logging.debug(f"Predictions made on test set: {predictions[:5]}")  # Log first few predictions

    # Calculate the Mean Squared Error or Accuracy
    if is_classification:
        accuracy = accuracy_score(y_test, predictions)
        logging.info(f"Random Forest Classifier model trained with accuracy: {accuracy}")
        results = {
            "model": "Random Forest Classifier",
            "accuracy": accuracy,
            "predictions": predictions.tolist(),
            "actual_values": y_test.tolist(),
            "feature_importance": dict(zip(X.columns, model.feature_importances_))
        }
    else:
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Random Forest Regressor model trained with MSE: {mse}")
        results = {
            "model": "Random Forest Regressor",
            "mse": mse,
            "predictions": predictions.tolist(),
            "actual_values": y_test.tolist(),
            "feature_importance": dict(zip(X.columns, model.feature_importances_))
        }
    
    save_evaluation_results('random_forest', results)
    
    return {"Results": results}



def train_linear_regression(params, target_column):
    logging.debug("Training Linear Regression with params: %s", params)

    logging.info(f"Params: {params}, Target column: {target_column}")

    # Capture the output of clean_data function
    cleaned_filepath, scaled_df = clean_data('./static/uploads/cleaned_data.csv', target_column)

    # Log the outputs for verification
    logging.debug(f"cleaned_filepath: {cleaned_filepath}")
    logging.debug(f"scaled_df type: {type(scaled_df)}")
    logging.debug(f"scaled_data sample (first few rows): {scaled_df[:5]}")

    # Check if the target column exists in the dataframe
    if target_column not in scaled_df.columns:
        logging.error(f"Target column '{target_column}' not found in the dataset.")
        raise ValueError(f"Target column '{target_column}' is missing.")

    # Separate features and target
    X = scaled_df.drop(target_column, axis=1)  # Features
    y = scaled_df[target_column]  # Target

    # Log the shapes of X and y
    logging.debug(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Log target distribution
    logging.info(f"Target column distribution: {y.value_counts()}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.get('test_size', 0.2), random_state=42)

    # Log the split
    logging.info(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

    # Create and fit the Linear Regression model with progress
    model = LinearRegression()
    logging.info("Fitting the Linear Regression model with progress...")

    # Use tqdm to show progress during model fitting
    # for _ in tqdm(range(1), desc="Training Progress"):
    #     model.fit(X_train, y_train)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    logging.debug(f"Predictions made on test set: {predictions[:5]}")  # Log first few predictions

    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    logging.info(f"Linear Regression model trained with MSE: {mse}")
    
    results = {
        "model": "Linear Regression",
        "mse": mse,
        "predictions": predictions.tolist(),
        "actual_values": y_test.tolist(),
        "feature_importance": dict(zip(X.columns, model.coef_))
    }
    save_evaluation_results('linear_regression', results)

    # return {"model": model, "mse": mse}
    return {"Results": results}



def train_logistic_regression(params, target_column):
    logging.debug("Training Logistic Regression with params: %s", params)
    
    logging.info(f"Params: {params}, Target column: {target_column}")
    
    try:
        # Capture the output of clean_data function
        cleaned_filepath, scaled_df = clean_data('./static/uploads/cleaned_data.csv', target_column)
        
        # Log the outputs for verification
        logging.debug(f"cleaned_filepath: {cleaned_filepath}")
        
        # Type check for scaled_df
        if not isinstance(scaled_df, pd.DataFrame):
            logging.error(f"Expected scaled_df to be a DataFrame, but got {type(scaled_df)}")
            raise TypeError(f"scaled_df is of type {type(scaled_df)}, expected pandas DataFrame.")
        
        logging.debug(f"scaled_df type: {type(scaled_df)}")
        logging.debug(f"scaled_data sample (first few rows):\n{scaled_df.head()}")  # Log first few rows for verification

        # Separate the target column before encoding
        if target_column not in scaled_df.columns:
            logging.error(f"Target column '{target_column}' not found in the dataset.")
            raise ValueError(f"Target column '{target_column}' is missing.")
        
        target = scaled_df[target_column]
        scaled_df = scaled_df.drop(target_column, axis=1)

        # **Ensure the target is categorical**
        if pd.api.types.is_numeric_dtype(target):
            logging.info("Target column is continuous. Converting it to categorical by binning.")
            # Example: Bin the target into two classes using median
            target = pd.cut(target, bins=2, labels=[0, 1])
            logging.info("Target column successfully converted to categorical.")
        else:
            logging.info("Target column is already categorical.")

        # Identify categorical columns
        categorical_cols = scaled_df.select_dtypes(include=['object']).columns.tolist()
        logging.info(f"Identified categorical columns: {categorical_cols}")
        
        # Convert categorical variables using One-Hot Encoding
        if categorical_cols:
            df_encoded = pd.get_dummies(scaled_df, columns=categorical_cols, drop_first=True)
            logging.info(f"One-Hot Encoding applied. New shape of the dataset: {df_encoded.shape}")
        else:
            df_encoded = scaled_df
            logging.info("No categorical columns found. Proceeding with the original dataset.")
        
        # Re-add the target column to the encoded dataframe
        df_encoded[target_column] = target
        logging.debug(f"Re-added target column '{target_column}'. New shape of dataset: {df_encoded.shape}")
        
        # Separate features and target
        X = df_encoded.drop(target_column, axis=1)  # Features
        y = df_encoded[target_column]  # Target
        
        # Log the shapes of X and y
        logging.debug(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Log the split
        logging.info(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

        # Create and fit the Logistic Regression model with verbose output using `tqdm`
        max_iter = params.get('max_iter', 100)
        model = LogisticRegression(max_iter=max_iter, warm_start=True, solver='lbfgs')

        logging.info("Fitting the Logistic Regression model...")
        
        # Use tqdm to track progress over multiple iterations
        # for i in tqdm(range(1, max_iter + 1), desc="Training Progress"):
            # model.set_params(max_iter=i)
            # model.fit(X_train, y_train)
        model.fit(X_train, y_train)
        logging.info("Model fitting completed.")

        # Make predictions
        predictions = model.predict(X_test)
        logging.debug(f"Predictions made on test set: {predictions[:5]}")  # Log first few predictions

        # Calculate the accuracy score
        accuracy = accuracy_score(y_test, predictions)
        logging.info(f"Logistic Regression model trained with accuracy: {accuracy}")
        
        # Calculate additional metrics
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        results = {
            "model": "Logistic Regression",
            "accuracy": accuracy,
            "predictions": predictions.tolist(),
            "actual_values": y_test.tolist(),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "feature_importance": dict(zip(X.columns, model.coef_[0]))
        }
        
        save_evaluation_results('logistic_regression', results)

        return {"Results": results, }

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        return {"error": str(e)}



def train_svm(params, target_column):
    logging.debug("Training SVM with params: %s", params)
    logging.info(f"Params: {params}, Target column: {target_column}")
    
    cleaned_filepath, scaled_df = clean_data('./static/uploads/cleaned_data.csv', target_column)
    logging.debug(f"Cleaned filepath: {cleaned_filepath}")
    
    if target_column not in scaled_df.columns:
        logging.error(f"Target column '{target_column}' not found.")
        raise ValueError(f"Target column '{target_column}' is missing.")
    
    # Extract and prepare target and features
    target = scaled_df[target_column]
    scaled_df = scaled_df.drop(target_column, axis=1)
    
    # If the target is continuous, convert it to binary classification
    if pd.api.types.is_numeric_dtype(target):
        logging.info("Converting continuous target to categorical for SVM.")
        target = pd.cut(target, bins=2, labels=[0, 1])
    
    # One-Hot Encode categorical features if present
    categorical_cols = scaled_df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df_encoded = pd.get_dummies(scaled_df, columns=categorical_cols, drop_first=True)
    else:
        df_encoded = scaled_df.copy()
    
    df_encoded[target_column] = target
    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Create and train SVM model (using SVC for classification)
    C = params.get('C', 1.0)
    kernel = params.get('kernel', 'rbf')
    model = SVC(C=C, kernel=kernel, probability=True)
    logging.info("Fitting the SVM model...")
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logging.info(f"SVM model accuracy: {accuracy}")
    
    results = {
        "model": "SVM",
        "accuracy": accuracy,
        "predictions": predictions.tolist(),
        "actual_values": y_test.tolist(),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True)
    }
    save_evaluation_results('svm', results)
    return {"Results": results}



def train_lstm(params, target_column):
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    logging.debug("Training LSTM with params: %s", params)

    # Capture the output of clean_data function
    cleaned_filepath, scaled_df = clean_data('./static/uploads/cleaned_data.csv', target_column)

    # Log the outputs for verification
    logging.debug(f"cleaned_filepath: {cleaned_filepath}")
    logging.debug(f"scaled_data type: {type(scaled_df)}")
    logging.debug(f"scaled_data sample (first few rows): {scaled_df.head()}")  # Assuming scaled_df is a DataFrame

    # # Exclude non-numeric columns (e.g., 'Date') from the features
    numeric_columns = scaled_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column not in numeric_columns:
        raise ValueError(f"Target column '{target_column}' is not numeric.")

    features = [col for col in numeric_columns if col != target_column]

    # # Check if scaled_df is a valid DataFrame
    if isinstance(scaled_df, pd.DataFrame):
        training_data_len = int(np.ceil(len(scaled_df) * 0.8))
        train_data = scaled_df[0:int(training_data_len)]
        test_data = scaled_df[training_data_len:]
    else:
        logging.error("scaled_df is not a valid format (expected DataFrame)")
        raise ValueError("Invalid data format in scaled_df")

    # # Create the dataset for features and labels (assuming time series)
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[features].iloc[i-60:i].values)  # Use only numeric columns for input
        y_train.append(train_data.iloc[i][target_column])  # The target variable

    for i in range(60, len(test_data)):
        x_test.append(test_data[features].iloc[i-60:i].values)
        y_test.append(test_data.iloc[i][target_column])
    
    # # Convert to NumPy arrays and ensure the type is float32
    x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32)
    x_test, y_test = np.array(x_test).astype(np.float32), np.array(y_test).astype(np.float32)

    if np.isnan(x_train).any() or np.isnan(y_train).any():
        logging.error("Training data contains NaN values. Please clean the data.")
        raise ValueError("NaN values present in the data.")

    # # Reshape x_train to be 3D (samples, time steps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    logging.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    if len(x_train.shape) != 3:
        logging.error("x_train is not 3D. LSTM expects 3D input (samples, time steps, features).")
        raise ValueError("Incorrect shape for LSTM input.")

    # # Create LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # # Early stopping based on user input
    callbacks = []
    if params.get('early_stopping', False):
        callbacks.append(EarlyStopping(monitor='val_loss', patience=2))

    logging.info("Starting model training with progress...")

    tf.autograph.set_verbosity(0)
    
    # # Train the model with progress tracking using tqdm
    # with tqdm(total=params.get('epochs', 10), desc="Training LSTM") as pbar:
    #     for epoch in range(params.get('epochs', 10)):
    #         history = model.fit(x_train, y_train, batch_size=params.get('batch_size', 32), epochs=1, callbacks=callbacks, verbose=0)
    #         pbar.update(1)  # Update progress bar by 1 after each epoch

    history = model.fit(x_train, y_train, batch_size=params.get('batch_size', 32), epochs=params.get('epochs', 10), callbacks=callbacks, verbose=0)

    model.summary()
    
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    logging.info("LSTM model trained successfully")

    # # Inverse transform predictions and actual values if scaled
    
    scaler = MinMaxScaler()
    scaler.fit(scaled_df[target_column].values.reshape(-1, 1))

    train_predict = scaler.inverse_transform(train_predict).flatten().tolist()
    test_predict = scaler.inverse_transform(test_predict).flatten().tolist()
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten().tolist()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten().tolist()

    mse = mean_squared_error(y_test, test_predict)
    logging.info(f"Linear Regression model trained")
    
    # # Calculate training loss and accuracy if needed
    loss = history.history['loss'][-1]  # Get the final loss

    # # Log metrics
    # logging.info("Calculating metrics...")
    # train_mae = mean_absolute_error(y_train, train_predict)
    # train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
    # test_mae = mean_absolute_error(y_test, test_predict)
    # test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
    
    # print('Train Mean Absolute Error:', train_mae)
    # print('Train Root Mean Squared Error:', train_rmse)
    # print('Test Mean Absolute Error:', test_mae)
    # print('Test Root Mean Squared Error:', test_rmse)
    
    logging.info(f"Final training loss: {loss}")
    
    # results = {
    #    "model": "LSTM",
    #     "loss": float(history.history['loss'][-1])
    #     "train_mae": float(train_mae),  # Convert to Python float
    #     "train_rmse": float(train_rmse),  # Convert to Python float
    #     "test_mae": float(test_mae),  # Convert to Python float
    #     "test_rmse": float(test_rmse),
    #     # Note: LSTM might not have a straightforward accuracy metric, 
    #     # so you could consider a different evaluation method.
    # }
    # save_evaluation_results('lstm', results)
    
    # return {
    #  # Return the model if needed, but don't include in the JSON response
    #     "Results": results,    # Return loss or other useful metrics
    # }  
    
    results = {
        "model": "LSTM",
        "loss": float(loss),
        "mse":float(mse),
        "train_mae": float(mean_absolute_error(y_train, train_predict)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_predict))),
        "test_mae": float(mean_absolute_error(y_test, test_predict)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_predict))),
        "predictions": test_predict,
        "actual_values": y_test
    }
    save_evaluation_results('lstm', results)
    
    return {
        "Results": results
    }  
    
    
    
    
    
    # if target_column not in scaled_df.columns:
    #     raise ValueError(f"Target column '{target_column}' is not numeric.")

    # features = [col for col in scaled_df.columns if col != target_column]

    # training_data_len = int(np.ceil(len(scaled_df) * 0.8))
    # train_data = scaled_df[:training_data_len]
    # test_data = scaled_df[training_data_len:]

    # x_train, y_train, x_test, y_test = [], [], [], []
    # for i in range(60, len(train_data)):
    #     x_train.append(train_data[features].iloc[i-60:i].values)
    #     y_train.append(train_data.iloc[i][target_column])

    # for i in range(60, len(test_data)):
    #     x_test.append(test_data[features].iloc[i-60:i].values)
    #     y_test.append(test_data.iloc[i][target_column])
    
    # x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32)
    # x_test, y_test = np.array(x_test).astype(np.float32), np.array(y_test).astype(np.float32)

    # if np.isnan(x_train).any() or np.isnan(y_train).any():
    #     logging.error("Training data contains NaN values. Please clean the data.")
    #     raise ValueError("NaN values present in the data.")

    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # logging.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    # logging.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # if len(x_train.shape) != 3:
    #     logging.error("x_train is not 3D. LSTM expects 3D input (samples, time steps, features).")
    #     raise ValueError("Incorrect shape for LSTM input.")

    # model = Sequential()
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    # model.add(LSTM(units=50, return_sequences=False))
    # model.add(Dense(units=25))
    # model.add(Dense(units=1))

    # model.compile(optimizer='adam', loss='mean_squared_error')

    # callbacks = []
    # if params.get('early_stopping', False):
    #     callbacks.append(EarlyStopping(monitor='val_loss', patience=2))

    # logging.info("Starting model training...")

    # history = model.fit(x_train, y_train, batch_size=params.get('batch_size', 32), epochs=params.get('epochs', 10), callbacks=callbacks, verbose=0)

    # model.summary()
    
    # train_predict = model.predict(x_train)
    # test_predict = model.predict(x_test)

    # logging.info("LSTM model trained successfully")

    # scaler = MinMaxScaler()
    # scaler.fit(scaled_df[target_column].values.reshape(-1, 1))

    # train_predict = scaler.inverse_transform(train_predict)
    # test_predict = scaler.inverse_transform(test_predict)
    # y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # loss = history.history['loss'][-1]  # Get the final loss

    # logging.info(f"Final training loss: {loss}")
    
    # results = {
    #     "model": "LSTM",
    #     "loss": float(loss),
    #     "train_mae": float(mean_absolute_error(y_train, train_predict)),
    #     "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_predict))),
    #     "test_mae": float(mean_absolute_error(y_test, test_predict)),
    #     "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_predict))),
    #     "predictions": test_predict.tolist(),
    #     "actual_values": y_test.tolist()
    # }
    # save_evaluation_results('lstm', results)
    
    # return {
    #     "Results": results
    # }  
    
    
    
def save_evaluation_results(model_name, results):
    """ Save model evaluation results to a JSON file. """
    results_dir = './static/results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{model_name}_evaluation.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f)
    logging.info(f"Evaluation results saved at {results_path}")


# Function to handle the training based on user input
def train_model(model_name, params, target_column):
    try:
        logging.info(f"Training model: {model_name} with parameters: {params}")

        if model_name == 'random_forest':
            return train_random_forest( params, target_column)
        elif model_name == 'linear_regression':
            return train_linear_regression(params, target_column)
        elif model_name == 'logistic_regression':
            return train_logistic_regression(params, target_column)
        elif model_name == 'lstm':
            return train_lstm(params, target_column)
        elif model_name == 'svm':
            return train_svm(params, target_column)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise    
    
def load_model(model_path):
    return joblib.load(model_path)
