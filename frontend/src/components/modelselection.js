// modelselection.js
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

function ModelSelection() {
  const [model, setModel] = useState("random_forest");
  const [testSize, setTestSize] = useState(0.2);
  const [batchSize, setBatchSize] = useState(32);
  const [epochs, setEpochs] = useState(10);
  const [earlyStopping, setEarlyStopping] = useState(false);
  const [targetColumn, setTargetColumn] = useState("target");
  const [cleanedFile, setCleanedFile] = useState(null); // New state for cleaned file
  const [notification, setNotification] = useState("");
  const [trainingResults, setTrainingResults] = useState(null);
  const navigate = useNavigate();

  // Handle file selection
  const handleFileChange = (e) => {
    setCleanedFile(e.target.files[0]); // Store the selected file
  };

  const handleTrainModel = async () => {
    if (!cleanedFile) {
      setNotification("Please upload a cleaned file before training.");
      return;
    }

    // Update notification and clear any previous training results
    setNotification("Training model, please wait...");
    setTrainingResults(null);

    const formData = new FormData();
    formData.append("model", model);
    formData.append("test_size", testSize);
    formData.append("batch_size", batchSize);
    formData.append("epochs", epochs);
    formData.append("early_stopping", earlyStopping);
    formData.append("target_column", targetColumn);
    formData.append("file", cleanedFile); // Add cleaned file to the form data

    const response = await fetch("http://localhost:5000/train-model", {
      method: "POST",
      body: formData, // Send as form data
    });

    if (response.ok) {
      const data = await response.json();
      // alert(`Model trained successfully! Results: ${JSON.stringify(data.result)}`);
      setNotification("Model trained successfully!");
      console.log("Training results: ", data.result);
      setTrainingResults(data.result); // Save the training results to state
      // navigate('/evaluate');
    } else {
      const errorData = await response.json();
      // alert(`Error: ${errorData.message}`);
      setNotification(`Error: ${errorData.message}`);
    }
  };

  return (
    <div>
      <h1>Model Selection</h1>
      <form>
        <label>
          Select Model:
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="random_forest">Random Forest</option>
            <option value="linear_regression">Linear Regression</option>
            <option value="logistic_regression">Logistic Regression</option>
            <option value="lstm">LSTM</option>
            <option value="svm">SVM</option>
          </select>
        </label>
        <br />
        <label>
          Test Size:
          <input
            type="number"
            value={testSize}
            onChange={(e) => setTestSize(e.target.value)}
            step="0.01"
            min="0"
            max="1"
          />
        </label>
        <br />
        <label>
          Batch Size:
          <input
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(e.target.value)}
            min="1"
          />
        </label>
        <br />
        <label>
          Epochs:
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(e.target.value)}
            min="1"
          />
        </label>
        <br />
        <label>
          Early Stopping:
          <input
            type="checkbox"
            checked={earlyStopping}
            onChange={(e) => setEarlyStopping(e.target.checked)}
          />
        </label>
        <br />
        <label>
          Target Column:
          <input
            type="text"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
          />
        </label>
        <br />
        <label>
          Upload Cleaned File:
          <input type="file" onChange={handleFileChange} accept=".csv" />{" "}
          {/* File input for cleaned file */}
        </label>
        <br />
      </form>
      <button type="button" onClick={handleTrainModel}>
        Train Model
      </button>
      {notification && <div className="notification">{notification}</div>}
      {/* Display training results once available */}
      {trainingResults && (
        <div className="training-results">
          <h2>Training Results</h2>
          <ul>
            {trainingResults.Results.model !== undefined && (
              <li>Model: {trainingResults.Results.model}</li>
            )}
            {trainingResults.Results.loss !== undefined && (
              <li>Loss: {trainingResults.Results.loss}</li>
            )}
            {trainingResults.Results.accuracy !== undefined && (
              <li>Accuracy: {trainingResults.Results.accuracy}</li>
            )}
            {trainingResults.Results.train_mae !== undefined && (
              <li>Train MAE: {trainingResults.Results.train_mae}</li>
            )}
            {trainingResults.Results.train_rmse !== undefined && (
              <li>Train RMSE: {trainingResults.Results.train_rmse}</li>
            )}
            {trainingResults.Results.test_mae !== undefined && (
              <li>Test MAE: {trainingResults.Results.test_mae}</li>
            )}
            {trainingResults.Results.test_rmse !== undefined && (
              <li>Test RMSE: {trainingResults.Results.test_rmse}</li>
            )}
            {trainingResults.Results.mse !== undefined && (
              <li>MSE: {trainingResults.Results.mse}</li>
            )}
            {trainingResults.Results.predictions !== undefined && (
              <li>
                Predictions:{" "}
                {Array.isArray(trainingResults.Results.predictions)
                  ? `${trainingResults.Results.predictions.length} predictions`
                  : trainingResults.Results.predictions}
              </li>
            )}
          </ul>
        </div>
      )}
      <div className="footer">
        <div>
          <a href="/upload-data">
            <button>
              <svg
                class="w-6 h-6 text-gray-800 dark:text-white"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                fill="none"
                viewBox="0 0 24 24"
              >
                <path
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="m15 19-7-7 7-7"
                />
              </svg>
              Data Cleaning
            </button>
          </a>
        </div>
        <div>
          <a href="/evaluate">
            <button>
              Model Evaluation
              <svg
                class="w-6 h-6 text-gray-800 dark:text-white"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                fill="none"
                viewBox="0 0 24 24"
              >
                <path
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="m9 5 7 7-7 7"
                />
              </svg>
            </button>
          </a>
        </div>
      </div>
    </div>
  );
}
export default ModelSelection;
