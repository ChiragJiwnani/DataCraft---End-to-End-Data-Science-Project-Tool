import React, { useState } from "react";
import Plot from "react-plotly.js";
// import DisplayPlots from "./displayplots";

function ModelEvaluation() {
  const [evaluationData, setEvaluationData] = useState(null);
  const [error, setError] = useState(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/evaluate", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Error uploading file: " + response.statusText);
      }

      const data = await response.json();
      console.log("Evaluation Data:", data);
      if (data.error) {
        setError(data.error);
        setEvaluationData(null);
      } else {
        setEvaluationData(data);
        setError(null);
      }
    } catch (error) {
      console.error(error);
      setError(error.message);
    }
  };

  // Safe metric display function
  const renderMetric = (value, decimals = 2) => {
    if (value === undefined || value === null) return "N/A";
    return `${(value * 100).toFixed(decimals)}%`;
  };

  return (
    <div className="container">
      <h1>Evaluate Model</h1>
      <div className="inputbox">
        <input type="file" onChange={handleFileUpload} accept=".json" />
      </div>

      {error && <div className="error">{error}</div>}

      {evaluationData && (
        <div>
          <h3>Results for {evaluationData.model}</h3>

          {/* Metrics Summary */}
          {evaluationData.metrics && (
            <div className="metrics-summary">
              <h3>Model Performance</h3>
              <table>
                <tbody>
                  <tr>
                    <td>Accuracy:</td>
                    <td>{renderMetric(evaluationData.metrics.accuracy)}</td>
                  </tr>
                  <tr>
                    <td>MSE:</td>
                    <td>{renderMetric(evaluationData.metrics.mse)}</td>
                  </tr>
                  <tr>
                    <td>Precision:</td>
                    <td>{renderMetric(evaluationData.metrics.precision)}</td>
                  </tr>
                  <tr>
                    <td>Recall:</td>
                    <td>{renderMetric(evaluationData.metrics.recall)}</td>
                  </tr>
                  <tr>
                    <td>F1 Score:</td>
                    <td>{renderMetric(evaluationData.metrics.f1)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}

          {/* Plots */}
          <div className="plots-container">
            {evaluationData.plots?.actual_vs_predicted && (
              <div className="plot-card">
                <Plot
                  data={evaluationData.plots.actual_vs_predicted.data}
                  layout={evaluationData.plots.actual_vs_predicted.layout}
                  style={{ width: "100%", height: "500px" }}
                />
              </div>
            )}

            {evaluationData.plots?.confusion_matrix && (
              <div className="plot-card">
                <Plot
                  data={evaluationData.plots.confusion_matrix.data}
                  layout={evaluationData.plots.confusion_matrix.layout}
                  style={{ width: "100%", height: "500px" }}
                />
              </div>
            )}

            {evaluationData.plots?.feature_importance && (
              <div className="plot-card">
                <Plot
                  data={evaluationData.plots.feature_importance.data}
                  layout={evaluationData.plots.feature_importance.layout}
                  style={{ width: "100%", height: "600px" }}
                />
              </div>
            )}

            {evaluationData.plots?.evaluation_metrics_plot && (
              <div className="plot-card">
                <Plot
                  data={evaluationData.plots.evaluation_metrics_plot.data}
                  layout={evaluationData.plots.evaluation_metrics_plot.layout}
                  style={{ width: "100%", height: "400px" }}
                />
              </div>
            )}

            {evaluationData.plots?.mse_plot && (
              <div className="plot-card">
                <Plot
                  data={evaluationData.plots.mse_plot.data}
                  layout={evaluationData.plots.mse_plot.layout}
                  style={{ width: "100%", height: "400px" }}
                />
              </div>
            )}

            {evaluationData.plots?.shap_feature_importance && (
              <div className="plot-card">
                <Plot
                  data={evaluationData.plots.shap_feature_importance.data}
                  layout={evaluationData.plots.shap_feature_importance.layout}
                  style={{ width: "100%", height: "400px" }}
                />
              </div>
            )}

            {evaluationData.plots?.roc_curve && (
              <div className="plot-card">
                <Plot
                  data={evaluationData.plots.roc_curve.data}
                  layout={evaluationData.plots.roc_curve.layout}
                  style={{ width: "100%", height: "400px" }}
                />
              </div>
            )}

            {evaluationData.plots?.loss_plot && (
              <div className="plot-card">
                <Plot
                  data={evaluationData.plots.loss_plot.data}
                  layout={evaluationData.plots.loss_plot.layout}
                  style={{ width: "100%", height: "400px" }}
                />
              </div>
            )}

            {evaluationData.plots?.accuracy_plot && (
              <div className="plot-card">
                <Plot
                  data={evaluationData.plots.accuracy_plot.data}
                  layout={evaluationData.plots.accuracy_plot.layout}
                  style={{ width: "100%", height: "400px" }}
                />
              </div>
            )}

          </div>
        </div>
      )}
      <div className="footer">
        <div>
          <a href="/upload-data">
            <button>
              <svg
                className="w-6 h-6 text-gray-800 dark:text-white"
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                fill="none"
                viewBox="0 0 24 24"
              >
                <path
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="m17 16-4-4 4-4m-6 8-4-4 4-4"
                />
              </svg>
              Data Cleaning
            </button>
          </a>
        </div>
        <div>
          <a href="/train-model">
            <button>
              <svg
                className="w-6 h-6 text-gray-800 dark:text-white"
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                fill="none"
                viewBox="0 0 24 24"
              >
                <path
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="m15 19-7-7 7-7"
                />
              </svg>
              Model Selection
            </button>
          </a>
        </div>
      </div>
    </div>
  );
}

export default ModelEvaluation;
