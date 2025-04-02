"use strict";

document.addEventListener("DOMContentLoaded", function () {
  var models = ["RandomForest", "LinearRegression", "LogisticRegression", "LSTM"];
  var evaluationContainer = document.getElementById("evaluation-results"); // Fetch logs and evaluation results for each model

  models.forEach(function (model) {
    fetch("/evaluate-model?model=".concat(model)).then(function (response) {
      return response.json();
    }).then(function (data) {
      displayModelLogs(model, data);
      displayPlots(model);
    })["catch"](function (error) {
      return console.error("Error fetching evaluation data for ".concat(model, ":"), error);
    });
  });

  function displayModelLogs(modelName, logs) {
    var modelLogsSection = document.createElement("div");
    modelLogsSection.classList.add("model-logs");
    modelLogsSection.innerHTML = "<h3>".concat(modelName, " Evaluation</h3>"); // Append logs (e.g., MSE, Classification Report)

    if (logs.mse) {
      modelLogsSection.innerHTML += "<p><strong>MSE:</strong> ".concat(logs.mse, "</p>");
    }

    if (logs.classification_report) {
      var report = logs.classification_report;
      var reportHTML = "<pre>".concat(JSON.stringify(report, null, 2), "</pre>");
      modelLogsSection.innerHTML += "<p><strong>Classification Report:</strong></p>".concat(reportHTML);
    }

    evaluationContainer.appendChild(modelLogsSection);
  }

  function displayPlots(modelName) {
    var plotContainer = document.createElement("div");
    plotContainer.classList.add("model-plots"); // Append Confusion Matrix plot

    var confusionMatrix = document.createElement("img");
    confusionMatrix.src = "/static/".concat(modelName, "_confusion_matrix.png");
    confusionMatrix.alt = "".concat(modelName, " Confusion Matrix");
    plotContainer.appendChild(confusionMatrix); // Append ROC Curve plot if applicable

    if (["RandomForest", "LogisticRegression", "LSTM"].includes(modelName)) {
      var rocCurve = document.createElement("img");
      rocCurve.src = "/static/".concat(modelName, "_roc_curve.png");
      rocCurve.alt = "".concat(modelName, " ROC Curve");
      plotContainer.appendChild(rocCurve);
    } // Append Loss plot for LSTM


    if (modelName === "LSTM") {
      var lossPlot = document.createElement("img");
      lossPlot.src = "/static/LSTM_loss.png";
      lossPlot.alt = "LSTM Loss Plot";
      plotContainer.appendChild(lossPlot);
    }

    evaluationContainer.appendChild(plotContainer);
  }
});
//# sourceMappingURL=evaluation.dev.js.map
