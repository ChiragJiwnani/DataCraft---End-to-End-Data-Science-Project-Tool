"use strict";

function evaluateModel(modelName, yTest, yPred) {
  var yProba,
      history,
      response,
      evaluationData,
      _args = arguments;
  return regeneratorRuntime.async(function evaluateModel$(_context) {
    while (1) {
      switch (_context.prev = _context.next) {
        case 0:
          yProba = _args.length > 3 && _args[3] !== undefined ? _args[3] : null;
          history = _args.length > 4 && _args[4] !== undefined ? _args[4] : null;
          _context.next = 4;
          return regeneratorRuntime.awrap(fetch('http://localhost:5000/evaluate/${modelName}', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              model_name: modelName,
              y_test: yTest,
              y_pred: yPred // y_proba: yProba,
              // history: history,

            })
          }));

        case 4:
          response = _context.sent;

          if (response.ok) {
            _context.next = 8;
            break;
          }

          console.error('Error fetching evaluation data:', response.statusText);
          return _context.abrupt("return");

        case 8:
          _context.next = 10;
          return regeneratorRuntime.awrap(response.json());

        case 10:
          evaluationData = _context.sent;
          displayEvaluationResults(evaluationData, modelName);

        case 12:
        case "end":
          return _context.stop();
      }
    }
  });
}

function displayEvaluationResults(data, modelName) {
  // Display classification report
  console.log('Classification Report:', data.classification_report || "MSE: ".concat(data.mse)); // Create Image Elements for each plot

  var container = document.getElementById('evaluation-results');
  container.innerHTML = ''; // Clear previous results

  var confusionMatrixImg = document.createElement('img');
  confusionMatrixImg.src = "./static/".concat(modelName, "_confusion_matrix.png");
  confusionMatrixImg.alt = 'Confusion Matrix';
  container.appendChild(confusionMatrixImg);
  var rocCurveImg = document.createElement('img');
  rocCurveImg.src = "./static/".concat(modelName, "_roc_curve.png");
  rocCurveImg.alt = 'ROC Curve';
  container.appendChild(rocCurveImg);
  var lossImg = document.createElement('img');
  lossImg.src = "./static/".concat(modelName, "_loss.png");
  lossImg.alt = 'Training and Validation Loss';
  container.appendChild(lossImg); // Additional reporting information can be displayed in the console or on the webpage
}

evaluateModel(modelName, yTest, yPred);
//# sourceMappingURL=modelevaluation.dev.js.map
