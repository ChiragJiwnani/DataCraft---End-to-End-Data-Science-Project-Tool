// displayplots.js
// import React from 'react';

// const DisplayPlots = ({ modelName }) => {
//     return (
//         <div>
//             <h2>{modelName} Evaluation Results</h2>
//             <div>
//                 <img src={`./static/results/plots/${modelName}_predictions_plot.png`} alt="Predictions Plot" />
//                 <img src={`./static/results/plots/${modelName}_mse_plot.png`} alt="MSE Plot" />
//             </div>
//         </div>
//     );
// };

// export default DisplayPlots;






import React from "react";

const DisplayPlots = ({ plots }) => {
console.log("Received Plots:", plots); 

  return (
    <div className="plots-container">
      {Object.keys(plots).length > 0 ? (
        Object.keys(plots).map((key) => (
          <div key={key}>
            <h3>{key.replace("_", " ")}</h3>
            <img src={`data:image/png;base64,${plots[key]}`} alt={key} />
          </div>
        ))
      ) : (
        <p>No plots available</p>
      )}
    </div>
  );
};

export default DisplayPlots;
