
# modelevaluation.py
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix, roc_curve, auc
import base64
from io import BytesIO
import logging

RESULTS_DIR = './static/results'
PLOTS_DIR = './static/plots'

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_evaluation_results(results_file):
    """Load model evaluation results from a JSON file."""
    results_path = os.path.join(RESULTS_DIR, results_file)
    print(results_file)
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def generate_plots(results_file):
    """Generate interactive plots using Plotly and return their URLs."""
    try:
        results = load_evaluation_results(results_file)
        if not results:
            logging.error("Failed to load results")
            return {"error": "Failed to load results"}
            
        plots = {}
    
        # 1. Actual vs Predicted Comparison Plot
        if 'actual_values' in results and 'predictions' in results:
            print(11)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=results['actual_values'],
                mode='markers',
                name='Actual',
                marker=dict(color='blue', size=8)
            ))
            fig.add_trace(go.Scatter(
                y=results['predictions'],
                mode='markers',
                name='Predicted',
                marker=dict(color='red', size=6)
            ))
            fig.update_layout(
                title="Actual vs Predicted Values",
                xaxis_title="Sample Index",
                yaxis_title="Value"
            )
            plots['actual_vs_predicted'] = fig.to_dict()
        else:
            logging.warning("Missing 'actual_values' or 'predictions' in results.")
            
        # 2. Confusion Matrix Heatmap
        # if 'confusion_matrix' in results:
        #     print(12)
        #     cm = np.array(results['confusion_matrix'])
        
        #     # Validate the confusion matrix shape
        #     if len(cm.shape) != 2 or cm.shape[0] != cm.shape[1]:
        #         logging.error("Confusion matrix has an unexpected shape: %s", cm.shape)
        #         raise ValueError("Confusion matrix should be a square array.")
        
        #     # Dynamically generate labels based on the number of classes
        #     num_classes = cm.shape[0]
        #     x_labels = [f'Predicted {i}' for i in range(num_classes)]
        #     y_labels = [f'Actual {i}' for i in range(num_classes)]
        
        #     fig = go.Figure(data=go.Heatmap(
        #         z=cm,
        #         x=x_labels,
        #         y=y_labels,
        #         colorscale='Blues',
        #         text=cm,
        #         texttemplate="%{text}",
        #         textfont={"size": 16}
        #     ))
        #     fig.update_layout(
        #         title="Confusion Matrix",
        #         xaxis_title="Predicted",
        #         yaxis_title="Actual",
        #         width=600,
        #         height=500
        #     )
        #     plots['confusion_matrix'] = fig.to_dict()
        # else:
        #     logging.warning("Missing 'confusion_matrix' in results.")


        # 3. Feature Importance Plot (for models that support it)
        if 'feature_importance' in results:
            print(13)
            features = list(results['feature_importance'].keys())
            importance = list(results['feature_importance'].values())
            fig = go.Figure(go.Bar(
                x=importance,
                y=features,
                orientation='h'
            ))
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=600
            )
            plots['feature_importance'] = fig.to_dict()
        else:
            logging.warning("Missing 'feature_importance' in results.")

        # 4. Classification Metrics Plot
        if 'classification_report' in results:
            print(14)
            report = results['classification_report']
            metrics = ['precision', 'recall', 'f1-score']
            classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            
            fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)
            
            for i, metric in enumerate(metrics):
                fig.add_trace(go.Bar(
                    x=classes,
                    y=[report[cls][metric] for cls in classes],
                    name=metric
                ), row=1, col=i+1)
                
            fig.update_layout(
                title="Classification Metrics",
                showlegend=False,
                height=400
            )
            plots['classification_metrics'] = fig.to_dict()
        else:
            logging.warning("Missing 'classification_report' in results.")

        # MSE Plot (for regression models)
        if 'mse' in results:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['MSE'],
                y=[results['mse']],
                marker_color='red'
            ))
            fig.update_layout(
                title=f"{results.get('model', 'Model')} Mean Squared Error",
                yaxis_title="MSE",
                width=800,  # Set default width here
                height=500,
                margin=dict(l=50, r=50, b=100, t=100, pad=4)
            )
            plots['mse_plot'] = fig.to_dict()
        
        # Accuracy Plot (for classification models)
        if 'accuracy' in results:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Accuracy'],
                y=[results['accuracy']],
                marker_color='green'
            ))
            fig.update_layout(
                title=f"{results.get('model', 'Model')} Accuracy",
                yaxis_title="Accuracy",
                yaxis_range=[0, 1],
                width=800,  # Set default width here
                height=500,
                margin=dict(l=50, r=50, b=100, t=100, pad=4)
            )
            plots['accuracy_plot'] = fig.to_dict()
        
        # Loss Plot (for neural networks)
        if 'loss' in results:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Loss'],
                y=[results['loss']],
                marker_color='purple'
            ))
            fig.update_layout(
                title=f"{results.get('model', 'Model')} Loss",
                yaxis_title="Loss",
                width=800,  # Set default width here
                height=500,
                margin=dict(l=50, r=50, b=100, t=100, pad=4)
            )
            plots['loss_plot'] = fig.to_dict()
        
        # Evaluation Metrics Plot (for LSTM)
        if all(key in results for key in ['train_mae', 'train_rmse', 'test_mae', 'test_rmse']):
            metrics = ['Train MAE', 'Train RMSE', 'Test MAE', 'Test RMSE']
            values = [results['train_mae'], results['train_rmse'], 
                     results['test_mae'], results['test_rmse']]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=metrics,
                y=values,
                marker_color=['blue', 'green', 'orange', 'red']
            ))
            fig.update_layout(
                title=f"{results.get('model', 'Model')} Evaluation Metrics",
                yaxis_title="Values",
                width=800,  # Set default width here
                height=500,
                margin=dict(l=50, r=50, b=100, t=100, pad=4)
            )
            plots['evaluation_metrics_plot'] = fig.to_dict()
        
        # Confusion Matrix (for classification models)
        # if 'confusion_matrix' in results:
        #     cm = np.array(results['confusion_matrix'])
        #     fig = go.Figure(data=go.Heatmap(
        #         z=cm,
        #         colorscale='Blues',
        #         showscale=True
        #     ))
        #     fig.update_layout(
        #         title="Confusion Matrix",
        #         xaxis_title="Predicted",
        #         yaxis_title="Actual",
        #         width=800,  # Set default width here
        #         height=500,
        #         margin=dict(l=50, r=50, b=100, t=100, pad=4)
                
        #     )
        #     plots['confusion_matrix'] = fig.to_dict()
            
        # ROC Curve (for classification models)
        if 'roc_curve' in results:
            fpr, tpr, _ = results['roc_curve']
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC (AUC = {roc_auc:.2f})',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='grey', dash='dash')
            ))
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                width=800,  # Set default width here
                height=500,
                margin=dict(l=50, r=50, b=100, t=100, pad=4)
            )
            plots['roc_curve'] = fig.to_dict()
        
        # SHAP Feature Importance
        if 'shap_values' in results and 'X_test' in results:
            try:
                shap_values = np.array(results['shap_values'])
                X_test = pd.DataFrame(results['X_test'])
                
                # Create SHAP summary plot
                fig = go.Figure()
                
                # For each feature, add a violin plot
                for i, feature in enumerate(X_test.columns):
                    fig.add_trace(go.Violin(
                        x=[feature]*len(shap_values[:, i]),
                        y=shap_values[:, i],
                        name=feature,
                        box_visible=True,
                        meanline_visible=True
                    ))
                
                fig.update_layout(
                    title="SHAP Feature Importance",
                    xaxis_title="Features",
                    yaxis_title="SHAP Value",
                    showlegend=False,
                    width=800,  # Set default width here
                    height=500,
                    margin=dict(l=50, r=50, b=100, t=100, pad=4)
                )
                plots['shap_feature_importance'] = fig.to_dict()
            except Exception as e:
                print(f"Error creating SHAP plot: {str(e)}")
             
        # Dynamically build the metrics dictionary based on available keys

        metrics = {}
        if 'accuracy' in results:
            print(15)
            metrics['accuracy'] = results['accuracy']
        if 'mse' in results:
            print(16)
            metrics['mse'] = results['mse']
        if 'classification_report' in results:
            print(17)
            weighted_avg = results['classification_report'].get('weighted avg', {})
            metrics['precision'] = weighted_avg.get('precision')
            metrics['recall'] = weighted_avg.get('recall')
            metrics['f1'] = weighted_avg.get('f1-score')

        return {
            "model": results["model"],
            "plots": plots,
            "metrics": metrics
        }

    except Exception as e:
        logging.error(f"Error generating plots: {str(e)}")
        return {"error": str(e)}
    
    
    
        #  # 1. Actual vs Predicted Comparison Plot
        # if 'actual_values' in results and 'predictions' in results:
        #     fig = go.Figure()
        #     fig.add_trace(go.Scatter(
        #         y=results['actual_values'],
        #         mode='markers',
        #         name='Actual Values',
        #         marker=dict(color='blue', size=8)
        #     ))
        #     fig.add_trace(go.Scatter(
        #         y=results['predictions'],
        #         mode='markers',
        #         name='Predictions',
        #         marker=dict(color='red', size=6)
        #     ))
        #     fig.update_layout(
        #         title=f"{results['model']} - Actual vs Predicted",
        #         xaxis_title="Sample Index",
        #         yaxis_title="Value",
        #         showlegend=True
        #     )
        #     plots['actual_vs_predicted'] = fig.to_dict()

        # # 2. Confusion Matrix Heatmap
        # if 'confusion_matrix' in results:
        #     cm = np.array(results['confusion_matrix'])
        #     fig = go.Figure(data=go.Heatmap(
        #         z=cm,
        #         x=['Predicted 0', 'Predicted 1'],
        #         y=['Actual 0', 'Actual 1'],
        #         colorscale='Blues',
        #         text=cm,
        #         texttemplate="%{text}",
        #         textfont={"size":16}
        #     ))
        #     fig.update_layout(
        #         title="Confusion Matrix",
        #         xaxis_title="Predicted",
        #         yaxis_title="Actual",
        #         width=600,
        #         height=500
        #     )
        #     plots['confusion_matrix'] = fig.to_dict()

        # # 3. Feature Importance Plot (for models that support it)
        # if 'feature_importance' in results:
        #     features = list(results['feature_importance'].keys())
        #     importance = list(results['feature_importance'].values())
        #     fig = go.Figure(go.Bar(
        #         x=importance,
        #         y=features,
        #         orientation='h'
        #     ))
        #     fig.update_layout(
        #         title="Feature Importance",
        #         xaxis_title="Importance",
        #         yaxis_title="Features",
        #         height=600
        #     )
        #     plots['feature_importance'] = fig.to_dict()

        # # 4. Classification Metrics Plot
        # if 'classification_report' in results:
        #     report = results['classification_report']
        #     metrics = ['precision', 'recall', 'f1-score']
        #     classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            
        #     fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)
            
        #     for i, metric in enumerate(metrics):
        #         fig.add_trace(go.Bar(
        #             x=classes,
        #             y=[report[cls][metric] for cls in classes],
        #             name=metric
        #         ), row=1, col=i+1)
                
        #     fig.update_layout(
        #         title="Classification Metrics",
        #         showlegend=False,
        #         height=400
        #     )
        #     plots['classification_metrics'] = fig.to_dict()

        # return {
        #     "model": results["model"],
        #     "plots": plots,
        #     "metrics": {
        #         "accuracy": results.get("accuracy"),
        #         "precision": results['classification_report'].get('weighted avg', {}).get('precision'),
        #         "recall": results['classification_report'].get('weighted avg', {}).get('recall'),
        #         "f1": results['classification_report'].get('weighted avg', {}).get('f1-score')
        #     }
        # }
        
        # # Predictions Plot (for regression models)
        # if 'predictions' in results and isinstance(results['predictions'], list):
        #     fig = go.Figure()
        #     fig.add_trace(go.Scatter(
        #         y=results['predictions'],
        #         mode='lines',
        #         name='Predictions',
        #         line=dict(color='blue')
        #     ))
        #     fig.update_layout(
        #         title=f"{results.get('model', 'Model')} Predictions",
        #         xaxis_title="Sample Index",
        #         yaxis_title="Predicted Value",
        #         width=800,  # Set default width here
        #         height=500,
        #         margin=dict(l=50, r=50, b=100, t=100, pad=4)
        #     )
        #     plots['predictions_plot'] = fig.to_dict()
        
        
        # return {"model": results["model"], "plots": plots}
       
        