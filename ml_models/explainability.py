import json
import shap
import logging
from sklearn.metrics import accuracy_score,precision_score,f1_score 

logger  = logging.getLogger("streamquant.explainability")

def generate_trade_explanation(model, current_features,feature_names,prediction_is_bullish):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(current_features)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]  
    shap_values = shap_values[0] if shap_values.ndim > 1 else shap_values
    feature_values = current_features.iloc[0] if hasattr(current_features, 'iloc') else current_features[0]
    contributions = []
    for name,val,shap_val in zip(feature_names, feature_values, shap_values):
        contributions.append({
            "feature": name,
            "value": round(val,2),
            "shap_value": shap_val
        })
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    top_drivers = contributions[:3]
    direction = "BULLISH" if prediction_is_bullish else "BEARISH"
    return{
        "direction": direction,
        "top_drivers": top_drivers
    }

def export_model_metrics(y_true,y_pred,ticker,filepath="models_metrics.json"):
    metrics = {
        ticker:{
            "Accuracy":round(accuracy_score(y_true,y_pred),4),
            "Precision":round(precision_score(y_true,y_pred,zero_division=0),4),
            "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4)
        }
    }
    try:
        with open(filepath,'r') as f:
            exisiting_data = json.load(f)

    except(FileNotFoundError,json.JSONDecodeError):
        exisiting_data={}

    exisiting_data.update(metrics)
    with open(filepath,'r') as f:
        json.dump(exisiting_data,f,indent=4)
    logger.info(f"Exported health metrics for {ticker} to {filepath}")

    return metrics