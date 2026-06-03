import pickle
import pandas as pd

try:
    with open('drug_decission_tree.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"Model type: {type(model)}")
    if hasattr(model, 'feature_names_in_'):
        print(f"Features: {model.feature_names_in_}")
    elif hasattr(model, 'n_features_in_'):
        print(f"Number of features: {model.n_features_in_}")
    
    with open('drug_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler type: {type(scaler)}")
    if hasattr(scaler, 'feature_names_in_'):
        print(f"Scaler features: {scaler.feature_names_in_}")

except Exception as e:
    print(f"Error: {e}")
