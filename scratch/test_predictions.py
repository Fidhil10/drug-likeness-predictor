import joblib
import pandas as pd
import numpy as np

model = joblib.load('drug_decission_tree.pkl')
scaler = joblib.load('drug_scaler.pkl')

# Feature names: ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
# Try a few inputs
tests = [
    [23, 0, 0, 0, 25.355], # Age 23, Sex F (0?), BP HIGH (0?), Chol HIGH (0?), Na_to_K 25.355
    [47, 1, 1, 0, 13.093], # Age 47, Sex M (1?), BP LOW (1?), Chol HIGH (0?), Na_to_K 13.093
]

for test in tests:
    arr = np.array(test).reshape(1, -1)
    # Scaler was for 1 feature? Let's check model.feature_names_in_ again.
    # If the scaler was applied to Na_to_K (index 4)
    test_scaled = test.copy()
    test_scaled[4] = scaler.transform([[test[4]]])[0][0]
    
    pred = model.predict([test_scaled])
    print(f"Input: {test}, Prediction index: {pred[0]}")
