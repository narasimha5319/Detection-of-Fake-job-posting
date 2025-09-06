# Select a sample from the test set (taking the first example from the test set)
sample_index = 0
sample_data = X_test.iloc[sample_index]
sample_true_label = y_test.iloc[sample_index]

# Choose a model (e.g., Random Forest) for prediction
rf_model = models['Random Forest']

# Predict the class (fraudulent or not) for the selected sample
sample_pred = rf_model.predict([sample_data])
sample_pred_prob = rf_model.predict_proba([sample_data])[:, 1]

# Output the sample details and prediction result
print("Sample Data:")
print(sample_data)
print(f"\nTrue Label: {sample_true_label}")
print(f"Predicted Label: {sample_pred[0]}")
print(f"Prediction Confidence (Fraudulent): {sample_pred_prob[0]:.2f}")

# Output the results to check if the model is working correctly
if sample_pred[0] == 1:
    print(f"The job posting is predicted to be FAKE (Confidence: {sample_pred_prob[0]:.2f})")
else:
    print(f"The job posting is predicted to be REAL (Confidence: {1 - sample_pred_prob[0]:.2f})")
