# Import necessary libraries
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the data
data = pd.read_csv('/content/fake_job_postings.csv')

# Fill missing values and encode categorical variables
data.fillna('', inplace=True)
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

X = data.drop('fraudulent', axis=1)  # Features
y = data['fraudulent']  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Apply SMOTE for initial oversampling
smote = SMOTE(random_state=42)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)

# Step 2: Apply ADASYN for further oversampling to focus on harder cases
adasyn = ADASYN(random_state=42)
X_resampled_adasyn, y_resampled_adasyn = adasyn.fit_resample(X_resampled_smote, y_resampled_smote)

# Step 3: Apply Tomek Links for undersampling to remove noisy points
tomek = TomekLinks()
X_resampled, y_resampled = tomek.fit_resample(X_resampled_adasyn, y_resampled_adasyn)

# Define a function to evaluate models
def evaluate_model(y_test, y_pred, y_proba=None):
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    # Add ROC AUC if probabilities are provided
    if y_proba is not None:
        metrics['ROC AUC'] = roc_auc_score(y_test, y_proba)
    return metrics

# Define and train models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True)
}

# Initialize dictionary to store results
results = {}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)

    # If model supports predict_proba, get probability scores for ROC AUC
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Evaluate and store the results
    results[model_name] = evaluate_model(y_test, y_pred, y_proba)

# Convert results dictionary to a DataFrame for easy comparison
results_df = pd.DataFrame(results).T

# Print results
print("Model Comparison:")
print(results_df)
