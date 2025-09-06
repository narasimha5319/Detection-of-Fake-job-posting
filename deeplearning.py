import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
data = pd.read_csv('/content/fake_job_postings.csv')

# Fill missing values and encode categorical columns
data.fillna('', inplace=True)
for column in data.select_dtypes(include=['object']).columns:
    data[column] = pd.factorize(data[column])[0]

# Feature and target split
X = data.drop('fraudulent', axis=1)
y = data['fraudulent']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (important for neural networks and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Autoencoder model
def build_autoencoder(input_dim):
    input_layer = tf.keras.layers.Input(shape=(input_dim,))

    # Encoder
    encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    bottleneck = tf.keras.layers.Dense(32, activation='relu')(encoded)

    # Decoder
    decoded = tf.keras.layers.Dense(64, activation='relu')(bottleneck)
    decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
    output_layer = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)

    # Full autoencoder
    autoencoder = tf.keras.models.Model(input_layer, output_layer)

    # Encoder model
    encoder = tf.keras.models.Model(input_layer, bottleneck)

    # Decoder model (starting from the bottleneck layer)
    decoder_input = tf.keras.layers.Input(shape=(32,))
    decoder_output = tf.keras.layers.Dense(64, activation='relu')(decoder_input)
    decoder_output = tf.keras.layers.Dense(128, activation='relu')(decoder_output)
    decoder_output = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder_output)
    decoder = tf.keras.models.Model(decoder_input, decoder_output)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder, encoder, decoder

# Train the Autoencoder model
input_dim = X_train_scaled.shape[1]
autoencoder, encoder, decoder = build_autoencoder(input_dim)
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=256, validation_data=(X_test_scaled, X_test_scaled))

# Generate synthetic data by sampling from the bottleneck (latent) layer
encoded_data = encoder.predict(X_train_scaled)

# Sample random latent vectors (simulating new data)
latent_samples = np.random.normal(size=encoded_data.shape)

# Decode the latent samples to generate synthetic data
synthetic_data = decoder.predict(latent_samples)

# Combine original data and synthetic data
X_train_augmented = np.vstack([X_train_scaled, synthetic_data])
y_train_augmented = np.hstack([y_train, y_train[:len(synthetic_data)]])  # Using the same labels for synthetic data

# Initialize models
models = {
    'SVM': SVC(random_state=42, probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced')
}

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # For ROC AUC

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    return accuracy, precision, recall, f1, roc_auc, y_pred

# Store results in a dictionary for each model
results = {}

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    accuracy, precision, recall, f1, roc_auc, y_pred = evaluate_model(model, X_train_augmented, y_train_augmented, X_test_scaled, y_test)

    # Store results for the model
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# Convert results into a DataFrame for easy comparison
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)






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
