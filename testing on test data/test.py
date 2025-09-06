# Prepare a list to store the results
model_comparison_list = []

# Train and evaluate each model and store results
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    accuracy, precision, recall, f1, roc_auc, conf_matrix = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Add results to the list
    model_comparison_list.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })

    # Display results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    print("-" * 50)

# Convert the list of results into a DataFrame
model_comparison = pd.DataFrame(model_comparison_list)

# Display model comparison table
print("\nModel Comparison Table:")
print(model_comparison)
