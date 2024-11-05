import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
from datetime import datetime

# Define the directory containing CSV files
directory = 'data/spanish/'  # Replace with your directory path

# Directory to save processed results
output_dir = 'processed_results'
os.makedirs(output_dir, exist_ok=True)

# Define columns for feature engineering
columns = [str(i) for i in range(256)]

# Function to process a single CSV file
def process_csv(file_path, meta):
    # Load data and metadata
    df = pd.read_csv(file_path)

    # Define the labeling function
    def labeling(row):
        translation = row['word_type'] == 'translation'
        if row['prime'] == row['target']:
            return translation
        know_prime = row['prime'] == 'english' or row.get(row['prime'], 0) == 1
        know_target = row['target'] == 'english' or row.get(row['target'], 0) == 1
        return 1 if translation and know_prime and know_target else 0

    # Apply the labeling function
    df['label'] = df.apply(labeling, axis=1).astype(float)
    df = df[(df['prime'] == 'english') & (df['target'] == 'spanish')]
    # Separate translation and unrelated rows
    df_translation = df[df['word_type'] == 'translation']
    df_unrelated = df[df['word_type'] == 'unrelated']

    # Initialize a list to collect the results
    difference_data = []

    # Calculate differences for each word and participant
    for word in df_translation['word'].unique():
        for participant in df_translation['participant'].unique():
            translation_row = df_translation[(df_translation['participant'] == participant) & (df_translation['word'] == word)][columns]
            unrelated_row = df_unrelated[(df_unrelated['participant'] == participant) & (df_unrelated['word'] == word)][columns]
            # print(translation_row.shape)
            # print(unrelated_row.shape)
            if not translation_row.empty and not unrelated_row.empty:
                translation_values = translation_row.to_numpy().flatten()
                unrelated_values = unrelated_row.to_numpy().flatten()
                difference = translation_values - unrelated_values
                difference_data.append([word, participant] + difference.tolist())

    # Create DataFrame with difference features
    columns_with_metadata = ['word', 'participant'] + columns
    df_difference = pd.DataFrame(difference_data, columns=columns_with_metadata)
    df_difference = df_difference.merge(meta, on='participant')

    # Define features and target
    X = df_difference[columns]
    y = df_difference['spanish'].astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(scale_pos_weight=(y == 0).sum() / (y == 1).sum(), eval_metric='aucpr', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Evaluate model
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Create a summary dictionary for the current file
    result_summary = {
        'filename': os.path.basename(file_path),
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'classification_report': report
    }
    return result_summary

# List to store all results
all_results = []

# Load metadata once
meta = pd.read_csv('codes/metadata.csv')

# Iterate through all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        print(f"Processing {filename}...")
        result = process_csv(file_path, meta)
        all_results.append(result)

# Save the results to a new CSV file
result_df = pd.DataFrame(all_results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_df.to_csv(f'{output_dir}/summary_results_{timestamp}.csv', index=False)

print(f"Results saved to {output_dir}/summary_results_{timestamp}.csv")
