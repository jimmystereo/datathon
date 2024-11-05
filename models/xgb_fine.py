import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# Load the CSV and filter the DataFrame
df = pd.read_csv('data/spanish/CP6.csv')
df = df[(df['prime'] == 'english') & (df['target'] == 'spanish')]

# Define the labeling function
def labeling(row):
    translation = row['word_type'] == 'translation'
    know_prime = row['prime'] == 'english' or row[row['prime']] == 1
    know_target = row['target'] == 'english' or row[row['target']] == 1

    return 1 if translation and know_prime and know_target else 0

# Apply the labeling function to create the target variable
df['label'] = df.apply(labeling, axis=1).astype(float)

# Define columns to calculate differences
columns = [str(i) for i in range(256)] + ['label']

# Group by 'word' and 'participant', calculating max - min for specified columns
df = df.groupby(['word', 'participant'])[columns].agg(lambda x: x.max() - x.min()).reset_index()

# Print class distribution
class_counts = df['label'].value_counts()
print("Class distribution:")
print(class_counts)

# Prepare features and target variable
X = df.iloc[:, 2:-1]
y = df['label']

# Initialize the XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='roc_auc', cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                           verbose=1, n_jobs=-1)

# Fit the grid search
grid_search.fit(X, y)

# Print the best parameters and score
print("Best parameters found: ", grid_search.best_params_)
print("Best ROC AUC score: ", grid_search.best_score_)

# Use the best model for predictions
best_model = grid_search.best_estimator_

# Evaluate the best model using cross-validation
cv_accuracy = cross_val_score(best_model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
cv_auc = cross_val_score(best_model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc')

# Print cross-validated scores
print(f"Cross-validated Accuracy: {cv_accuracy.mean():.2f} ± {cv_accuracy.std():.2f}")
print(f"Cross-validated AUC: {cv_auc.mean():.2f} ± {cv_auc.std():.2f}")

