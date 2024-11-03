import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# Load the CSV and create labels
df = pd.read_csv('spanish_short.csv')
df.columns
df = df.pivot(
    index=['word', 'participant', 'prime', 'target', 'word_type'],  # Rows
    columns='time',  # Use the unique index as columns
    values='Fp1'  # Values to aggregate
).reset_index()
# Define the labeling function
def labeling(row):
    if row['prime'] == row['target']:
        return True
    translation = row['word_type'] == 'translation'
    know_prime = row['prime'] == 'english' or row.get(row['prime'], 0) == 1
    know_target = row['target'] == 'english' or row.get(row['target'], 0) == 1
    return know_prime and know_target and translation

# Apply the labeling function and create the target variable
df['label'] = df.apply(labeling, axis=1).astype(int)
# Check the balance of the classes
class_counts = df['label'].value_counts()

# Print the class counts
print("Class distribution:")
print(class_counts)

# # Visualize the class distribution
# plt.figure(figsize=(8, 5))
# class_counts.plot(kind='bar', color=['blue', 'orange'])
# plt.title('Class Distribution of the Target Variable')
# plt.xlabel('Class')
# plt.ylabel('Number of Instances')
# plt.xticks(ticks=[0, 1], labels=['Not Match', 'Match'], rotation=0)
# plt.grid(axis='y')
# plt.show()

# Select features (update based on your feature columns)
X = df.iloc[:,5:-1]
y = df['label']
# Initialize the XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define cross-validation with Stratified K-Folds
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Perform cross-validation for accuracy and AUC
cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
cv_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

# Print cross-validated scores
print(f"Cross-validated Accuracy: {cv_accuracy.mean():.2f} ± {cv_accuracy.std():.2f}")
print(f"Cross-validated AUC: {cv_auc.mean():.2f} ± {cv_auc.std():.2f}")

# df.to_csv('spanish_short_t.csv', encoding = 'utf-8-sig', index=False)


df[df['participant'] == 8]
def plot_compare(df, title):
    import matplotlib.pyplot as plt
    df[df['label'] == True].iloc[:, 5:-1].transpose().iloc[:,0].plot(label = 'True')
    df[df['label'] == False].iloc[:, 5:-1].transpose().iloc[:,0].plot(label = 'False')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()
plot_compare(df, 'Accuracy')



# import matplotlib.pyplot as plt
# df[df['label'] == True].iloc[:, 5:-10].transpose().iloc[:,1].plot(label = 'True')
# df[df['label'] == False].iloc[:, 5:-10].transpose().iloc[:,1].plot(label = 'False')
# plt.legend(loc='lower right')
# plt.show()

df