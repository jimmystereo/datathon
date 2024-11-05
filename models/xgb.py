import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# Load the CSV and create labels
df = pd.read_csv('data/spanish/Cz.csv')
# df.columns
# df = df.pivot(
#     index=['word', 'participant', 'prime', 'target', 'word_type'],  # Rows
#     columns='time',  # Use the unique index as columns
#     values='Fp1'  # Values to aggregate
# ).reset_index()
# Define the labeling function
def labeling(row):
    translation = row['word_type'] == 'translation'

    # Check if prime is the same as target
    if row['prime'] == row['target']:
        return translation

    # Check if the prime and target are known languages
    know_prime = row['prime'] == 'english' or row[row['prime']] == 1
    know_target = row['target'] == 'english' or row[row['target']] == 1
    # print(know_prime, know_target)
    # Return True if all conditions are met
    if  translation and know_prime and know_target:
        return 1
    else:
        return 0
df = df[(df['prime']=='english') & (df['target']=='spanish')]
df
# Apply the labeling function and create the target variable
df['label'] = df.apply(labeling, axis=1).astype(float)
# Define the columns over which you want to calculate the difference
columns = [str(i) for i in range(256)] + ['label']

# Group by the specified columns and apply the difference (max - min) for each column
# df = df.groupby(['word', 'participant'])[columns].agg(lambda x: x.max() - x.min()).reset_index()
# df = df[df['label'] != 2]



# Check the balance of the classes
import matplotlib.pyplot as plt

# Plot class distribution
class_counts = df['label'].value_counts()
class_counts.plot(kind='bar', color='skyblue')

# Add labels and title
plt.xlabel("Know Spanish")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.xticks(rotation=0)  # Optional: adjust rotation for better readability
plt.savefig('dist.png')
# Show the plot
plt.show()


# Select features (update based on your feature columns)
X = df.iloc[:,2:-1]
y = df['label']
# Initialize the XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define cross-validation with Stratified K-Folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation for accuracy and AUC
# cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
# cv_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
#
# # Print cross-validated scores
# print(f"Cross-validated Accuracy: {cv_accuracy.mean():.2f} ± {cv_accuracy.std():.2f}")
# print(f"Cross-validated AUC: {cv_auc.mean():.2f} ± {cv_auc.std():.2f}")

# df.to_csv('spanish_short_t.csv', encoding = 'utf-8-sig', index=False)

df['participant'].value_counts()
df[(df['participant'] == 39) & (df['word']=='book')]
def plot_compare(df, title):
    import matplotlib.pyplot as plt
    # df[df['label'] == True].iloc[:, 5:-1].transpose().plot(label = 'True')
    # df[df['label'] == False].iloc[:, 5:-1].transpose().plot(label = 'False')

    df[columns].transpose().plot(label = "Don't know spanish")
    df[columns].transpose().plot(label = 'Know Spanish')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()

df[df['label']==1].reset_index()
plot_df = df[(df['participant'] == 39) & (df['word']=='book')]
# Define columns2 as the specific columns you want to include in the plot
# Example: If you want to plot columns from 0 to 255
# Define columns2 as the specific columns you want to include in the plot

# Define columns2 as the specific columns you want to include in the plot
columns2 = [str(i) for i in range(256)]

# Create a figure and axis
fig, ax = plt.subplots()

# Extract the "Translation" data for participant 39, word 'book'
translation_data = df[(df['participant'] == 39) & (df['word'] == 'book') & (df['word_type'] == 'translation')][columns2].transpose()
ax.plot(translation_data, label="Translation")

# Extract the "Unrelated" data for participant 39, word 'book'
unrelated_data = df[(df['participant'] == 39) & (df['word'] == 'book') & (df['word_type'] == 'unrelated')][columns2].transpose()
ax.plot(unrelated_data, label="Unrelated")

# Add legend, title, and labels
ax.legend(loc='lower right')
ax.set_title("Cz of participant 39")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Show the combined plot
# plt.show()
plt.savefig('39.png')



