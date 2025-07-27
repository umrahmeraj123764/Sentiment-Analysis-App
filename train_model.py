import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. Load and Prepare Data ---
print("Loading dataset...")
df = pd.read_csv("sentiment_dataset_50k.csv")

# Drop any potential missing values
df.dropna(subset=['text', 'sentiment'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Encode labels
print("Preparing data...")
df['label'] = df['sentiment'].astype('category')
label_mapping = dict(enumerate(df['label'].cat.categories))
df['label'] = df['label'].cat.codes

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# --- 2. Build and Tune the Model ---
print("Building pipeline...")
# We use TfidfVectorizer as it often performs better than CountVectorizer
pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Define a parameter grid for GridSearchCV
# This will test different combinations of parameters to find the best ones.
param_grid = {
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)], # Test unigrams and bigrams
    'tfidfvectorizer__max_df': [0.9, 0.95],          # Ignore words that appear too frequently
    'multinomialnb__alpha': [0.1, 0.5, 1.0],          # Smoothing parameter
}

print("Starting hyperparameter tuning with GridSearchCV...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

print("\n--- Best Parameters Found ---")
print(grid_search.best_params_)

# --- 3. Evaluate the Best Model ---
print("\n--- Model Evaluation on Test Set ---")
y_pred = best_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_mapping.values(), output_dict=True)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_mapping.values()))

# --- 4. Generate and Save Confusion Matrix Plot ---
print("Generating confusion matrix plot...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_mapping.values(), yticklabels=label_mapping.values())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")


# --- 5. Save Artifacts ---
print("\nSaving model and artifacts...")
# Save the trained model
joblib.dump(best_model, 'model.pkl')

# Save the label mapping
joblib.dump(label_mapping, 'label_mapping.pkl')

# Save performance metrics to a JSON file
performance_metrics = {
    'accuracy': accuracy,
    'classification_report': report
}
with open('performance_metrics.json', 'w') as f:
    json.dump(performance_metrics, f, indent=4)

print("\nModel training and saving process completed successfully!")

