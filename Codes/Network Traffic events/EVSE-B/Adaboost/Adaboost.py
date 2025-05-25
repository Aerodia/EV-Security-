#Adaboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.multiclass import unique_labels

# Load cleaned dataset
df = pd.read_parquet("merged_cleaned.parquet")

# Set target column
target = 'application_name'

# Drop non-informative or identifier-like columns
drop_cols = ['id', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
             'src_oui', 'dst_oui', 'application_category_name', 'application_is_guessed']

# Drop them if they exist
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Encode target labels
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])

# Select only numeric features
X = df.drop(columns=[target]).select_dtypes(include=['number'])
y = df[target]

# Stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train AdaBoost Classifier
clf = AdaBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2))
print("Precision:", round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2))
print("Recall:", round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2))
print("F1 Score:", round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2))

# Classification report
labels_in_use = unique_labels(y_test, y_pred)
class_names_in_use = label_encoder.inverse_transform(labels_in_use)

print("\n Classification Report:\n", classification_report(
    y_test, y_pred,
    labels=labels_in_use,
    target_names=class_names_in_use,
    zero_division=0
))
