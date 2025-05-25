import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
print("Starting")
# Read CSV with low_memory=False to suppress warnings
df = pd.read_csv("/content/drive/My Drive/merged_cleaned-EVSE A.csv", low_memory=False)

# Drop rows with missing target values
df = df.dropna(subset=['application_name'])

# Encode target label
le = LabelEncoder()
df['application_name'] = le.fit_transform(df['application_name'])

# Drop non-numeric / identifier columns
drop_cols = ['id', 'expiration_id', 'src_ip', 'src_mac', 'dst_ip', 'dst_mac', 
             'application_category_name', 'application_is_guessed', 
             'application_confidence', 'src_oui', 'dst_oui']
df = df.drop(columns=drop_cols, errors='ignore')

# Convert all remaining columns to numeric (if any are still objects)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values (after conversion)
df = df.dropna()

# Define features and target
X = df.drop('application_name', axis=1)
y = df['application_name']

# Train-test split
labels = unique_labels(y_test, y_pred)
print(classification_report(y_test, y_pred, labels=labels, target_names=le.inverse_transform(labels)))


# Train AdaBoost
model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
