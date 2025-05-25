import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import warnings

print("Starting")
warnings.filterwarnings("ignore")

# === Config ===
target = 'application_name'
drop_cols = ['id', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
             'src_oui', 'dst_oui', 'application_category_name', 'application_is_guessed']
max_chunks = 1
top_k_features = 50

# === Load Data in Chunks ===
print("Reading Parquet file in chunks...")
pq_file = pq.ParquetFile("merged_cleaned.parquet")
df_list = []

for i in range(min(max_chunks, pq_file.num_row_groups)):
    print(f"Reading chunk {i + 1} of {min(max_chunks, pq_file.num_row_groups)}")
    chunk = pq_file.read_row_group(i).to_pandas()
    print(f"Chunk shape: {chunk.shape}")
    chunk.drop(columns=[col for col in drop_cols if col in chunk.columns], inplace=True)
    df_list.append(chunk)

df = pd.concat(df_list, ignore_index=True)
print("Final concatenated DataFrame shape:", df.shape)

# === Encode Target ===
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])
print("Label encoding complete.")

# === Prepare Features ===
X_raw = df.drop(columns=[target])
numeric_cols = X_raw.select_dtypes(include=['number']).columns

# Downcast numeric columns to save memory
for col in numeric_cols:
    X_raw[col] = pd.to_numeric(X_raw[col], downcast='float')

X = X_raw[numeric_cols]
y = df[target]
print("Feature matrix shape before selection:", X.shape)

# === Remove Constant Features ===
var_thresh = VarianceThreshold()
X_var = var_thresh.fit_transform(X)
print("Shape after VarianceThreshold:", X_var.shape)

# === Select Top K Features ===
selector = SelectKBest(score_func=f_classif, k=min(top_k_features, X_var.shape[1]))
X_selected = selector.fit_transform(X_var, y)
print("Shape after SelectKBest:", X_selected.shape)

# === Remove Rare Classes for Stable Stratified Split ===
value_counts = pd.Series(y).value_counts()
valid_classes = value_counts[value_counts >= 5].index
X_selected = X_selected[pd.Series(y).isin(valid_classes)]
y = y[pd.Series(y).isin(valid_classes)]

# === Normalize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
print("Data split complete. Training...")

# === Define Individual Models ===
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
mlp_clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                        alpha=0.0001, learning_rate_init=0.001, max_iter=500,
                        early_stopping=True, random_state=42)
log_clf = LogisticRegression(max_iter=500, random_state=42)

# === Voting Classifier ===
ensemble = VotingClassifier(
    estimators=[('rf', rf_clf), ('mlp', mlp_clf), ('log', log_clf)],
    voting='soft'  # Use soft voting for better performance
)

# === Train Ensemble ===
ensemble.fit(X_train, y_train)
print("Ensemble model trained successfully!")

# === Predictions & Evaluation ===
y_pred = ensemble.predict(X_test)
print("Predictions complete.")

labels_in_use = unique_labels(y_test, y_pred)
class_names = label_encoder.inverse_transform(labels_in_use)

print("\n--- Evaluation Metrics ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2))
print("Precision:", round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2))
print("Recall:", round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2))
print("F1 Score:", round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, labels=labels_in_use, target_names=class_names, zero_division=0))

# === Confusion Matrix ===
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=labels_in_use)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=90)
plt.tight_layout()
plt.show()
