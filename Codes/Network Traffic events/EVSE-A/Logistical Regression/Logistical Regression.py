import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import RFE

print("Starting")

try:
    # Read CSV with low_memory=False to suppress warnings
    print("Loading data...")
    df = pd.read_csv("merged_cleaned-EVSE A.csv", low_memory=False)

    # Drop rows with missing target values
    df = df.dropna(subset=['application_name'])

    # Encode target label
    le = LabelEncoder()
    df['application_name'] = le.fit_transform(df['application_name'])

    # Select most important features for analysis (already determined in your example)
    selected_features = [
        'src_port', 'dst_port', 'protocol', 'ip_version',
        'bidirectional_packets', 'bidirectional_bytes', 'bidirectional_duration_ms',
        'bidirectional_min_ps', 'bidirectional_mean_ps', 'bidirectional_max_ps',
        'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_max_piat_ms',
        'bidirectional_syn_packets', 'bidirectional_ack_packets', 'bidirectional_fin_packets'
    ]

    # Keep only selected features and target
    df = df[selected_features + ['application_name']]

    # Optimize memory by converting columns to more efficient types
    df['src_port'] = pd.to_numeric(df['src_port'], downcast='integer', errors='coerce')
    df['dst_port'] = pd.to_numeric(df['dst_port'], downcast='integer', errors='coerce')
    df['protocol'] = pd.to_numeric(df['protocol'], downcast='integer', errors='coerce')
    df['ip_version'] = pd.to_numeric(df['ip_version'], downcast='integer', errors='coerce')
    # Apply similar optimization for other numeric features if possible

    # Drop rows with NaN values
    df = df.dropna()

    # Remove rare classes (those with fewer than 40 samples)
    class_counts = df['application_name'].value_counts()
    rare_classes = class_counts[class_counts < 40].index
    df = df[~df['application_name'].isin(rare_classes)]

    print(f"Remaining classes after filtering rare ones: {len(df['application_name'].unique())}")
    print(f"Remaining data size: {df.shape}")

    # Define features and target
    X = df.drop('application_name', axis=1)
    y = df['application_name']

    # Normalize features to non-negative range (important for Logistic Regression)
    print("Scaling features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply Random Oversampling to balance class distribution
    print("Applying oversampling...")
    ros = RandomOverSampler(random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X_scaled, y)

    print(f"Size after oversampling: {X_balanced.shape}")

    # Stratified train-test split to maintain label distribution
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    # Feature selection using RFE (Recursive Feature Elimination) for logistic regression
    print("Selecting important features using RFE...")
    selector = RFE(LogisticRegression(solver='saga', max_iter=1000), n_features_to_select=10)  # Adjust the number of features
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Train Logistic Regression model with incremental learning
    print("Training model with incremental learning...")
    model = LogisticRegression(solver='saga', max_iter=1000, warm_start=True)
    model.fit(X_train_selected, y_train)

    # Predict and evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test_selected)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Additional metrics
    print("\nOverall Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro Precision:", precision_score(y_test, y_pred, average='macro', zero_division=0))
    print("Macro Recall:", recall_score(y_test, y_pred, average='macro', zero_division=0))
    print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro', zero_division=0))

except MemoryError as e:
    print("Memory Error: The process is using too much memory.")
    print(f"Error Details: {e}")

except Exception as e:
    print("An error occurred during processing.")
    print(f"Error Details: {e}")
