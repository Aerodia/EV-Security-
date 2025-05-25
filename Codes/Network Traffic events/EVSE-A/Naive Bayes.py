import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler

print("Starting...")

try:
    # Step 1: Read only necessary columns
    drop_cols = [
        'id', 'expiration_id', 'src_ip', 'src_mac', 'dst_ip', 'dst_mac',
        'application_category_name', 'application_is_guessed',
        'application_confidence', 'src_oui', 'dst_oui'
    ]
    
    df = pd.read_csv("merged_cleaned-EVSE A.csv", low_memory=False)
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Step 2: Drop rows with missing target
    df.dropna(subset=['application_name'], inplace=True)

    # Step 3: Encode target labels
    le = LabelEncoder()
    df['application_name'] = le.fit_transform(df['application_name'].astype(str))

    # Step 4: Remove rare classes (<40 samples)
    class_counts = df['application_name'].value_counts()
    valid_classes = class_counts[class_counts >= 40].index
    df = df[df['application_name'].isin(valid_classes)]

    print(f"Remaining classes: {len(valid_classes)}")
    print(f"Remaining data shape: {df.shape}")

    # Step 5: Convert features to numeric and downcast to save memory
    feature_cols = df.columns.drop('application_name')
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    for col in feature_cols:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')

    # Step 6: Split features and target
    X = df.drop('application_name', axis=1)
    y = df['application_name']
    del df
    gc.collect()

    # Step 7: Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    del X
    gc.collect()

    # Step 8: Train-test split BEFORE oversampling
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    del X_scaled, y
    gc.collect()

    # Step 9: Oversample only the training data
    ros = RandomOverSampler(random_state=42, sampling_strategy='not majority')
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
    del X_train, y_train
    gc.collect()

    print(f"Training data shape after oversampling: {X_train_balanced.shape}")

    # Step 10: Train and evaluate model
    model = GaussianNB()
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)

    # Step 11: Evaluate
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("Macro Precision:", precision_score(y_test, y_pred, average='macro', zero_division=0))
    print("Macro Recall:", recall_score(y_test, y_pred, average='macro', zero_division=0))
    print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro', zero_division=0))

except MemoryError as e:
    print("Memory Error: The process is using too much memory.")
    print(f"Error Details: {e}")

except Exception as e:
    print("An error occurred during processing.")
    print(f"Error Details: {e}")
