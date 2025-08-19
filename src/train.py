import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model():
    # Load the processed data (ensure the 'is_high_risk' column is included)
    df = pd.read_csv("data/processed/processed_data_with_risk.csv")  # Update with correct path if needed
    
    # Check if 'is_high_risk' column exists
    if 'is_high_risk' not in df.columns:
        print("Error: 'is_high_risk' column not found in the data")
        return

    # Features and target
    X = df.drop(columns=["is_high_risk"])  # Features
    y = df["is_high_risk"]  # Target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier (or any other classifier you prefer)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()
