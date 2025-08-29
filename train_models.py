import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

datasets = {
    "diabetes": "data/diabetes.csv",
    "heart": "data/heart.csv",
    "kidney": "data/kidney.csv",
    "liver": "data/liver.csv",
    "breast_cancer": "data/breast_cancer.csv"
}

for disease, path in datasets.items():
    print(f"Training model for {disease}...")

    if not os.path.exists(path):
        print(f"❌ Dataset {path} not found. Skipping...")
        continue

    try:
        data = pd.read_csv(path)

        # Use last column as target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        with open(f"models/{disease}_model.pkl", "wb") as f:
            pickle.dump(model, f)

        print(f"✅ {disease} model saved successfully.")

    except Exception as e:
        print(f"⚠️ Error training {disease}: {e}")
