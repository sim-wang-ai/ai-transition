import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def load_data():
# Small dataset shipped with seaborn would require seaborn; to avoid extra deps we'll create a tiny synthetic dataset,
# or instruct user to provide CSV. Here we create a toy dataset for demonstration.
df = pd.DataFrame({
"age": [22, 38, 26, 35, 35, 54, 2, 27, 14, 4],
"fare": [7.25, 71.2833, 7.925, 53.1, 8.05, 51.8625, 21.075, 11.1333, 30.0708, 16.7],
"sex_male": [1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
"survived": [0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
})
X = df[["age", "fare", "sex_male"]]
y = df["survived"]
return X, y

def train_and_save():
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(solver="liblinear")
)

pipeline.fit(X_train, y_train)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
if name == "main":
train_and_save()
