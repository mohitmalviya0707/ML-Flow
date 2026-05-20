import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import dagshub

# Enable autologging
mlflow.autolog()

# DagsHub tracking
dagshub.init(repo_owner='mohitmalviya0707', repo_name='ML-Flow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/mohitmalviya0707/ML-Flow.mlflow")

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train model – everything is auto-logged
with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)
    test_accuracy = rf.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
