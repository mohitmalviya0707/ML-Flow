from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

# Load data
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)

    # Log child runs
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("cv_mean_accuracy", grid_search.cv_results_["mean_test_score"][i])

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_accuracy", best_cv_score)

    # Test evaluation
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Log datasets
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_dataset = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_dataset, "training")

    test_df = X_test.copy()
    test_df['target'] = y_test
    test_dataset = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_dataset, "testing")

    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(best_model, "random_forest")
    mlflow.set_tag("author", "Mohit Malviya")

    print("Best params:", best_params)
    print("Best CV accuracy:", best_cv_score)
    print("Test accuracy:", test_accuracy)