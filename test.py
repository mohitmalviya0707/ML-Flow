import mlflow
print("Current URI:", mlflow.get_tracking_uri())
mlflow.set_tracking_uri("file:./my_mlruns")   # change karo
print("New URI:", mlflow.get_tracking_uri())

