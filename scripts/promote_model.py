import os 
import mlflow

def promote_model():
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        return EnvironmentError("CAPSTONE_TEST envirenment variable is not set")
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner="gauravbosamiya"
    repo_name="end-to-end-mlops-pipeline-next-word-predictor"

    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    
    clinet = mlflow.MlflowClient()
    
    model_name = "my_model"
    latest_version_staging = clinet.get_latest_versions(model_name, stages=["Staging"])[0].version
    
    prod_version = clinet.get_latest_versions(model_name, stages=["Production"])
    for version in prod_version:
        clinet.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )
        
    clinet.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to production")
    
if __name__=="__main__":
    promote_model()