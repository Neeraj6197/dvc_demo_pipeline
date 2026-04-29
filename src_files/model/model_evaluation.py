import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import mlflow
import mlflow.sklearn
import os
from src_files.logger import logging

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")  # Update with your MLflow server URI if needed

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            pipe = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return pipe
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        logging.info('Model evaluation started...')
        y_pred = model.predict_output(X_test)
        metrics_dict = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, average="weighted"),
                        "recall": recall_score(y_test, y_pred, average="weighted"),
                        "f1_score": f1_score(y_test, y_pred, average="weighted")
                    }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run(run_name="Education-Loan-model") as run:  # Start an MLflow run
        try:
            clf = load_model('./models/education_loan_model.pkl')
            test_data = load_data('./data/processed/test_processed.csv')
            feature_columns = ['applicant_age', 'applicant_cibil',
                            'applicant_networth', 'kyc_status', 'date', 'co_applicant',
                            'products_sold_previously', 'co_applicant_age', 'rm_skillset',
                            'university', 'unfavourable_profession', 'co_applicant_cibil',
                            'co_applicant_networth', 'co_applicant_salary', 'co_applicant_dti',
                            'co_applicant_existing_emi', 'loan_size', 'loan_duration', 'loan_type',
                            'interest_rate', 'eligible_emi', 'monthly_emi', 'actual_loan_size']
            
            X_test = test_data[feature_columns + ['day','month']].drop(columns='date',axis=1)
            y_test = test_data["label"]

            metrics = evaluate_model(clf, X_test, y_test)
            os.makedirs('./reports', exist_ok=True)
            save_metrics(metrics, './reports/metrics.json')
            
            # Log metrics to MLflow
            logging.info('Logging metrics to MLflow...')
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "model")
            
            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')
            logging.info('Metrics file logged to MLflow')
            
        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()