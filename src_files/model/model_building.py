import os
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import yaml
from src_files.logger import logging
from src_files.model.helper.model_post_processor import EducationLoanPipeline


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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        logging.info("Building Column Transformer Object.")
        feature_columns = ['applicant_age', 'applicant_cibil',
                            'applicant_networth', 'kyc_status', 'date', 'co_applicant',
                            'products_sold_previously', 'co_applicant_age', 'rm_skillset',
                            'university', 'unfavourable_profession', 'co_applicant_cibil',
                            'co_applicant_networth', 'co_applicant_salary', 'co_applicant_dti',
                            'co_applicant_existing_emi', 'loan_size', 'loan_duration', 'loan_type',
                            'interest_rate', 'eligible_emi', 'monthly_emi', 'actual_loan_size']
        
        cat_cols = df[feature_columns].select_dtypes(exclude='number').columns.to_list()
        cat_cols.remove('date')
                        
        ct = ColumnTransformer(
                                transformers=[
                                    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=5), cat_cols)
                                ], 
                                remainder='passthrough', 
                                verbose_feature_names_out=False
                            )

        X_batch = ct.fit_transform(df[feature_columns + ['day','month']].drop(columns='date'))
        y_batch = df['label']
        logging.info('Column Transformer Object built successfully.')
        return X_batch, y_batch, ct
    
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

def train_model(X_train, y_train):
    """Train the Logistic Regression model."""
    try:
        logging.info('Training the model...')
        clf = XGBClassifier(learning_rate=0.05, n_estimators=200, max_depth=5)
        clf.fit(X_train, y_train)
        logging.info('Model training completed')
        return clf
    
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(ct, model,file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        logging.info('Saving the model...')
        pipe = EducationLoanPipeline(ct,model)
        with open(file_path, 'wb') as file:
            pickle.dump(pipe, file)
        logging.info('Pipeline Model saved to %s', file_path)
        
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:

        train_data = load_data('./data/processed/train_processed.csv')
        X_train, y_train, ct = preprocess_data(train_data)
        model = train_model(X_train, y_train)
        os.makedirs('./models', exist_ok=True)
        save_model(ct, model,'./models/education_loan_model.pkl')
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()