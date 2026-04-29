import pickle
import pandas as pd
from src_files.logger import logging
import time
import json

pipe_path = "./models/education_loan_model.pkl"

# Load the model from a pickle file
with open(pipe_path, 'rb') as file:
    pipe = pickle.load(file)
logging.info("Pipeline Model Loaded.")
# print(pipe)
#load the test data:
test_df = pd.read_csv("./data/processed/train_processed.csv")
print("size:",test_df.shape)
#prediction on test data:
feature_columns = ['applicant_age', 'applicant_cibil',
                    'applicant_networth', 'kyc_status', 'date', 'co_applicant',
                    'products_sold_previously', 'co_applicant_age', 'rm_skillset',
                    'university', 'unfavourable_profession', 'co_applicant_cibil',
                    'co_applicant_networth', 'co_applicant_salary', 'co_applicant_dti',
                    'co_applicant_existing_emi', 'loan_size', 'loan_duration', 'loan_type',
                    'interest_rate', 'eligible_emi', 'monthly_emi', 'actual_loan_size']

test_df = test_df[feature_columns + ['day','month']].drop(columns='date')

start_time = time.time()
print("Generating Predictions.",start_time)
#print(test_df.head(1).to_json(orient='records'))

prediction = pipe.predict_output(test_df)
print(len(prediction))
#result = pipe.process_output(prediction, test_df)
end_time = time.time()
print("Results generated at",end_time)
time_taken = end_time - start_time
print(f"Time taken for prediction: {time_taken:.4f} seconds")
