import pandas as pd
import numpy as np
import datetime as dt

def pre_prediction_processor(loan_application, results,input_data, predictions, recovery):
        """
        Placeholder function for model prediction.
        """
        # Implement the model prediction logic here
        # Convert input data into a DataFrame
        test_df = pd.DataFrame([loan_application.model_dump()])
        feature_columns = ['applicant_age', 'applicant_cibil',
       'applicant_networth', 'kyc_status', 'date', 'co_applicant',
       'co_applicant_age', 'rm_skillset',
       'university', 'unfavourable_profession', 'co_applicant_cibil',
       'co_applicant_networth', 'co_applicant_salary',
       'co_applicant_existing_emi', 'loan_size', 'loan_duration', 'loan_type',
       'interest_rate']

        ##lowercasing the values
        test_df = test_df[feature_columns]
        test_df["products_sold_previously"] = input_data["no_of_products_sold_previously"]
        case_flag = 0
        #type check:
        try:
          #int
          test_df["applicant_age"] = test_df["applicant_age"].astype(int)
          test_df["applicant_cibil"] = test_df["applicant_cibil"].astype(int)
          test_df["applicant_networth"] = test_df["applicant_networth"].astype(int)
          test_df["co_applicant_age"] = test_df["co_applicant_age"].astype(int)
          test_df["co_applicant_cibil"] = test_df["co_applicant_cibil"].astype(int)
          test_df["co_applicant_networth"] = test_df["co_applicant_networth"].astype(int)
          test_df["co_applicant_salary"] = test_df["co_applicant_salary"].astype(int)
          test_df["co_applicant_existing_emi"] = test_df["co_applicant_existing_emi"].astype(int)
          test_df["loan_size"] = test_df["loan_size"].astype(int)
          test_df["loan_duration"] = test_df["loan_duration"].astype(int)
          test_df["interest_rate"] = test_df["interest_rate"].astype(float)
          test_df["products_sold_previously"] = test_df["products_sold_previously"].astype(int)

          #str
          test_df["kyc_status"] = test_df["kyc_status"].str.lower()
          test_df["co_applicant"] = test_df["co_applicant"].str.lower()
          test_df["rm_skillset"] = test_df["rm_skillset"].str.lower()
          test_df["university"] = test_df["university"].str.lower()
          test_df["unfavourable_profession"] = test_df["unfavourable_profession"].astype(str).str.lower()
          test_df["loan_type"] = test_df["loan_type"].str.lower()

        except Exception as e:
          predictions.append({"Education Loan": "Not Good"})
          recovery_text = {"summary_message":"Invalid value.",
                          "factors":"",
                          "rm_recovery_message":e,
                          "srm_recovery_message":e}
                          
          recovery.append(recovery_text)
          
          case_flag = 1
          #return results

        ##additional columns:
        if input_data["co_applicant"].lower() == "available":
          test_df["co_applicant_dti"] = np.round(input_data["co_applicant_existing_emi"] / input_data["co_applicant_salary"],2)
          test_df["eligible_emi"] = ((input_data["co_applicant_salary"])*0.5) - input_data["co_applicant_existing_emi"]
        else:
          test_df["co_applicant_dti"] = 0
          test_df["eligible_emi"] = 0
        test_df["monthly_emi"] = np.round((test_df["loan_size"]*((test_df["interest_rate"]/100)/12)*((1+((test_df["interest_rate"]/100)/12))**(test_df["loan_duration"])))/((1+((test_df["interest_rate"]/100)/12))**(test_df["loan_duration"])-1),2)
        test_df["actual_loan_size"] = test_df["monthly_emi"] * test_df["loan_duration"]

        try:
          dt.datetime.strptime(test_df["date"].values[0],"%Y-%m-%d")
          if isinstance(test_df["date"], str):
            pass
        except Exception as e:
          predictions.append({"Education Loan": "Not Good"})
          recovery_text = {"summary_message":"Invalid value for date.",
                          "factors":input_data["date"],
                          "rm_recovery_message":"Please enter the date as in yyyy-mm-dd format with exact calender dates. For eg: 2025-12-25.",
                          "srm_recovery_message":"Please enter the date as in yyyy-mm-dd format with exact calender dates. For eg: 2025-12-25."}
                          
          recovery.append(recovery_text)
          
          case_flag = 1
          #return results

        try:
          test_df['day'] = pd.to_datetime(test_df['date']).dt.day
          test_df['month'] = pd.to_datetime(test_df['date']).dt.month
          #drop date column  
          test_df = test_df.drop(columns="date")
          return test_df, results, predictions, recovery
        
        except Exception as e:
          predictions.append({"Education Loan": "Not Good"})
          recovery_text = {"summary_message":"Invalid value for date.",
                          "factors":input_data["date"],
                          "rm_recovery_message":"Please enter the date as in yyyy-mm-dd format with exact calender dates. For eg: 2025-12-25.",
                          "srm_recovery_message":"Please enter the date as in yyyy-mm-dd format with exact calender dates. For eg: 2025-12-25."}
                          
          recovery.append(recovery_text)
          
          case_flag = 1
          return results, test_df, predictions, recovery