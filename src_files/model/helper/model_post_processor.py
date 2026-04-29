import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src_files.logger import logging

recovery = []
predictions = []

class EducationLoanPipeline:
   
    def __init__(self, ct, model):
        self.ct = ct
        self.model = model
    
    def model_pipeline(self):
       pipe = Pipeline(steps=[('preprocessor', self.ct), 
                              ('model', self.model)])
       return pipe
       
    def post_processor(self,prediction,df, recovery, predictions):
          """
                This function is a placeholder for post-processing logic.
                It can be used to transform the model output into a more interpretable format.
          """  

          #prediction label:
          pipe = self.model_pipeline()
          logging.info("Analyzing model predictions.")
          prediction = int(prediction[0])
          logging.info("Applying labels to the predictions.")
          if prediction == 0:
            prediction_label = "Fair"
          elif prediction == 1:
            prediction_label = "Good"
          else:
            prediction_label = "Not Good"

          applicant_features = ['applicant_age','applicant_cibil','applicant_networth','kyc_status',
                                'university','loan_size', 'loan_duration']

          co_applicant_features = ['applicant_age','kyc_status', 'co_applicant_age',
                                  'university', 'co_applicant_cibil','co_applicant_networth',
                                  'co_applicant_salary', 'loan_size', 'loan_duration']

          if df["co_applicant"].values[0].lower() != "available":
              feature_list = {i:str(df[i].values[0]) for i in applicant_features}
          else:
              feature_list = {i:str(df[i].values[0]) for i in co_applicant_features}

          logging.info("Generating recovery messages.")
          #Good
          if prediction == 1:
            if df["rm_skillset"].values[0].lower() in ['low effort','follows through','not-knowledgeable']:
                recovery_text = {"summary_message":f"This is a Good opportunity because of following factors:",
                                "factors":feature_list,
                                "rm_recovery_message":"The opportunity is good, but the RM should be knowledgeable with all the product features.",
                                "srm_recovery_message":"The opportunity is good, but the RM should be knowledgeable with all the product features."}

            elif df["co_applicant"].values[0].lower() != "available":
                recovery_text = {"summary_message":f"This is a Good opportunity because of following factors:",
                                "factors":feature_list,
                                "rm_recovery_message":"Loan Application can be processed further.",
                                "srm_recovery_message":"Loan Application can be processed further."}
            elif str(df["unfavourable_profession"].values[0]).lower() == "true":
              prediction_label = "Fair"
              feature_list["unfavourable_profession"] = True
              recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                        "factors":{"unfavourable_profession":"True"},
                                  "rm_recovery_message":"The profession is unfavourable for the current loan application. However, other factors can be considered.",
                                  "srm_recovery_message":"The profession is unfavourable for the current loan application. However, other factors can be considered."}
            else:
              recovery_text = {"summary_message":f"This is a Good opportunity because of following factors:",
                                      "factors":feature_list,
                              "rm_recovery_message":"Loan Application can be processed further.",
                              "srm_recovery_message":"Loan Application can be processed further."}                        
            recovery.append(recovery_text)

          #Fair
          if prediction == 0:
            if df["rm_skillset"].values[0].lower() in ['low effort','follows through','not-knowledgeable']:
                recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                "factors":{"rm_skillset":df["rm_skillset"].values[0].lower()},
                                "rm_recovery_message":"The opportunity is Fair, RM should increase efforts to convert it to a good opportunity.",
                                "srm_recovery_message":"The opportunity is Fair, RM should increase efforts to convert it to a good opportunity."}
                recovery.append(recovery_text)
            elif str(df["unfavourable_profession"].values[0]).lower() == "true":
              feature_list["unfavourable_profession"] = True
              recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                        "factors":feature_list,
                                  "rm_recovery_message":"The profession is unfavourable for the current loan application. However, other factors can be considered.",
                                  "srm_recovery_message":"The profession is unfavourable for the current loan application. However, other factors can be considered."}
              recovery.append(recovery_text)
            elif df["co_applicant"].values[0].lower() != "available":
                recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                      "factors":feature_list,
                                "rm_recovery_message":"In order to convert it to a good opportunity decrease the loan size or add a co-applicant.",
                                "srm_recovery_message":"In order to convert it to a good opportunity decrease the loan size or add a co-applicant."}
                                
                recovery.append(recovery_text)
            elif (df["eligible_emi"].values[0] < df["monthly_emi"].values[0]) & (df["co_applicant_networth"].values[0]*0.07 >= df["loan_size"].values[0]):
                recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                      "factors":feature_list,
                                "rm_recovery_message":"The salary of the co-applicant is not adequate incase of need, however, bank may consider co-applicant's networth as a factor.",
                                "srm_recovery_message":"The salary of the co-applicant is not adequate incase of need, however, bank may consider co-applicant's networth as a factor."}
                recovery.append(recovery_text)
            elif ((df["loan_duration"].values[0]/12) + df["applicant_age"].values[0]) > 80:
                prediction_label = "Fair"
                recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                      "factors":feature_list,
                                "rm_recovery_message":"Please decrease the loan duration.",
                                "srm_recovery_message":"Please decrease the loan duration."}
                recovery.append(recovery_text)
            elif df["loan_duration"].values[0] >=72:
              for i in range(1,109):
                temp_df = df.copy()
                temp_df["loan_duration"] -= i
                pred = int(self.predict_output(temp_df)[0])
                if pred == 1:
                  if df["co_applicant"].values[0].lower() == "not-available":
                      feature_list = {i:str(df[i].values[0]) for i in applicant_features}
                  else:
                      feature_list = {i:str(df[i].values[0]) for i in co_applicant_features}
                  prediction_label ="Fair"
                  recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                  "factors":feature_list,
                                  "rm_recovery_message":f"In order to convert it to a good opportunity decrease the loan duration by {i} months.",
                                  "srm_recovery_message":f"In order to convert it to a good opportunity decrease the loan duration by {i} months."}
                                  
                  break
                else:
                  feature_list["loan_size"] = df["loan_size"]
                  prediction_label ="Not Good"
                  if df["co_applicant"].values[0].lower() == "not-available":
                    recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                      "factors":{"co_applicant":df["co_applicant"].values[0],
                                                 "loan_size":str(df["loan_size"].values[0]),
                                                 "loan_duration":str(df["loan_duration"].values[0])},
                                      "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request.",
                                      "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request."}
                  elif ((df["eligible_emi"].values[0] > df["monthly_emi"].values[0]) or (df["co_applicant_networth"].values[0]*0.07 >= df["loan_size"].values[0])) and ((df["loan_duration"].values[0]/12) + df["applicant_age"].values[0]) <= 80:
                    prediction_label = "Good"
                    recovery_text = {"summary_message":f"This is a Good opportunity because of following factors:",
                                      "factors":feature_list,
                                      "rm_recovery_message":"Loan Application can be processed further.",
                                      "srm_recovery_message":"Loan Application can be processed further."}
                  else:
                    recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                      "factors":{"co_applicant_salary":str(df["co_applicant_salary"].values[0]),
                                                 "co_applicant_dti" : str(df["co_applicant_dti"].values[0]*100),
                                                 "loan_size":str(df["loan_size"].values[0]),
                                                 "loan_duration":str(df["loan_duration"].values[0])},
                                      "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer.",
                                      "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer.",}
                                      
              recovery.append(recovery_text)
            else:
                for i in range(1,15):
                  temp_df = df.copy()
                  temp_df["loan_size"] -= i*10000
                  print(temp_df["loan_size"].values[0])
                  if temp_df["loan_size"].values[0] < 100000:
                    recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                              "factors":{"co_applicant_salary":str(df["co_applicant_salary"].values[0]),
                                                 "co_applicant_dti" : str(df["co_applicant_dti"].values[0]*100),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                    "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the repayment capacity of the customer.",
                                    "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the repayment capacity of the customer."}
                                    
                    prediction_label = "Not Good"
                    break
                  elif ((df["eligible_emi"].values[0] > df["monthly_emi"].values[0]) or (df["co_applicant_networth"].values[0]*0.07 >= df["loan_size"].values[0])) and ((df["loan_duration"].values[0]/12) + df["applicant_age"].values[0]) <= 80:
                    prediction_label = "Good"
                    recovery_text = {"summary_message":f"This is a Good opportunity because of following factors:",
                                      "factors":feature_list,
                                      "rm_recovery_message":"Loan Application can be processed further.",
                                      "srm_recovery_message":"Loan Application can be processed further."}
                    break
                  else:
                    temp_df["monthly_emi"] = np.round((temp_df["loan_size"]*((temp_df["interest_rate"]/100)/12)*((1+((temp_df["interest_rate"]/100)/12))**(temp_df["loan_duration"])))/((1+((temp_df["interest_rate"]/100)/12))**(temp_df["loan_duration"])-1),2)
                    temp_df["actual_loan_size"] = temp_df["monthly_emi"] * temp_df["loan_duration"]
                    pred = int(self.predict_output(temp_df)[0])
                    if pred == 1:
                      recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                            "factors":{"co_applicant_salary": str(df["co_applicant_salary"].values[0]),
                                                 "co_applicant_dti" : str(df["co_applicant_dti"].values[0]*100),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                      "rm_recovery_message":f"Since the bank's policy is to consider the co-applicant's repaying capacity at the time of application (and consider the applicant's repaying capacity at a later stage) the present loan amount exceeds the co-applicant's repaying capacity. Please consider decreasing the loan amount.",
                                      "srm_recovery_message":f"Since the bank's policy is to consider the co-applicant's repaying capacity at the time of application (and consider the applicant's repaying capacity at a later stage) the present loan amount exceeds the co-applicant's repaying capacity. Please consider decreasing the loan amount."}
                                      
                      break

                    else:
                      recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                              "factors":{"co_applicant_salary": str(df["co_applicant_salary"].values[0]),
                                                 "co_applicant_dti" : str(df["co_applicant_dti"].values[0]*100),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                    "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the repayment capacity of the customer.",
                                    "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the repayment capacity of the customer."}
                                    
                      prediction_label = "Not Good"
                recovery.append(recovery_text)


          #Not Good
          if prediction == 2:
            feature_list = {"co_applicant":df["co_applicant"]}
            flag = 0
            if df["co_applicant"].values[0].lower() == "available":
                feature_list.pop("co_applicant")
            if df["university"].values[0].lower() == "grade c":
                feature_list["university"] = "grade c"
                flag = 2
                recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                        "factors":feature_list,
                              "rm_recovery_message":"Loan Application should not be processed further as the university lies in Grade C category.",
                              "srm_recovery_message":"Loan Application should not be processed further as the university lies in Grade C category."}
            if df["kyc_status"].values[0].lower() == "non-compliant":
                feature_list["kyc_status"] = "non-compliant"
                flag = 2
                recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                        "factors":feature_list,
                              "rm_recovery_message":"Loan Application should not be processed further as the KYC status needs to be compliant.",
                              "srm_recovery_message":"Loan Application should not be processed further as the KYC status needs to be compliant."}
            
            if (df["co_applicant_cibil"].values[0] < 600) & (df["co_applicant"].values[0].lower() == "available"):
              feature_list["co_applicant_cibil"] = str(df["co_applicant_cibil"].values[0])
              flag = 1
            if df["co_applicant_dti"].values[0] >= 0.5:
              feature_list["co_applicant_dti"] = str(df["co_applicant_dti"].values[0])
              flag = 1
            if (df["applicant_cibil"].values[0] < 600) & (df["co_applicant"].values[0].lower() != "available") & (flag == 0):
              feature_list["applicant_cibil"] = str(df["applicant_cibil"].values[0])
              flag = 1

            if flag == 1:
              recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                        "factors":feature_list,
                              "rm_recovery_message":"Loan Application should not be processed further for the current applicant because of the stated factors.",
                              "srm_recovery_message":"Loan Application should not be processed further for the current applicant because of the stated factors."}
                              
              recovery.append(recovery_text)
            ##    
            elif flag == 0:
              if df["rm_skillset"].values[0].lower() in ['low effort','not-knowledgeable']:
                  if df["co_applicant"].values[0].lower() == "not-available":
                          feature_list = {i:str(df[i].values[0]) for i in applicant_features}
                  else:
                          feature_list = {i:str(df[i].values[0]) for i in co_applicant_features}
                  feature_list["rm_skillset"] = df["rm_skillset"]
                  recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                "factors":{"rm_skillset":df["rm_skillset"].values[0].lower()},
                                "rm_recovery_message":"The opportunity is Fair, RM should increase efforts to convert it to a good opportunity.",
                                "srm_recovery_message":"The opportunity is Fair, RM should increase efforts to convert it to a good opportunity."}
                  prediction_label = "Fair"
                  recovery.append(recovery_text)
              elif df["loan_size"] <= 1000000:
                  if df["loan_duration"] < 150:
                    for i in range(1,25):
                      temp_df = df.copy()
                      temp_df["loan_size"] -= i*10000
                      if temp_df["loan_size"].values[0] < 100000:
                        if df["co_applicant"].values[0].lower() == "not-available":
                          recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                                          "factors":{"co_applicant": str(df["co_applicant"].values[0]),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                                "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request.",
                                                "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request."}
                                                
                        else:
                          recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                            "factors":{"co_applicant_salary": str(df["co_applicant_salary"].values[0]),
                                                 "co_applicant_dti" : str(df["co_applicant_dti"].values[0]*100),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                            "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer.",
                                            "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer."}
                                            
                        break
                      else:
                        temp_df["monthly_emi"] = np.round((temp_df["loan_size"]*((temp_df["interest_rate"]/100)/12)*((1+((temp_df["interest_rate"]/100)/12))**(temp_df["loan_duration"])))/((1+((temp_df["interest_rate"]/100)/12))**(temp_df["loan_duration"])-1),2)
                        temp_df["actual_loan_size"] = temp_df["monthly_emi"] * temp_df["loan_duration"]
                        pred = int(self.predict_output(temp_df)[0])
                        if pred == 1:
                          if df["co_applicant"].values[0].lower() == "not-available":
                              feature_list = {i:str(df[i].values[0]) for i in applicant_features}
                              prediction_label ="Fair"
                              recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                          "factors":{"co_applicant":"Not-Available",
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                          "rm_recovery_message":f"This is a Fair opportunity because the loan amount is quite high with respect to the repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request. Please decrease the loan amount or add a co-applicant.",
                                          "srm_recovery_message":f"This is a Fair opportunity because the loan amount is quite high with respect to the repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request. Please decrease the loan amount or add a co-applicant."}
                          
                          else:
                              feature_list = {i:str(df[i].values[0]) for i in co_applicant_features}
                              prediction_label ="Fair"
                              recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                          "factors":{"co_applicant_salary": str(df["co_applicant_salary"].values[0]),
                                                 "co_applicant_dti" : str(df["co_applicant_dti"].values[0]*100),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                          "rm_recovery_message":f"Since the bank's policy is to consider the co-applicant's repaying capacity at the time of application (and consider the applicant's repaying capacity at a later stage) the present loan amount exceeds the co-applicant's repaying capacity. Please consider decreasing the loan amount.",
                                          "srm_recovery_message":f"Since the bank's policy is to consider the co-applicant's repaying capacity at the time of application (and consider the applicant's repaying capacity at a later stage) the present loan amount exceeds the co-applicant's repaying capacity. Please consider decreasing the loan amount."}
                          break
                        else:
                          feature_list["loan_size"] = df["loan_size"]
                          if df["co_applicant"].values[0].lower() == "not-available":
                            recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                                          "factors":{"co_applicant":"Not-Available",
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                                "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request.",
                                                "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request."}
                                                
                          else:
                            recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                                          "factors":{"co_applicant_salary":df["co_applicant_salary"].values[0],
                                                 "co_applicant_dti" : str(df["co_applicant_dti"].values[0]*100),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                                "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer.",
                                                "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer."}
                                                
                  else:
                      for i in range(1,109):
                        temp_df = df.copy()
                        temp_df["loan_duration"] = int(df["loan_duration"].values[0]) - i
                        pred = int(self.predict_output(temp_df)[0])
                        if pred == 1:
                          if df["co_applicant"].values[0].lower() == "not-available":
                              feature_list = {i:str(df[i].values[0]) for i in applicant_features}
                          else:
                              feature_list = {i:str(df[i].values[0]) for i in co_applicant_features}
                          prediction_label ="Fair"
                          recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                          "factors":feature_list,
                                            "rm_recovery_message":f"In order to convert it to a good opportunity decrease the loan duration by {i} months.",
                                            "srm_recovery_message":f"In order to convert it to a good opportunity decrease the loan duration by {i} months."}
                                            
                          break
                        else:
                          feature_list["loan_size"] = df["loan_size"]
                          if df["co_applicant"].values[0].lower() == "not-available":
                            recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                              "factors":{"co_applicant": str(df["co_applicant"].values[0]),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                              "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request.",
                                              "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request."}
                                              
                          else:
                            recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                            "factors":{"co_applicant_salary": str(df["co_applicant_salary"].values[0]),
                                                 "co_applicant_dti" : str(df["co_applicant_dti"].values[0]*100),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                            "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer.",
                                            "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer."}
                                            
                  recovery.append(recovery_text)
              else:
                if df["loan_duration"] < 150:
                  for i in range(1,5):
                    temp_df = df.copy()
                    temp_df["loan_size"] -= i*100000
                    temp_df["monthly_emi"] = np.round((temp_df["loan_size"]*((temp_df["interest_rate"]/100)/12)*((1+((temp_df["interest_rate"]/100)/12))**(temp_df["loan_duration"])))/((1+((temp_df["interest_rate"]/100)/12))**(temp_df["loan_duration"])-1),2)
                    temp_df["actual_loan_size"] = temp_df["monthly_emi"] * temp_df["loan_duration"]
                    pred = int(pipe.predict_output(temp_df)[0])
                    if pred == 1:
                      if df["co_applicant"].values[0].lower() == "not-available":
                              feature_list = {i:str(df[i].values[0]) for i in applicant_features}
                              prediction_label ="Fair"
                              recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                          "factors":{"co_applicant":"Not-Available",
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                          "rm_recovery_message":f"This is a Fair opportunity because the loan amount is quite high with respect to the repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request. Please decrease the loan amount or add a co-applicant.",
                                          "srm_recovery_message":f"This is a Fair opportunity because the loan amount is quite high with respect to the repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request. Please decrease the loan amount or add a co-applicant."}
                          
                      else:
                          feature_list = {i:str(df[i].values[0]) for i in co_applicant_features}
                          prediction_label ="Fair"
                          recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                      "factors":{"co_applicant_salary": str(df["co_applicant_salary"].values[0]),
                                              "co_applicant_dti" : str(df["co_applicant_dti"].values[0]*100),
                                              "loan_size": str(df["loan_size"].values[0]),
                                              "loan_duration": str(df["loan_duration"].values[0])},
                                      "rm_recovery_message":f"Since the bank's policy is to consider the co-applicant's repaying capacity at the time of application (and consider the applicant's repaying capacity at a later stage) the present loan amount exceeds the co-applicant's repaying capacity. Please consider decreasing the loan amount.",
                                      "srm_recovery_message":f"Since the bank's policy is to consider the co-applicant's repaying capacity at the time of application (and consider the applicant's repaying capacity at a later stage) the present loan amount exceeds the co-applicant's repaying capacity. Please consider decreasing the loan amount."}
                      break
                    else:
                      feature_list["loan_size"] = df["loan_size"]
                      if df["co_applicant"].values[0].lower() == "not-available":
                        recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                                      "factors":{"co_applicant": df["co_applicant"].values[0],
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                            "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request.",
                                            "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request."}
                                            
                      else:
                        recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                                      "factors":{"co_applicant_salary":df["co_applicant_salary"].values[0],
                                                 "co_applicant_dti" : str(df["co_applicant_dti"].values[0]*100),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                            "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer.",
                                            "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer."}
                                            
                else:
                      for i in range(1,109):
                        temp_df = df.copy()
                        temp_df["loan_duration"] -= i
                        pred = int(self.predict_output(temp_df)[0])
                        if pred == 1:
                          if df["co_applicant"].values[0].lower() == "not-available":
                              feature_list = {i:str(df[i].values[0]) for i in applicant_features}
                          else:
                              feature_list = {i:str(df[i].values[0]) for i in co_applicant_features}
                          prediction_label ="Fair"
                          recovery_text = {"summary_message":f"This is a Fair opportunity because of following factors:",
                                            "factors":feature_list,
                                      "rm_recovery_message":f"In order to convert it to a good opportunity decrease the loan duration by {i} months.",
                                      "srm_recovery_message":f"In order to convert it to a good opportunity decrease the loan duration by {i} months."}
                                      
                          break
                        else:
                          feature_list["loan_size"] = df["loan_size"]
                          if df["co_applicant"].values[0].lower() == "not-available":
                            recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                                          "factors":{"co_applicant": str(df["co_applicant"].values[0]),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                                "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request.",
                                                "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer. Additionally, there is no co-applicant associated with this loan request."}
                                                
                          else:
                            recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                                          "factors":{"co_applicant_salary":df["co_applicant_salary"].values[0],
                                                 "co_applicant_dti" : str(df["co_applicant_dti"].values[0]*100),
                                                 "loan_size": str(df["loan_size"].values[0]),
                                                 "loan_duration": str(df["loan_duration"].values[0])},
                                                "rm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer.",
                                                "srm_recovery_message":"The loan application should not be processed because the loan amount is too high with respect to the present repayment capacity of the customer."}
                                                
                  
                recovery.append(recovery_text)
            else:
                if not recovery_text:
                  recovery_text = {"summary_message":f"This is Not a Good opportunity because of following factors:",
                                            "factors":feature_list,
                                  "rm_recovery_message":"Loan Application should not be processed further for the current applicant.",
                                  "srm_recovery_message":"Loan Application should not be processed further for the current applicant."}
                                  
                  recovery.append(recovery_text)
                else:
                  recovery.append(recovery_text)
          predictions.append({"Education Loan": prediction_label})  
          results={"predictions": predictions,
                  "recovery_details":recovery}
          logging.info("Post-processing Predictions completed successfully.")
          return results

    def predict_output(self,df):
        pipe = self.model_pipeline()
        prediction = pipe.predict(df)
        return prediction
    
    def process_output(self,prediction,df):        
        result = self.post_processor(prediction,df,recovery, predictions)
        return result