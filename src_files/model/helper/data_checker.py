import pandas as pd
import numpy as np
import datetime as dt


def data_checker(input_data,recovery,predictions,results):
      """
        Validate the input DataFrame for the model post processor.
      """
      ##data check:
      if (int(input_data["applicant_age"]) > 60) | (int(input_data["applicant_age"]) < 18):
        recovery_text = {"summary_message":f"Invalid value for applicant_age.",
                         "factors":{"applicant_age":input_data["applicant_age"]},
                         "rm_recovery_message":"Invalid value for applicant_age. For an eligible applicant age should be between 18-60 years.",
                         "srm_recovery_message":"Invalid value for applicant_age. For an eligible applicant age should be between 18-60 years."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        #return results

      elif input_data["applicant_gender"].lower() not in ["male","female","other"]:
        recovery_text = {"summary_message":f"Invalid value for applicant's gender.",
                         "factors":{"applicant_gender":input_data["applicant_gender"]},
                         "rm_recovery_message":"Invalid value for applicant's gender. Please select a valid gender from Male/Female/Other.",
                         "srm_recovery_message":"Invalid value for applicant's gender. Please select a valid gender from Male/Female/Other."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        #return results

      elif input_data["applicant_education_level"].lower() not in ['masters', 'bachelors', 'high school diploma']:
        recovery_text = {"summary_message":f"Invalid value for applicant_education_level.",
                         "factors":{"applicant_education_level":input_data["applicant_education_level"]},
                         "rm_recovery_message":"Invalid value for applicant_education_level. Please select a value from Masters/Bachelors/High School Diploma.",
                         "srm_recovery_message":"Invalid value for applicant_education_level. Please select a value from Masters/Bachelors/High School Diploma."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        #return results

      elif input_data["applicant_region"].lower() not in ['rural', 'urban']:
        recovery_text = {"summary_message":f"Invalid value for applicant_region.",
                         "factors":{"applicant_region":input_data["applicant_region"]},
                         "rm_recovery_message":"Invalid value for applicant_region. Please select a value from Rural/Urban.",
                         "srm_recovery_message":"Invalid value for applicant_region. Please select a value from Rural/Urban."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif input_data["applicant_marital_status"].lower() not in ['married', 'single']:
        recovery_text = {"summary_message":f"Invalid value for applicant_marital_status.",
                        "factors":{"applicant_marital_status":input_data["applicant_marital_status"]},
                         "rm_recovery_message":"Invalid value for applicant_marital_status. Please select a value from Single/Married.",
                         "srm_recovery_message":"Invalid value for applicant_marital_status. Please select a value from Single/Married."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (int(input_data["applicant_cibil"]) > 900) or (int(input_data["applicant_cibil"]) < 300 & int(input_data["applicant_cibil"]) > 0):
        recovery_text = {"summary_message":f"Invalid value for applicant_cibil.",
                         "factors":{"applicant_cibil":input_data["applicant_cibil"]},
                         "rm_recovery_message":"Invalid value for applicant_cibil. Please enter a value between 300-900.",
                         "srm_recovery_message":"Invalid value for applicant_cibil. Please enter a value between 300-900."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (int(input_data["applicant_networth"]) < 0):
        recovery_text = {"summary_message":f"Invalid value for applicant_networth.",
                         "factors":{"applicant_networth":input_data["applicant_networth"]},
                         "rm_recovery_message":"Invalid value for applicant_networth. Please enter a positive value.",
                         "srm_recovery_message":"Invalid value for applicant_networth. Please enter a positive value."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results
      
      elif (int(input_data["applicant_networth"]) > 100000000):
        recovery_text = {"summary_message":f"Invalid value for applicant_networth.",
                         "factors":{"applicant_networth":input_data["applicant_networth"]},
                         "rm_recovery_message":"Max networth value allowed is 10,00,00,000/-.",
                         "srm_recovery_message":"Max networth value allowed is 10,00,00,000/-."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif input_data["kyc_status"].lower() not in ['compliant', 'non-compliant']:
        recovery_text = {"summary_message":f"Invalid value for kyc_status.",
                          "factors":{"kyc_status":input_data["kyc_status"]},
                         "rm_recovery_message":"Invalid value for kyc_status.Please select a value from Compliant/Non-Compliant.",
                         "srm_recovery_message":"Invalid value for kyc_status.Please select a value from Compliant/Non-Compliant."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif input_data["rm_skillset"].lower() not in ['knowledgeable regarding education loans', 'low effort','follows through','not-knowledgeable']:
        recovery_text = {"summary_message":f"Invalid value for rm_skillset.",
                         "factors":{"rm_skillset":input_data["rm_skillset"]},
                         "rm_recovery_message":"Invalid value for rm_skillset. Please select a value from Knowledgeable Regarding Education Loans/Low Effort/Follows Through/Not-Knowledgeable.",
                         "srm_recovery_message":"Invalid value for rm_skillset. Please select a value from Knowledgeable Regarding Education Loans/Low Effort/Follows Through/Not-Knowledgeable."}
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (int(input_data["no_of_products_sold_previously"]) < 0) | (int(input_data["no_of_products_sold_previously"]) > 2):
          recovery_text = {"summary_message": "Invalid value for no_of_products_sold_previously.",
                          "factors":{"no_of_products_sold_previously":input_data["no_of_products_sold_previously"]},
                          "rm_recovery_message": "Please enter a value for no_of_products_sold_previously between 0-2.",
                          "srm_recovery_message": "Please enter a value for no_of_products_sold_previously between 0-2."}
                          
          recovery.append(recovery_text)
          predictions.append({"Education Loan": "Not Good"})
          
          #return results

      elif input_data["co_applicant"].lower() not in ["available","not-available"]:
        recovery_text = {"summary_message":f"Invalid value for co_applicant.",
                         "factors":{"co_applicant":input_data["co_applicant"]},
                         "rm_recovery_message":"Invalid value for co_applicant. Please select a value from Available/Not-Available.",
                         "srm_recovery_message":"Invalid value for co_applicant. Please select a value from Available/Not-Available."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (input_data["co_applicant"].lower() != "available") & (int(input_data["co_applicant_age"]) > 0):
        recovery_text = {"summary_message":f"Invalid value for co_applicant_age when co-applicant is not-available.",
                      "factors":{"co_applicant":input_data["co_applicant"],
                                 "co_applicant_age":input_data["co_applicant_age"]},
                      "rm_recovery_message":"Please enter the value as 0 if co-applicant is not-available.",
                      "srm_recovery_message":"Please enter the value as 0 if co-applicant is not-available."}
                      
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (input_data["co_applicant"].lower() != "available") & (int(input_data["co_applicant_cibil"]) > 0):
        recovery_text = {"summary_message":f"Invalid value for co_applicant_cibil when co-applicant is not-available.",
                      "factors":{"co_applicant":input_data["co_applicant"],
                                 "co_applicant_cibil":input_data["co_applicant_cibil"]},
                      "rm_recovery_message":"Please enter the value as 0 if co-applicant is not-available.",
                      "srm_recovery_message":"Please enter the value as 0 if co-applicant is not-available."}
                      
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (input_data["co_applicant"].lower() != "available") & (int(input_data["co_applicant_salary"]) > 0):
        recovery_text = {"summary_message":f"Invalid value for co_applicant_salary when co-applicant is not-available.",
                      "factors":{"co_applicant":input_data["co_applicant"],
                                 "co_applicant_salary":input_data["co_applicant_salary"]},
                      "rm_recovery_message":"Please enter the value as 0 if co-applicant is not-available.",
                      "srm_recovery_message":"Please enter the value as 0 if co-applicant is not-available."}
                      
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (input_data["co_applicant"].lower() == "available") & (int(input_data["co_applicant_salary"]) > 100000000):
        recovery_text = {"summary_message":f"Invalid value for co_applicant_salary.",
                          "factors":{"co_applicant_salary":input_data["co_applicant_salary"]},
                         "rm_recovery_message":"Max salary value allowed is 10,00,00,000/-.",
                         "srm_recovery_message":"Max salary value allowed is 10,00,00,000/-."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (input_data["co_applicant"].lower() != "available") & (int(input_data["co_applicant_networth"]) > 0):
        recovery_text = {"summary_message":f"Invalid value for co_applicant_networth when co-applicant is not-available.",
                      "factors":{"co_applicant":input_data["co_applicant"],
                                 "co_applicant_networth":input_data["co_applicant_networth"]},
                      "rm_recovery_message":"Please enter the value as 0 if co-applicant is not-available.",
                      "srm_recovery_message":"Please enter the value as 0 if co-applicant is not-available."}
                      
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (input_data["co_applicant"].lower() == "available") & (int(input_data["co_applicant_networth"]) > 100000000):
        recovery_text = {"summary_message":f"Invalid value for co_applicant_networth.",
                         "factors":{"co_applicant_networth":input_data["co_applicant_networth"]},
                         "rm_recovery_message":"Max networth value allowed is 10,00,00,000/-.",
                         "srm_recovery_message":"Max networth value allowed is 10,00,00,000/-."}
                         
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (input_data["co_applicant"].lower() != "available") & (int(input_data["co_applicant_existing_emi"]) > 0):
        recovery_text = {"summary_message":f"Invalid value for co_applicant_existing_emi when co-applicant is not-available.",
                      "factors":{"co_applicant":input_data["co_applicant"],
                                 "co_applicant_existing_emi":input_data["co_applicant_existing_emi"]},
                      "rm_recovery_message":"Please enter the value as 0 if co-applicant is not-available.",
                      "srm_recovery_message":"Please enter the value as 0 if co-applicant is not-available."}
                      
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (input_data["co_applicant"].lower() == "available") & (int(input_data["co_applicant_age"]) <= 0):
        recovery_text = {"summary_message":f"Invalid value for co_applicant_age when co-applicant is available.",
                      "factors":{"co_applicant_age":input_data["co_applicant_age"]},
                      "rm_recovery_message":"Please enter a valid age for the co-applicant.",
                      "srm_recovery_message":"Please enter a valid age for the co-applicant."}
                      
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (input_data["co_applicant"].lower() == "available") & (int(input_data["co_applicant_cibil"]) < 300):
        recovery_text = {"summary_message":f"Invalid value for co_applicant_cibil when co-applicant is available.",
                      "factors":{"co_applicant_cibil":input_data["co_applicant_cibil"]},
                      "rm_recovery_message":"Please enter a cibil score bertween 300-900.",
                      "srm_recovery_message":"Please enter a cibil score bertween 300-900."}
                      
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (input_data["co_applicant"].lower() == "available") & (int(input_data["co_applicant_salary"]) <= 0):
        recovery_text = {"summary_message":f"Invalid value for co_applicant_salary when co-applicant is available.",
                      "factors":{"co_applicant_salary":input_data["co_applicant_salary"]},
                      "rm_recovery_message":"Please enter a value greater than 0 if a co-applicant is available.",
                      "srm_recovery_message":"Please enter a value greater than 0 if a co-applicant is available."}
                      
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (input_data["co_applicant"].lower() == "available") & (int(input_data["co_applicant_networth"]) < 0):
        recovery_text = {"summary_message":f"Invalid value for co_applicant_networth when co-applicant is available.",
                      "factors":{"co_applicant_networth":input_data["co_applicant_networth"]},
                      "rm_recovery_message":"Please enter a value greater than 0 if a co-applicant is available.",
                      "srm_recovery_message":"Please enter a value greater than 0 if a co-applicant is available."}
                      
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif int(input_data["co_applicant_age"]) > 70:
        recovery_text = {"summary_message": "Invalid value for co_applicant_age.",
                        "factors":{"co_applicant_age":input_data["co_applicant_age"]},
                        "rm_recovery_message": "Co-Applicant's age should be less than 70 years.",
                        "srm_recovery_message": "Co-Applicant's age should be less than 70 years."}
                        
        recovery.append(recovery_text)
        predictions.append({"Education Loan": "Not Good"})
        
        #return results

      elif (int(input_data["co_applicant_age"]) < int(input_data["applicant_age"])) & (int(input_data["co_applicant_age"]) != 0):
          recovery_text = {"summary_message": "Invalid co_applicant_age relation.",
                          "factors": {"co_applicant_age": input_data["co_applicant_age"],
                                      "applicant_age": input_data["applicant_age"]},
                          "rm_recovery_message": "Co-Applicant should not be younger than the applicant.",
                          "srm_recovery_message": "Co-Applicant should not be younger than the applicant."}
                          
          recovery.append(recovery_text)
          predictions.append({"Education Loan": "Not Good"})
          
          #return results

      elif (int(input_data["co_applicant_cibil"]) > 900) or (int(input_data["co_applicant_cibil"]) < 300) and (int(input_data["co_applicant_cibil"]) > 0):
          recovery_text = {"summary_message": "Invalid value for co_applicant_cibil.",
                          "factors":{"co_applicant_cibil":input_data["co_applicant_cibil"]},
                          "rm_recovery_message": "CIBIL score should be between 300-900.",
                          "srm_recovery_message": "CIBIL score should be between 300-900."}
                          
          recovery.append(recovery_text)
          predictions.append({"Education Loan": "Not Good"})
          
          #return results

      elif (input_data["co_applicant"].lower() == "available") and (int(input_data["co_applicant_salary"]) <= 0):
          recovery_text = {"summary_message": "Invalid co_applicant_salary.",
                           "factors":{"co_applicant_salary":input_data["co_applicant_salary"]},
                          "rm_recovery_message": "Please enter a valid amount for co_applicant's salary.",
                          "srm_recovery_message": "Please enter a valid amount for co_applicant's salary."}
                          
          predictions.append({"Education Loan": "Not Good"})
          
          #return results

      elif int(input_data["co_applicant_networth"]) < 0:
          recovery_text = {"summary_message": "Invalid value for co_applicant_networth.",
                          "factors":{"co_applicant_networth":input_data["co_applicant_networth"]},
                          "rm_recovery_message": "Please enter a positive value for co_applicant_networth.",
                          "srm_recovery_message": "Please enter a positive value for co_applicant_networth."}
                          
          recovery.append(recovery_text)
          predictions.append({"Education Loan": "Not Good"})
          
          #return results

      elif int(input_data["co_applicant_existing_emi"]) < 0:
          recovery_text = {"summary_message": "Invalid value for co_applicant_existing_emi.",
                          "factors":{"co_applicant_existing_emi":input_data["co_applicant_existing_emi"]},
                          "rm_recovery_message": "Please enter a positive value for co_applicant_existing_emi.",
                          "srm_recovery_message": "Please enter a positive value for co_applicant_existing_emi."}
                          
          recovery.append(recovery_text)
          predictions.append({"Education Loan": "Not Good"})
          
          #return results

      elif str(input_data["unfavourable_profession"]).lower() not in ['true', 'false']:
          recovery_text = {"summary_message":f"Invalid value for unfavourable_profession.",
                          "factors":{"unfavourable_profession":input_data["unfavourable_profession"]},
                          "rm_recovery_message":"Invalid value for unfavourable_profession.Please select a value from True/False.",
                          "srm_recovery_message":"Invalid value for unfavourable_profession.Please select a value from True/False."}
                          
          recovery.append(recovery_text)
          predictions.append({"Education Loan": "Not Good"})
          
          #return results

      elif (int(input_data["loan_size"]) < 100000) | (int(input_data["loan_size"]) > 20000000):
          recovery_text = {"summary_message": "Invalid value for loan_size.",
                          "factors":{"loan_size":input_data["loan_size"]},
                          "rm_recovery_message": "Please enter a loan amount between 1,00,000 - 2,00,00,000",
                          "srm_recovery_message": "Please enter a loan amount between 1,00,000 - 2,00,00,000"}
                          
          recovery.append(recovery_text)
          predictions.append({"Education Loan": "Not Good"})
          
          #return results

      elif int(input_data["loan_duration"]) <= 11 or int(input_data["loan_duration"]) > 180:
          recovery_text = {"summary_message": "Invalid value for loan_duration.",
                          "factors":{"loan_duration":input_data["loan_duration"]},
                          "rm_recovery_message": "Please enter a loan duration between 12-180 months.",
                          "srm_recovery_message": "Please enter a loan duration between 12-180 months."}
                          
          recovery.append(recovery_text)
          predictions.append({"Education Loan": "Not Good"})
          
          #return results
      
      elif (input_data["interest_rate"]) < 1.0 or (input_data["interest_rate"]) > 16.0:
          recovery_text = {"summary_message": "Invalid value for interest_rate.",
                          "factors":{"interest_rate":input_data["interest_rate"]},
                          "rm_recovery_message": "Please enter an interest_rate between 1.0-16.0 %.",
                          "srm_recovery_message": "Please enter an interest_rate between 1.0-16.0 %."}
                          
          recovery.append(recovery_text)
          predictions.append({"Education Loan": "Not Good"})
          
          #return results

      elif input_data["loan_type"].lower() not in ['domestic', 'international']:
          recovery_text = {"summary_message": "Invalid value for loan_type.",
                          "factors":{"loan_type":input_data["loan_type"]},
                          "rm_recovery_message": "Please select a value from Domestic/International.",
                          "srm_recovery_message": "Please select a value from Domestic/International."}
                          
          recovery.append(recovery_text)
          predictions.append({"Education Loan": "Not Good"})
          
          #return results

      elif input_data["university"].lower() not in ['grade a', 'grade b', 'grade c']:
          recovery_text = {"summary_message": "Invalid value for university.",
                          "factors":{"university":input_data["university"]},
                          "rm_recovery_message": "Please select a value from Grade A/Grade B/Grade C.",
                          "srm_recovery_message": "Please select a value from Grade A/Grade B/Grade C."}
                          
          recovery.append(recovery_text)
          predictions.append({"Education Loan": "Not Good"})
          
          #return results    
          
      else:
         #return results
         pass
      return predictions, recovery, results

