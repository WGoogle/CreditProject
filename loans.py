import pandas as pd
import numpy as np

#fico_range_low, fico_range_high, issue_d

col_names = [ "issue_d", "fico_range_low", "fico_range_high", "loan_amnt", "grade", "annual_inc", "loan_status", "purpose", "emp_length", "delinq_2yrs", "mths_since_last_delinq", "mths_since_last_major_derog"]
ficoscores = pd.read_csv("Loans2007-2018.csv", usecols = col_names, low_memory = True, dtype = {"issue_d":"str", "fico_range_low": "float64", 
"fico_range_high":"float64", "loan_amnt":"float64", "grade":"str", "annual_inc": "float64", "loan_status":"str", "purpose":"str", "emp_length":"str", "delinq_2yrs":"float64", "mths_since_last_delinq":"float64", "mths_since_last_major_derog":"float64"})

specific_grades = ["A", "B", "C", "D", "E", "F", "G"]
loanstatus = ["Current", "Fully Paid", "Charged Off"]
purpose = ["credit_card", "debt_consolidation"]
cleaned_data = ficoscores[ficoscores["grade"].isin(specific_grades)]
cleaned_data2 = cleaned_data[cleaned_data["loan_status"].isin(loanstatus)]
cleaned_data3 = cleaned_data2[cleaned_data2["purpose"].isin(purpose)]

cleaned_data3 = cleaned_data3.copy()
cleaned_data3["FICO"] = cleaned_data3[["fico_range_low", "fico_range_high"]].median(axis=1)
cleaned_data3["issue_d"] = pd.to_datetime(cleaned_data3["issue_d"], format = '%b-%Y')
cleaned_data3.loc[:, "Qrt"] = cleaned_data3["issue_d"].dt.to_period("Q")


qrt_avg = cleaned_data3.groupby("Qrt")["FICO"].mean().reset_index()
long_term_avg = qrt_avg["FICO"].mean()

sub = long_term_avg
qrt_avg["+/-"] = qrt_avg["FICO"] - sub


        
qrt_avg["Qrt. change"] = qrt_avg["+/-"].diff()


#rewards good loan grades
def label(val):
    if val in ["A", "B", "C", "D"]:
        return 1
    else:
        return 0
    

#rewards good standing (past history) -- Trust
def label2(val2):
    if val2/12.0 >=2:
        return 1
    elif val2/12.0 >=1:
        return 0.5
    else:
        return 0 

#rewards if loan status paid -- standing 
def label3(val3):
    if val3 == "Fully Paid":
        return 1
    else:
        return 0
    
#rewards if employment is long
def label4(val4):
    if val4 in ["10+ years", "9 years", "8 years"]:
        return 1
    elif val4 in ["7 years", "6 years"]:
        return 0.75
    elif val4 in ["5 years", "4 years"]:
        return 0.5
    elif val4 in ["3 years", "2 years"]:
        return 0.25
    else:
        return 0
    

#making employment numeric 
def label5(val5):
    if val5 == "10+ years":
        return 10
    elif val5 == "9 years":
        return 9
    elif val5 == "8 years":
        return 8
    
    elif val5 == "7 years":
        return 7

    elif val5 == "6 years":
        return 6

    elif val5 == "5 years":
        return 5
    
    elif val5 == "4 years":
        return 4
    
    elif val5 == "3 years":
        return 3
    
    elif val5 == "2 years":
        return 2
    
    else:
        return 1
    
def label_LGBM(val6):
    if val6 == "A":
        return 0
    if val6 == "B":
        return 1
    if val6 == "C":
        return 2
    if val6 == "D":
        return 3
    if val6 == "E":
        return 4
    if val6 == "F":
        return 5
    if val6 == "G":
        return 6

    
    

cleaned_data3_LGBM = cleaned_data3.copy()
cleaned_data3_LGBM["Grade"] = cleaned_data3_LGBM["grade"].apply(label_LGBM)
cleaned_data3["Prepare"] = cleaned_data3["grade"].apply(label)
cleaned_data3.loc[:, "Standing"] = cleaned_data3["loan_status"].apply(label3)
cleaned_data3v2 = cleaned_data3.copy()
cleaned_data3v2["emp"] = cleaned_data3v2["emp_length"].apply(label5)
cleaned_data3.loc[:, "Trust"] = cleaned_data3["mths_since_last_delinq"].apply(label2)
cleaned_data3.loc[:, "Work"] = cleaned_data3["emp_length"].apply(label4)
