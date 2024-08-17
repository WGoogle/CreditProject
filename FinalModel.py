import pandas as pd
import analyze as a
import loans as l
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics as m
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb
import lightgbm as lgb

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
    
def label2(val2):
    if val2/12.0 >=2:
        return max
    elif val2 == 0.0:
        return max
    elif pd.isna(val2):
        return max
    elif val2/12.0 >=1:
        return mean
    else:
        return 0 


x = l.cleaned_data3_LGBM.drop(["loan_amnt", "issue_d", "purpose", "fico_range_low", "fico_range_high", "loan_status", "delinq_2yrs","mths_since_last_major_derog"], axis = 1)



x2_join1 = a.final
x2_join2 = x

x2_join1["Key"] = x2_join1["Qrt"].astype(str)
x2_join2["Key"] = x2_join2["Qrt"].astype(str)

joined = pd.merge(x2_join1, x2_join2, on ="Key", how = "inner")
joined = joined.drop(["Condition", "index", "+/-", "Qrt. change", "Key", "Qrt_y", "Qrt_x"], axis =1)
max, mean = joined["mths_since_last_delinq"].max(), joined["mths_since_last_delinq"].mean()
joined["Trust"]= joined["mths_since_last_delinq"].apply(label2)
joined["Emp"] = joined["emp_length"].apply(label5)


#setting equal amounts of data
specific_grades_good = ["A", "B", "C", "D"]
specific_grades_bad =  ["E", "F", "G"]
joined3 = joined[joined["grade"].isin(specific_grades_good)]
joined2 = joined[joined["grade"].isin(specific_grades_bad)]


# 1628819 rows of good
# 138060 rows of bad
num_drop = 1490759
drop = joined3.sample(n=num_drop, random_state=2000).index
joined_good = joined3.drop(drop)
joined_bad = joined2


joined_final = pd.concat([joined_good, joined_bad], axis=0, ignore_index=True)
joined_final2 = joined_final.drop(["mths_since_last_delinq", "grade", "emp_length"], axis =1)
y = joined_final["grade"]

x_formodel = joined_final2.drop(["Grade"],axis=1)
y_formodel = y.map({"A": 1, "B": 1, "C": 1, "D": 1, "E": 0, "F": 0, "G": 0})



#training/setting model up

train_x, test_x, train_y, test_y = train_test_split(x_formodel,y_formodel, train_size = 0.7, test_size = 0.3, random_state = 2000)
s= MinMaxScaler()
train_x[["Emp", "FICO", "annual_inc", "Indicator", "Trust"]] = s.fit_transform(train_x[["Emp", "FICO", "annual_inc", "Indicator", "Trust"]])
test_x[["Emp", "FICO", "annual_inc", "Indicator", "Trust"]] = s.fit_transform(test_x[["Emp", "FICO", "annual_inc", "Indicator", "Trust"]])

parameters = {"objective": "multiclass", "num_class": 2, "metric": "multi_logloss",
    "verbosity": 0}
training = lgb.Dataset(train_x, label = train_y)
testing = lgb.Dataset(test_x, label = test_y, reference = training)
LGBM = lgb.train(parameters, training, num_boost_round = 100, valid_sets = [testing])


prediction_y = LGBM.predict(test_x, num_iteration=LGBM.best_iteration)
prediction_y.shape
prediction_y = np.argmax(prediction_y, axis = 1)

acc = m.accuracy_score(test_y, prediction_y)
class_report = m.classification_report(test_y, prediction_y)
matrix_c = m.confusion_matrix(test_y, prediction_y)


"""
plt.figure(figsize=(6, 5))
sb.heatmap(matrix_c, annot=True, fmt="d", cmap="Reds", cbar=True, 
           xticklabels=["Predicted Bad Borrower (0)', 'Predicted Good Borrower (1)"],
           yticklabels=["True Bad Borrower (0)', 'True Good Borrower (1)"])
plt.title("Confusion Matrix For LightGBM ML Model")
plt.xlabel("Predicted Classification")
plt.ylabel("True Classification")
plt.show()


print(f'"Model Accuracy: " {acc}')
print(class_report)

"""

def user_turn():
    emp_length = input("What is the employment length (in the following format, e.g., 10+ years, 4 years): ")
    FICO = float(input("What is the FICO score: "))
    annual_income = float(input("What is the annual income: "))
    indicator = float(input("What is the economic indicator -- Composite Consumer Confidence Amplitude Adjusted -- currently (historically between 96-103): "))
    trust = float(input("How many months since last delinquency: "))


   
    emp = label5(emp_length)
    trust = label2(trust)

    user = pd.DataFrame({"EMP":[emp], "Trust": [trust], "FICO": [FICO], "Annual_Inc": [annual_income], "Indicator":[indicator]})
    user[["EMP", "FICO", "Annual_Inc", "Indicator", "Trust"]] = s.fit_transform(user[["EMP", "FICO", "Annual_Inc", "Indicator", "Trust"]])
    user_prediction = LGBM.predict(user, num_iteration=LGBM.best_iteration)
    predicted_class = np.argmax(user_prediction, axis=1)

    if predicted_class[0] == 1:
        return "The model predicts that the loan is good and should be approved!"

    if predicted_class[0] == 0:
        return "The model predicts that the loan is bad and should not be approved!"
    
print(user_turn())
