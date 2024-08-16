import randomforest as rf
import analyze as a
import loans as l
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics as m
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sb
import lightgbm as lgb
#rewards good (past history) -- Trust revised as 0.0 likely means never, rather than 0 months since delinq;
def label2(val2):
    if val2/12.0 >=2:
        return 1
    elif val2 == 0.0:
        return 1
    elif pd.isna(val2):
        return 1
    elif val2/12.0 >=1:
        return 0.5
    else:
        return 0 


x = l.cleaned_data3_LGBM.drop([ "issue_d", "purpose", "fico_range_low", "fico_range_high", "loan_status", "emp_length", "delinq_2yrs","mths_since_last_major_derog"], axis = 1)



x2_join1 = a.final
x2_join2 = x

x2_join1["Key"] = x2_join1["Qrt"].astype(str)
x2_join2["Key"] = x2_join2["Qrt"].astype(str)

joined = pd.merge(x2_join1, x2_join2, on ="Key", how = "inner")
joined = joined.drop(["Condition", "index", "+/-", "Qrt. change", "Key", "Qrt_y", "Qrt_x"], axis =1)
joined["Trust"]= joined["mths_since_last_delinq"].apply(label2)
joined_final = joined.drop(["mths_since_last_delinq", "grade"], axis =1)
y = joined["grade"]

grade_class = y.value_counts()
#plt.pie(grade_class.values, labels=grade_class.index.values, autopct = "%1.1f%%")
#plt.title("Grades Dist.")
#plt.show()

x_formodel = joined_final.drop(["Grade"],axis=1)
y_formodel = y.map({"A": 0, "B": 0, "C": 0, "D": 0, "E": 1, "F": 1, "G": 1})


#training/setting model up

train_x, test_x, train_y, test_y = train_test_split(x_formodel,y_formodel, train_size = 0.7, test_size = 0.3, random_state = 2000)
s= StandardScaler()
s.fit(train_x)
train_x = s.transform(train_x)
test_x = s.transform(test_x)

parameters = {"objective": "multiclass", "num_class": 2, "metric": "multi_logloss",
    "verbose": 0}

training = lgb.Dataset(train_x, label = train_y)
testing = lgb.Dataset(test_x, label = test_y, reference = training)
LGBM = lgb.train(parameters, training, num_boost_round = 100, valid_sets = [testing])


prediction_y = LGBM.predict(test_x, num_iteration=LGBM.best_iteration)
prediction_y.shape
prediction_y = np.argmax(prediction_y, axis = 1)


acc = m.accuracy_score(test_y, prediction_y)
class_report = m.classification_report(test_y, prediction_y)
matrix_c = m.confusion_matrix(test_y, prediction_y)


#m.ConfusionMatrixDisplay(confusion_matrix=matrix_c).plot()
#plt.show()

#print(class_report)

TrueP = matrix_c[1,1]  
TrueN = matrix_c[0,0] 
FalseP = matrix_c[0,1] 
FalseN = matrix_c[1,0]

SN = TrueP/(TrueP+FalseN)
SP = TrueN/(TrueN+FalseP)
PR = TrueP/(TrueP + FalseP) #Precision
Recall = TrueP/(TrueP + FalseN)


print(f'"Sensitvity, Specificity: "{SN}, {SP}')