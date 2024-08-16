import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics as m
from sklearn.ensemble import RandomForestClassifier
import analyze as a
import loans as l
from scipy.stats import randint
import matplotlib.pyplot as plt


x = l.cleaned_data3v2.drop(["Prepare", "grade", "issue_d", "purpose", "fico_range_low", "fico_range_high", "loan_status", "emp_length", "delinq_2yrs","mths_since_last_major_derog"], axis = 1)



x2_join1 = a.final
x2_join2 = x

x2_join1["Key"] = x2_join1["Qrt"].astype(str)
x2_join2["Key"] = x2_join2["Qrt"].astype(str)

joined = pd.merge(x2_join1, x2_join2, on ="Key", how = "inner")
joined = joined.drop(["Condition", "index", "+/-", "Qrt. change", "Key", "Qrt_y", "Qrt_x"], axis =1)
x = joined
y = l.cleaned_data3v2["Prepare"].drop([0], axis = 0)



#training/setting model up
train_x, test_x, train_y, test_y = train_test_split(x,y, train_size = 0.7, test_size = 0.3)

"""
random_forest = RandomForestClassifier()
random_forest.fit(train_x, train_y)


#use test now
prediction_y = random_forest.predict(test_x)
acc = m.accuracy_score(test_y, prediction_y)  #91.48% accuracy first


parameters = {"n_estimators": randint(100, 250), "max_depth": randint(10,20)}

random_p = RandomizedSearchCV(random_forest, param_distributions = parameters, n_iter = 5, cv = 5) 
random_p.fit(train_x, train_y)

optimal = random_p.best_estimator_
optimal_fix = random_p.best_params_

#optimal parameters are max_depth:14 & n_estimators: 182




best_random_forest = RandomForestClassifier(n_estimators = 182, max_depth = 14, class_weight= "balanced")
best_random_forest.fit(train_x, train_y)

#use test again

prediction_y2 = best_random_forest.predict(test_x)
acc2 = m.accuracy_score(test_y, prediction_y2) #92.18% accuracy

matrix_c = m.confusion_matrix(test_y, prediction_y2)


m.ConfusionMatrixDisplay(confusion_matrix=matrix_c).plot()
plt.show()

TrueP = matrix_c[1,1]  
TrueN = matrix_c[0,0] 
FalseP = matrix_c[0,1] 
FalseN = matrix_c[1,0]

SN = TrueP/(TrueP+FalseN)
SP = TrueN/(TrueN+FalseP)
PR = TrueP/(TrueP + FalseP) #Precision
Recall = TrueP/(TrueP + FalseN)


print(f'"Confusion Matrix [TN, FP] [FN, TP]: " {matrix_c}')
print(f'"Model Accuracy: " {acc2}')
print(f'"Sensitvity, Specificity, Precision, Recall: " {SN}, {SP}, {PR}, {Recall}')

#hid so LGBM file would not be slowed down
"""