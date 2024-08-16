import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics as m
import analyze as a
import loans as l
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 

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
        return 1
    elif val2 == 0.0:
        return 1
    elif pd.isna(val2):
        return 1
    elif val2/12.0 >=1:
        return 0.5
    else:
        return 0 


x = l.cleaned_data3_LGBM.drop([ "issue_d", "purpose", "fico_range_low", "fico_range_high", "loan_status", "delinq_2yrs","mths_since_last_major_derog"], axis = 1)



x2_join1 = a.final
x2_join2 = x

x2_join1["Key"] = x2_join1["Qrt"].astype(str)
x2_join2["Key"] = x2_join2["Qrt"].astype(str)

joined = pd.merge(x2_join1, x2_join2, on ="Key", how = "inner")
joined = joined.drop(["Condition", "index", "+/-", "Qrt. change", "Key", "Qrt_y", "Qrt_x"], axis =1)
joined["Trust"]= joined["mths_since_last_delinq"].apply(label2)
joined["Emp"] = joined["emp_length"].apply(label5)
joined_final = joined.drop(["mths_since_last_delinq", "grade", "emp_length"], axis =1)
y = joined["grade"]


x_formodel = joined_final.drop(["Grade"],axis=1)
y_formodel = y.map({"A": 0, "B": 0, "C": 0, "D": 0, "E": 1, "F": 1, "G": 1})


#training/setting model up; can create the model mathemtically using NumPy (have experience/showcased in another project of mine, but using a library saves time, and prone to less errors)

train_x, test_x, train_y, test_y = train_test_split(x_formodel,y_formodel, train_size = 0.7, test_size = 0.3, random_state = 200)
s= StandardScaler()
s.fit(train_x)
train_x = s.transform(train_x)
test_x = s.transform(test_x)

#attempting to optimize K neightbors num
"""
#first time
accuracy = {}
for K in range (5,100,5):
    KNN = KNeighborsClassifier(n_neighbors = K)
    KNN.fit(train_x, train_y)
    prediction_y = KNN.predict(test_x)
    accuracy[K] = m.accuracy_score(test_y, prediction_y)
#second time
accuracy = {}
for K in range (30,250,20):
    KNN = KNeighborsClassifier(n_neighbors = K)
    KNN.fit(train_x, train_y)
    prediction_y = KNN.predict(test_x)
    accuracy[K] = m.accuracy_score(test_y, prediction_y)
"""
#plt.plot(range(30,250,20), accuracy.values())
#plt.ylabel("ACC")
#plt.xlabel("K-Value")
#plt.show()


KNN = KNeighborsClassifier(n_neighbors = 50)
KNN.fit(train_x, train_y)
prediction_y = KNN.predict(test_x)

acc = m.accuracy_score(test_y, prediction_y)
report = m.classification_report(test_y, prediction_y)
matrix_c = m.confusion_matrix(test_y, prediction_y)


m.ConfusionMatrixDisplay(confusion_matrix=matrix_c).plot()
plt.show()

print(f'"Model Accuracy: " {acc}')
print(report)
