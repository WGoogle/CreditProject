import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import analyze as a
import loans as l
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import statsmodels.api as stats
from sklearn import metrics as m
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as prc
# factors in annual_inc, FICO, mnths since last delinq, status of loan, 

def sigmoid(b):
    return 1 / (1+ np.exp(-b))

x = l.cleaned_data3.drop(["emp_length", "loan_amnt", "Prepare", "grade", "issue_d", "mths_since_last_major_derog", "Qrt", "purpose", "fico_range_low", "fico_range_high", "loan_status", "mths_since_last_delinq", "delinq_2yrs"], axis = 1)
y = l.cleaned_data3["Prepare"]

#training/setting model up
train_x, test_x, train_y, test_y = train_test_split(x,y, train_size = 0.7, test_size = 0.3, random_state = 100)
s = MinMaxScaler()
train_x[["annual_inc", "FICO"]] = s.fit_transform(train_x[["annual_inc", "FICO"]])

lg = LogisticRegression()
rfe = RFE(estimator=lg, n_features_to_select=5)
rfe = rfe.fit(train_x, train_y)
selected_columns = train_x.columns[rfe.support_]
c = train_x[selected_columns]

train_x_stats = stats.add_constant(c)

log = stats.GLM(train_y, train_x_stats, family = stats.families.Binomial())
r = log.fit()
summary = r.summary()



train_y_predicted = r.predict(stats.add_constant(train_x))
train_y_predicted2 = pd.DataFrame({"Prepare":train_y.values, "Prepare_Prob":train_y_predicted})
train_y_predicted2["Final_Pred"] = train_y_predicted2.Prepare_Prob.map(lambda ax: 0 if ax<0.5 else 1)

matrix_c = m.confusion_matrix(train_y_predicted2["Prepare"], train_y_predicted2["Final_Pred"])
Matrix_c_acc = m.accuracy_score(train_y_predicted2.Prepare, train_y_predicted2["Final_Pred"])

TrueP = matrix_c[1,1]  
TrueN = matrix_c[0,0] 
FalseP = matrix_c[0,1] 
FalseN = matrix_c[1,0]

SN = TrueP/(TrueP+FalseN)
SP = TrueN/(TrueN+FalseP)
PR = TrueP/(TrueP + FalseP) #Precision
Recall = TrueP/(TrueP + FalseN)



def draw_roc(ac, probs):
    fpr, tpr, th = m.roc_curve(ac, probs, drop_intermediate = False)
    auc_score = m.roc_auc_score(ac, probs)
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


dpr,tpr, th = m.roc_curve(train_y_predicted2["Prepare"], train_y_predicted2["Prepare_Prob"], drop_intermediate = False)

#draw_roc(train_y_predicted2["Prepare"], train_y_predicted2["Prepare_Prob"])


n = [float(x)/10 for x in range(10)]
for i in n:
    train_y_predicted2[i]= train_y_predicted2["Prepare_Prob"].map(lambda x: 1 if x > i else 0)


optimal = pd.DataFrame(columns = ["Prob", "Acc", "SN", "SP"])

new_n = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in new_n:
    matrix_c2 = m.confusion_matrix(train_y_predicted2["Prepare"], train_y_predicted2[i])
    total_first=sum(sum(matrix_c2))
    accuracy = (matrix_c2[0,0]+ matrix_c2[1,1])/ total_first
    SN = matrix_c2[0,0]/(matrix_c2[0,0]+ matrix_c2[0,1])
    SP = matrix_c2[1,1]/(matrix_c2[1,0]+matrix_c2[1,1])
    optimal.loc[i] =[i,accuracy,SN,SP]


#optimal.plot.line(x = "Prob", y=["Acc", "SN", "SP"])
#plt.show()


train_y_predicted2["Final_Pred2"] = train_y_predicted2.Prepare_Prob.map(lambda ax: 0 if ax<0.7514 else 1)
matrix_c3 = m.confusion_matrix(train_y_predicted2["Prepare"], train_y_predicted2["Final_Pred2"])
Matrix_c_acc3 = m.accuracy_score(train_y_predicted2.Prepare, train_y_predicted2["Final_Pred2"])

TrueP2 = matrix_c3[1,1]  
TrueN2 = matrix_c3[0,0] 
FalseP2 = matrix_c3[0,1] 
FalseN2 = matrix_c3[1,0]

SN2 = TrueP2/(TrueP2+FalseN2)
SP2 = TrueN2/(TrueN2+FalseP2)
PR2 = TrueP2/(TrueP2 + FalseP2) #Precision
Recall2 = TrueP2/(TrueP2 + FalseN2)

p, ri, th = prc(train_y_predicted2["Prepare"], train_y_predicted2["Prepare_Prob"])

#plt.plot(th, p[:-1], "g-")
#plt.plot(th, ri[:-1], "r-")
#plt.show()



train_y_predicted2["Final_Pred3"] = train_y_predicted2.Prepare_Prob.map(lambda ax: 0 if ax<0.6827 else 1)
matrix_c4 = m.confusion_matrix(train_y_predicted2["Prepare"], train_y_predicted2["Final_Pred3"])
Matrix_c_acc4 = m.accuracy_score(train_y_predicted2.Prepare, train_y_predicted2["Final_Pred3"])

TrueP3 = matrix_c4[1,1]  
TrueN3 = matrix_c4[0,0] 
FalseP3 = matrix_c4[0,1] 
FalseN3 = matrix_c4[1,0]

SN3 = TrueP3/(TrueP3+FalseN3)
SP3 = TrueN3/(TrueN3+FalseP3)
PR3 = TrueP3/(TrueP3 + FalseP3) #Precision
Recall3 = TrueP3/(TrueP3 + FalseN3)

#matrix_c4 is our final confusion matrix

#NOW time to test our model using test dataset
test_x[["annual_inc", "FICO"]] = s.fit_transform(test_x[["annual_inc", "FICO"]])

lg2 = LogisticRegression()
rfe2 = RFE(estimator=lg2, n_features_to_select=5)
rfe2 = rfe2.fit(test_x, test_y)
selected_columns2 = test_x.columns[rfe2.support_]
c2 = test_x[selected_columns2]

test_x_stats = stats.add_constant(c2)
log2 = stats.GLM(test_y, test_x_stats, family = stats.families.Binomial())
r2 = log.fit()

test_y_prediction = r2.predict(stats.add_constant(test_x))
test_y_prediction2 = pd.DataFrame(test_y_prediction)
test_y_final = pd.DataFrame(test_y)
test_y_prediction2.reset_index(drop=True, inplace=True)
test_y_final.reset_index(drop=True, inplace=True)
test_y_final2 = pd.concat([test_y_final, test_y_prediction2],axis=1)
test_y_final2 = test_y_final2.rename(columns = {0:"Probs"})
test_y_final2["Final_Pred"] = test_y_final2["Probs"].map(lambda x: 0 if x < 0.686 else 1)


#Final Test stats

matrix_test = m.confusion_matrix(test_y_final2["Prepare"], test_y_final2["Final_Pred"])
Matrix_test_acc = m.accuracy_score(test_y_final2.Prepare, test_y_final2["Final_Pred"])

TrueP4 = matrix_test[1,1]  
TrueN4 = matrix_test[0,0] 
FalseP4 = matrix_test[0,1] 
FalseN4 = matrix_test[1,0]

SN4 = TrueP4/(TrueP4+FalseN4)
SP4 = TrueN4/(TrueN4+FalseP4)
PR4 = TrueP4/(TrueP4 + FalseP4) #Precision
Recall4 = TrueP4/(TrueP4 + FalseN4)


print(f'"Confusion Matrix [TN, FP] [FN, TP]: " {matrix_test}')
print(f'"Model Accuracy: " {Matrix_test_acc}')
print(f'"Sensitvity, Specificity, Precision, Recall: {SN4}, {SP4}, {PR4}, {Recall4}')
