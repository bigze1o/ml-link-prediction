import PreProcess
import AdjacencyMatrix
import ScoreCN
import ScoreJC
import ScoreAA
import MachineLearning
import AUC

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import svm

PathFile = u'Dataset/USAir.csv'
Data_No_Weight = PreProcess.ReadDataNoWeight(PathFile)
Data_Adjacency = AdjacencyMatrix.Matrix_Link_Undirect(Data_No_Weight)
#AAScore = ScoreCN.getMatrix(Data_Adjacency)
# test, prediction = MachineLearning.Train_Test_Split(CNScore)
# print(AUC.getPrecision(test, prediction))

# JCScore = ScoreJC.getMatrix(Data_Adjacency)
# test, prediction = MachineLearning.Train_Test_Split(JCScore)
# print(AUC.getPrecision(test, prediction))

AAScore = ScoreAA.getMatrixHalf(Data_Adjacency)
#AAScore = np.unique(AAScore,axis =0)
#print(AAScore[332:345, :])
#test, prediction = MachineLearning.Train_Test_Split(AAScore)
# print(len(AAScore))
#print(AUC.getPrecision(test, prediction))

X_train, X_test, y_train, y_test = train_test_split(AAScore[:,:-1].reshape(-1,1), AAScore[:,-1], test_size=0.2, random_state=0)
# svm_te = svm.SVC(kernel='rbf', C=1, gamma = 0.001,random_state=0)
# svm_te.fit(X_train, y_train)
# #svm_te.fit(X_train, y_train)
# y_Prediction = svm_te.predict(X_test)
# print (AUC.getPrecision(y_test, y_Prediction))

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1]}]

#scores = ['precision', 'recall']
scores = ['precision']

for score in scores:

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()