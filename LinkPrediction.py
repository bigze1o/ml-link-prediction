import Initialize
from sklearn import svm
import numpy as np


TrainFile = u'Train.csv'
TestFile = u'Test.csv'

Train_Data, Test_Data = Initialize.Init(TrainFile, TestFile)
Train_Data = Initialize.Jaccard(Train_Data)

Input_Train = []
Output_Train = []
for i in range(len(Train_Data)):
    Input_Train.append(Train_Data[i][2])
    Output_Train.append(Train_Data[i][3])

Input_Train = np.array(Input_Train)
Output_Train = np.array(Output_Train)
Input_Train = Input_Train.reshape(-1, 1)

print(Input_Train[0:10])
svm_te = svm.SVC(kernel='rbf', gamma=1, C=1)
svm_te.fit(Input_Train, Output_Train)

print(svm_te.predict([[0.333]]))

