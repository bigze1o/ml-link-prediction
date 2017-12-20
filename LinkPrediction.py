import PreProcess
import AdjacencyMatrix
import ScoreCN
import ScoreJC
import ScoreAA
import MachineLearning
import AUC

PathFile = u'Data.csv'
Data_No_Weight = PreProcess.ReadDataNoWeight(PathFile)
Data_Adjacency = AdjacencyMatrix.Matrix_Link_Undirect(Data_No_Weight)
# CNScore = ScoreCN.getMatrix(Data_Adjacency)
# test, prediction = MachineLearning.Train_Test_Split(CNScore)
# print(AUC.getPrecision(test, prediction))

# JCScore = ScoreJC.getMatrix(Data_Adjacency)
# test, prediction = MachineLearning.Train_Test_Split(JCScore)
# print(AUC.getPrecision(test, prediction))

AAScore = ScoreAA.getMatrixHalf(Data_Adjacency)
test, prediction = MachineLearning.Train_Test_Split(AAScore)
# print(len(AAScore))
print(AUC.getPrecision(test, prediction))