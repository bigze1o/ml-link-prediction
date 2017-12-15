import Initialize

TrainFile = u'Train.csv'
TestFile = u'Test.csv'

Train_Data, Test_Data = Initialize.Init(TrainFile, TestFile)
Train_Data = Initialize.CommanNeighbor(Train_Data)
print(Train_Data)

