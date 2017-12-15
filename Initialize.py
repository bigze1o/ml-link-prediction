import pandas as pd

def DataShape(Data):
    #ListData contain SRT to TGT : Yes or No link
    List_Data = []

    #MaxNode of Graph
    MaxNode = 330

    #Create the empty Matrix[MaxNode * MaxNode][3] : [][0]: SRC, [][1]: TGT, [][2]: Link
    for i in range(1, MaxNode+1):
        for j in range(1, MaxNode+1):
            List_Data.append([i, j, 0, 0])

    #Check the Link on Matrix List_Data
    for i in range(len(Data)):
        col1 = Data.iat[i, 0]
        col2 = Data.iat[i, 1]
        rowOnList = (col1 - 1) * MaxNode + (col2 - 1)
        List_Data[rowOnList][3] = 1
    return List_Data

def Init(Train_File, Test_File):
    #Read file Trainning and Test File
    TrainData = pd.read_csv(Train_File)
    TestData = pd.read_csv(Test_File)
    #Matrix of train and test
    Matrix_Train = DataShape(TrainData)
    Matrix_Test = DataShape(TestData)
    return Matrix_Train, Matrix_Test
    
def CommanNeighbor(Matrix_Link):
    MaxNode = 330
    for i in range(1,MaxNode + 1):
        for j in range(1, MaxNode + 1):
            if i == j:
                continue
            cn = 0
            row_src_start = (i - 1) * MaxNode
            row_tgt_start = (j - 1) * MaxNode
            for k in range(1, MaxNode + 1):
                if Matrix_Link[row_src_start + k - 1][3] == 1 and Matrix_Link[row_tgt_start + k - 1][3] == 1:
                    cn += 1
            Matrix_Link[(i-1) * MaxNode + j - 1][2] = cn
            Matrix_Link[(j-1) * MaxNode + i - 1][2] = cn        
    return Matrix_Link

