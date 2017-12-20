import numpy as np

def getMaxNode(Data):
    maxNode = 0
    for i in range(len(Data)):
        maxNode = max(maxNode, Data[i, 0], Data[i, 1])
    return maxNode
    
def Matrix_Link_Undirect(Data):
    maxNode = getMaxNode(Data)
    MatrixData = np.zeros((maxNode, maxNode))
    for i in range(len(Data)):
        SRC = Data[i, 0]
        TGT = Data[i, 1]
        MatrixData[SRC - 1, TGT - 1] = 1
        MatrixData[TGT - 1, SRC - 1] = 1
    return MatrixData

def Matrix_Link(Data):
    maxNode = getMaxNode(Data)
    MatrixData = np.zeros((maxNode, maxNode))
    for i in range(len(Data)):
        SRC = Data[i, 0]
        TGT = Data[i, 1]
        MatrixData[SRC - 1, TGT - 1] = 1
    return MatrixData

def Matrix_Weight_Undirect(Data):
    maxNode = getMaxNode(Data)
    MatrixData = np.zeros((maxNode, maxNode))
    for i in range(len(Data)):
        SRC = Data[i, 0]
        TGT = Data[i, 1]
        MatrixData[SRC - 1, TGT - 1] = Data[i, 2]
        MatrixData[TGT - 1, SRC - 1] = Data[i, 2]
    return MatrixData

def Matrix_Weight(Data):
    maxNode = getMaxNode(Data)
    MatrixData = np.zeros((maxNode, maxNode))
    for i in range(len(Data)):
        SRC = Data[i, 0]
        TGT = Data[i, 1]
        MatrixData[SRC - 1, TGT - 1] = Data[i, 2]
    return MatrixData
