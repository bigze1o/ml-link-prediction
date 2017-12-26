import numpy as np
            
def getMatrixFull(matrixAdj):
    maxNode = len(matrixAdj)
    matrixScore = np.zeros((maxNode * maxNode, 2))
    for i in range(maxNode):
        for j in range(i, maxNode):
            rowSRC = i * maxNode + j
            rowTGT = j * maxNode + i
            tag_class = matrixAdj[i, j]
            matrixScore[rowSRC, 1] = tag_class
            matrixScore[rowTGT, 1] = tag_class
            cnScore = np.dot(matrixAdj[i, :], matrixAdj[j, :])
            matrixScore[rowSRC, 0] = cnScore
            matrixScore[rowTGT, 0] = cnScore
    return matrixScore

def getMatrixHalf(matrixAdj):
    maxNode = len(matrixAdj)
    matrixScore = []
    for i in range(maxNode):
        for j in range(i + 1, maxNode):
            cnScore = np.dot(matrixAdj[i, :], matrixAdj[j, :])
            tag_class = matrixAdj[i, j]
            matrixScore.append([cnScore, tag_class])
    return np.array(matrixScore)
