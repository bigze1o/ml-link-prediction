import numpy as np

def getMatrix(matrixAdj):
    maxNode = len(matrixAdj)
    matrixScore = np.zeros((maxNode * maxNode, 2))
    for i in range(maxNode):
        for j in range(i, maxNode):
            cnScore = np.dot(matrixAdj[i, :], matrixAdj[j, :])
            rowSRC = i * maxNode + j
            rowTGT = j * maxNode + i
            tag_class = matrixAdj[i, j]
            matrixScore[rowSRC, 0] = cnScore
            matrixScore[rowSRC, 1] = tag_class
            matrixScore[rowTGT, 0] = cnScore
            matrixScore[rowTGT, 1] = tag_class
    return matrixScore
            