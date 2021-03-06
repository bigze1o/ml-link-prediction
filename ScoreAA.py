import numpy as np
from math import log


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
            aaScore = 0.0
            for k in range(maxNode):
                if matrixAdj[i, k] == 1 and matrixAdj[i, k] == matrixAdj[j, k]:
                    res = np.sum(matrixAdj[k, :])
                    if log(res) != 0.0:
                        aaScore += 1.0/log(res)
            matrixScore[rowSRC, 0] = aaScore
            matrixScore[rowTGT, 0] = aaScore
    return matrixScore

def getMatrixHalf(matrixAdj):
    maxNode = len(matrixAdj)
    matrixScore = []
    for i in range(maxNode):
        for j in range(i + 1, maxNode):
            aaScore = 0.0
            for k in range(maxNode):
                if matrixAdj[i, k] == 1 and matrixAdj[i, k] == matrixAdj[j, k]:
                    res = np.sum(matrixAdj[k, :])
                    if log(res) != 0.0:
                        aaScore += 1.0/log(res)
            tag_class = matrixAdj[i, j]
            matrixScore.append([aaScore, tag_class])
    return np.array(matrixScore)


            