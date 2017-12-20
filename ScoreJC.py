import numpy as np

def getMatrix(matrixAdj):
    maxNode = len(matrixAdj)
    matrixScore = np.zeros((maxNode * maxNode, 2))
    for i in range(maxNode):
        for j in range(i, maxNode):
            cn = np.dot(matrixAdj[i, :], matrixAdj[j, :])
            rowSRC = i * maxNode + j
            rowTGT = j * maxNode + i
            tag_class = matrixAdj[i, j]
            res = 0
            for k in range(maxNode):
                if matrixAdj[i, k] == 1 or matrixAdj[j, k] == 1:
                    res += 1
            jcScore = float(cn/res)
            matrixScore[rowSRC, 0] = jcScore
            matrixScore[rowSRC, 1] = tag_class
            matrixScore[rowTGT, 0] = jcScore
            matrixScore[rowTGT, 1] = tag_class
    return matrixScore
            