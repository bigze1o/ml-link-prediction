import numpy as np

def getMatrixHalf(matrixAdj):
    maxNode = len(matrixAdj)
    matrixScore = []
    for i in range(maxNode):
        for j in range(i + 1, maxNode):
            cnScore = np.dot(matrixAdj[i, :], matrixAdj[j, :])
            jcTemp = 0
            for k in range(maxNode):
                if matrixAdj[i, k] == 1 or matrixAdj[j, k] == 1:
                    jcTemp += 1
            tag_class = matrixAdj[i, j]
            if jcTemp != 0:
                matrixScore.append([cnScore * 1.0 / jcTemp, tag_class])
            else:
                matrixScore.append([0.0, tag_class])
    return np.array(matrixScore)

            