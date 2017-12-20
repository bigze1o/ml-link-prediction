import numpy as np

def getPrecision(yTest, yPrediction):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(yTest)):
        if yTest[i]:
            if yPrediction[i]:
                TP += 1
            else:
                FP += 1
        else:
            if yPrediction[i]:
                FN += 1
            else:
                TN += 1
    #return ((TP+TN)*1.0/(TP+FP+TN+FN))
    #print(TP, FP, TN, FN)
    return (TP*1.0 / (TP+FP))

    