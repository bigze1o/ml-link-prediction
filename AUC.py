import numpy as np

def getPrecision(yTest, yPrediction):
    TP = 0
    FP = 0
    P = 0
    for i in range(len(yTest)):
        if yTest[i] == 1:
            if yPrediction[i] == 1:
                TP += 1
        else:
            if yPrediction[i] == 1:
                FP += 1
            P = 0
    #print(TP,FP)
    return (TP * 1.0 / (TP+FP))

    