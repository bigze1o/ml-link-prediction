import numpy as np

def getPrecision(yTest, yPrediction):
    TP = 0
    FP = 0
    for i in range(len(yTest)):
        if yTest[i]:
            if yPrediction[i]:
                TP += 1
            else:
                FP += 1
    return (TP*1.0/(TP+FP))

    