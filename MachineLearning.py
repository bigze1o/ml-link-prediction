from sklearn.model_selection import train_test_split
from sklearn import svm

def Train_Test_Split(MatrixScore):
    Predictor = MatrixScore[:, :-1]
    Target = MatrixScore[:, -1]
    X_Train, X_Test, y_Train, y_Test = train_test_split(Predictor, Target, test_size = 0.2)
    X_Train = X_Train.reshape(-1, 1)

    svm_te = svm.SVC(kernel='rbf', gamma=10, C=1)
    svm_te.fit(X_Train, y_Train)
    y_Prediction = svm_te.predict(X_Test)
    return y_Test, y_Prediction