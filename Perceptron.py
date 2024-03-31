import numpy as np
def activation_func(x):
    return np.where(x>0,1,0)
class Perceptron:
    def __init__(self,learning_rate = 0.01,n_iters = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
    def fit(self,X,y):
        n_sample,n_feature = X.shape
        self.weight=np.zeros(n_feature)
        self.bias = 0
        y_pred  = np.where(y>0,1,0)
        for _ in range(self.n_iters):
            for idx,x_i in enumerate(X):
                linear_output = np.dot(x_i,self.weight)+self.bias
                y_prediction = activation_func(linear_output)
                update = self.lr*(y_pred[idx]-y_prediction)
                self.weight+=update*x_i
                self.bias+=update
    def predict(self,X):
        linear_output = np.dot(X,self.weight)+self.bias
        y_p = activation_func(linear_output)
        return y_p
