import numpy as np 
class Logistic_model():

    def __init__(self, lr, x, y):
        self.learnRate = lr
        self.x = x
        self.target = y
        self.iteration  = 1000
        self.weight = x
    
    def sigmoid(self):
        z = np.dot(self.x, self.weight)
        return 1 / 1 + np.exp(-z)

    def gradian(self):
        h = self.sigmoid()
        return - self.target * np.log(h) - ( 1 - self.target) * np.log(1-h)

    def get_cost(self):
        h = self.sigmoid()
        return np.mean(h-self.target)

    def fit(self):
        for i in range( 0, self.iteration):
            self.weight -= self.learnRate * self.gradian()
    
    
x = Logistic_model(0.5, [1,2,3],[1,1,1])
print(x.fit())
print(x.weight)