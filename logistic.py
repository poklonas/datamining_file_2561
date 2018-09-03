class logistic_model():
    self.iteration
    self.learnRate
    self.weight
    self.feature
    self.target

    def __init__(self, lr, x, y):
        self.learnRate = lr
        self.feature = x
        self.target = y
        self.iteration  = 1000
    
    def sigmoid(self):
        z = np.dot(self.featrue, self.weight)
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
    
    
            