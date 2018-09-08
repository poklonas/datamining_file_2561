import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#D:\WORK\datamine\w5\datamining_file_2561

class BayesModel:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y) 
        self.prior_p = np.array([np.zeros_like(self.x), np.ones_like(self.x)])
        self.var = np.array([])
        self.mean = np.array([])
        for i in range(0 , self.y.max() - self.y.min() + 1):
            self.mean = np.append(  self.mean, np.mean(self.x[self.y == i]))
            self.var = np.append(  self.var, np.var(self.x[self.y == i]))

    def set_parm(self, var, mean):
        self.var = var
        self.mean = mean
    
    def prior_p_finder(self):
        count = 0
        for i in sorted(self.x):
            count_c1 = ((self.y == 1) & (self.x == i)).sum()
            count_x = (self.x == i).sum()
            prob_c1 = count_c1/count_x
            self.prior_p[1][count] = (prob_c1)
            self.prior_p[0][count] = (1 - prob_c1)
            count += 1

    def posterior(self):
        plt.plot(sorted(self.x), self.prior_p[0], '-r')
        plt.plot(sorted(self.x), self.prior_p[1], '-b')
        plt.ylabel('posterior probability')
        plt.xlabel('x')
        plt.show()

    # likelihood 
    def normal_dis_one(self, max, min, step):
        x1 = np.arange(min, max, step)
        x2 = np.arange(min, max, step)
        pr1 = (1/(np.sqrt(2 * np.pi ) * self.var[1])) *  np.exp((-1/2)*(((x1-self.mean[1])/self.var[1])**2) )
        pr0 = (1/(np.sqrt(2 * np.pi ) * self.var[0])) *  np.exp((-1/2)*(((x2-self.mean[0])/self.var[0])**2) )
        plt.plot(x1, pr1, '-r')
        plt.plot(x2, pr0, '-b')
        plt.xlabel('X')
        plt.ylabel('likelyhood')
        plt.show()
        

#######################################################
path = 'D:\WORK\datamine\w5\datamining_file_2561\Iris-2type.csv'
data = pd.read_csv(path, sep=',')
x = data['PetalLengthCm'].values
y = data['target'].values
#######################################################  เซ็ทโดยอินพุต
bayes = BayesModel( x, y)
bayes.normal_dis_one(10, -2, 0.1)
bayes.prior_p_finder()
bayes.posterior()
#######################################################  เซ็ทเอง
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([0,0,0,0,0,1,1,1,1,1])
bayes = BayesModel( x, y)
bayes.set_parm([1, 2], [2,7])
bayes.normal_dis_one(10, -2, 0.1)
bayes.prior_p_finder()
bayes.posterior()
########################################################