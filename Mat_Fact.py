import sys
import numpy as np
from copy import deepcopy

class Matrix_Fact():

    def __init__(self, Rate, D, alpha, beta, lmda, epochs, err):
        self.Rate = Rate
        self.users, self.events = Rate.shape
        self.D = D
        self.alpha = alpha
        self.beta = beta
        self.lmda = lmda
        self.epochs = epochs
        self.err = err

    
    def train(self):
        self.user_val = np.random.rand(self.users, self.D)/self.D
        self.event_val = np.random.rand(self.events, self.D)/self.D
        self.bs_user = np.zeros(self.users, dtype='uint8')
        self.bs_event = np.zeros(self.events, dtype='uint8')
        self.bias = np.uint8(np.mean(self.Rate[np.where(self.Rate != 0)]))

        self.sample = [
            (a, b, self.Rate[a, b])
            for a in range(self.users)
            for b in range(self.events)
            if self.Rate[a, b] > 0
        ]

        Mse = sys.float_info.max
        for a in range(self.epochs):
            P_mat = deepcopy(self.user_val)
            Q_mat = deepcopy(self.event_val)
            b_u_mat = deepcopy(self.bs_user)
            b_e_mat = deepcopy(self.bs_event)
            bias_mat = deepcopy(self.bias)
            np.random.shuffle(self.sample)
            self.SGD()
            mse = self.MSE()

            if Mse < mse:
                self.user_val = deepcopy(P_mat)
                self.event_val = deepcopy(Q_mat)
                self.bs_user = deepcopy(b_u_mat)
                self.bs_event = deepcopy(b_e_mat)
                self.bias = deepcopy(bias_mat)

                self.alpha *= self.lmda
            
            else:
                Mse = mse
            
            print("Epoch: %d,\t error=%.10f,\t alpha=%.10f" %(a+1, mse, self.alpha))
            if mse < self.err:
                break;
        
    def SGD(self):
        for a, b, c in self.sample:
            pred = self.rating(a, b)
            error = (c - pred)

            self.bs_user[a] += self.alpha * (error - self.beta * self.bs_user[a])
            self.bs_event[a] += self.alpha * (error - self.beta * self.bs_event[b])

            newP = self.user_val[a, :][:]

            self.user_val[a, :] += self.alpha * (error * self.event_val[b, :] - self.beta * self.user_val[a, :])
            self.event_val[b, :] += self.alpha * (error * newP - self.beta * self.event_val[b, :])

        
    def MSE(self):
        xr, yr = self.Rate.nonzero()
        pred = self.get_mat()
        error = 0
        for a, b in zip(xr, yr):
            error += pow(self.Rate[a, b] - pred[a, b], 2)
        return error/len(xr)

        

    def rating(self, a, b):
        pred = self.bias + self.bs_user[a] + self.bs_event[b] + self.user_val[a, :].dot(self.event_val[b, :].T)
        return pred

    def get_mat(self):
        return self.bias + self.bs_user[:, np.newaxis] + self.bs_event[np.newaxis:, ] + self.user_val.dot(self.event_val.T)


            

        