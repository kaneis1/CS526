import numpy as np
import q2

train_path='HW1\energydata\energy_train.csv'
val_path='HW1\energydata\energy_val.csv'
test_path='HW1\energydata\energy_test.csv'

def loss(x, y, beta, el, alpha):
    l2_reg = (1 - alpha) * np.sum(beta ** 2)
    l1_reg = alpha * np.sum(np.abs(beta))
    
    ls_loss = 0.5 * np.sum((y - x @ beta) ** 2)
    
    return ls_loss + el * (l1_reg + l2_reg)

def grad_step(x, y, beta, el, alpha, eta):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    gradient = -x.T @ (y - x @ beta)
    
    l2_grad = (1 - alpha) * 2 * beta
    
    beta = beta - eta * (gradient + el * l2_grad)
    
    beta = np.sign(beta) * np.maximum(0, np.abs(beta) - eta * el * alpha)
    
    return beta


class ElasticNet:
    def __init__(self, el, alpha, eta, batch, epoch):
       
        self.el = el
        self.alpha = alpha
        self.eta = eta
        self.batch = batch
        self.epoch = epoch
        self.beta = None
        return

    def coef(self):
        return self.beta

    def train(self, x, y):
        N, d = x.shape
        self.beta = np.zeros(d)  
        
        loss_history = []
        
        
        for ep in range(self.epoch):
            
            indices = np.random.permutation(N)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            
            for i in range(0, N, self.batch):
                x_batch = x_shuffled[i:i+self.batch]
                y_batch = y_shuffled[i:i+self.batch]
                
                
                for xi, yi in zip(x_batch, y_batch):
                    self.beta = grad_step(xi, yi, self.beta, self.el, self.alpha, self.eta)
            
            epoch_loss = loss(x, y, self.beta, self.el, self.alpha)
            loss_history.append({'epoch': ep, 'loss': epoch_loss})
            
        return loss_history

    def predict(self, x):
        return x @ self.beta

if __name__ == '__main__':
    el_net = ElasticNet(el=0.01, alpha=0.5, eta=0.01, batch=32, epoch=100)
    train_data,val_data,test_data=q2.load_data()
    trainx, trainy = q2.split_features_target(train_data)
    valx, valy = q2.split_features_target(val_data)
    testx, testy = q2.split_features_target(test_data)
    # Train the model
    history = el_net.train(trainx, trainy)

    # Predict new values
    predictions = el_net.predict(testx)
    print(predictions)
    # Get the learned coefficients
    coefficients = el_net.coef()
    
