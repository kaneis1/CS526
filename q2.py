import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

train_path='HW1\energydata\energy_train.csv'
val_path='HW1\energydata\energy_val.csv'
test_path='HW1\energydata\energy_test.csv'


def preprocess_data(trainx,valx,testx):
         
    trainx=np.array([17, 55, 7, 84, 17, 41, 18, 48, 17, 45, 6, 733, 92, 7, 63, 5])
    return trainx,trainx,trainx



    
