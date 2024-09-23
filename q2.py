import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

train_path='HW1\energydata\energy_train.csv'
val_path='HW1\energydata\energy_val.csv'
test_path='HW1\energydata\energy_test.csv'

def split_features_target(data):
    
    
    X = data.drop(columns=['Appliances']).to_numpy()  
    y = data['Appliances'].to_numpy()
    
    return X, y

def preprocess_data(trainx,valx,testx):
         
    train_data = pd.read_csv(trainx)
    train_data['date'] = pd.to_datetime(train_data['date'], format='%m/%d/%y %H:%M')
    
    train_data['hour'] = train_data['date'].dt.hour
    train_data['day_of_week'] = train_data['date'].dt.dayofweek
    train_data['month'] = train_data['date'].dt.month
    train_data = train_data.drop(columns=['date'])
    
      
    
    if train_data.isnull().values.any():
        train_data = train_data.fillna(train_data.mean())  
            
            
    val_data = pd.read_csv(valx)
    val_data['date'] = pd.to_datetime(val_data['date'], format='%m/%d/%y %H:%M')
    
    val_data['hour'] = val_data['date'].dt.hour
    val_data['day_of_week'] = val_data['date'].dt.dayofweek
    val_data['month'] = val_data['date'].dt.month
    val_data = val_data.drop(columns=['date'])
    
     
    
    if val_data.isnull().values.any():
        val_data = val_data.fillna(val_data.mean()) 
            
    test_data = pd.read_csv(testx)
    test_data['date'] = pd.to_datetime(test_data['date'], format='%m/%d/%y %H:%M')
    
    test_data['hour'] = test_data['date'].dt.hour
    test_data['day_of_week'] = test_data['date'].dt.dayofweek
    test_data['month'] = test_data['date'].dt.month
    test_data = test_data.drop(columns=['date'])
      
    
    if test_data.isnull().values.any():
        test_data = test_data.fillna(test_data.mean()) 
    
    return train_data,val_data,test_data


def eval_linear1(trainx, trainy, valx, valy, testx, testy):
         
    model = LinearRegression()
    
    
    model.fit(trainx, trainy)
    
    
    train_pred = model.predict(trainx)
    val_pred = model.predict(valx)
    test_pred = model.predict(testx)
    
    
    train_rmse = np.sqrt(mean_squared_error(trainy, train_pred))
    val_rmse = np.sqrt(mean_squared_error(valy, val_pred))
    test_rmse = np.sqrt(mean_squared_error(testy, test_pred))
    
    train_r2 = r2_score(trainy, train_pred)
    val_r2 = r2_score(valy, val_pred)
    test_r2 = r2_score(testy, test_pred)
    
    
    results = {
        'train-rmse': train_rmse,
        'train-r2': train_r2,
        'val-rmse': val_rmse,
        'val-r2': val_r2,
        'test-rmse': test_rmse,
        'test-r2': test_r2
    }
    
    return results

def eval_linear2(trainx, trainy, valx, valy, testx, testy):

    combined_x = np.vstack((trainx, valx))  
    combined_y = np.hstack((trainy, valy))
    
    model = LinearRegression()
    
    model.fit(combined_x, combined_y)
    
    train_pred = model.predict(trainx)
    val_pred = model.predict(valx)
    test_pred = model.predict(testx)
    
    train_rmse = np.sqrt(mean_squared_error(trainy, train_pred))
    val_rmse = np.sqrt(mean_squared_error(valy, val_pred))
    test_rmse = np.sqrt(mean_squared_error(testy, test_pred))
    
    train_r2 = r2_score(trainy, train_pred)
    val_r2 = r2_score(valy, val_pred)
    test_r2 = r2_score(testy, test_pred)
    
    results = {
        'train-rmse': train_rmse,
        'train-r2': train_r2,
        'val-rmse': val_rmse,
        'val-r2': val_r2,
        'test-rmse': test_rmse,
        'test-r2': test_r2
    }
    
    return results


def eval_ridge(trainx, trainy, valx, valy, testx, testy, alpha):
    
    
    model = Ridge(alpha=alpha)
    
    model.fit(trainx, trainy)
    
    train_pred = model.predict(trainx)
    val_pred = model.predict(valx)
    test_pred = model.predict(testx)
    
    train_rmse = np.sqrt(mean_squared_error(trainy, train_pred))
    val_rmse = np.sqrt(mean_squared_error(valy, val_pred))
    test_rmse = np.sqrt(mean_squared_error(testy, test_pred))
    
    train_r2 = r2_score(trainy, train_pred)
    val_r2 = r2_score(valy, val_pred)
    test_r2 = r2_score(testy, test_pred)
    
    results = {
        'train-rmse': train_rmse,
        'train-r2': train_r2,
        'val-rmse': val_rmse,
        'val-r2': val_r2,
        'test-rmse': test_rmse,
        'test-r2': test_r2
    }
    
    return results


def eval_lasso(trainx, trainy, valx, valy, testx, testy, alpha):
    
    model = Lasso(alpha=alpha)
    
    model.fit(trainx, trainy)
    
    train_pred = model.predict(trainx)
    val_pred = model.predict(valx)
    test_pred = model.predict(testx)
    
    train_rmse = np.sqrt(mean_squared_error(trainy, train_pred))
    val_rmse = np.sqrt(mean_squared_error(valy, val_pred))
    test_rmse = np.sqrt(mean_squared_error(testy, test_pred))
    
    train_r2 = r2_score(trainy, train_pred)
    val_r2 = r2_score(valy, val_pred)
    test_r2 = r2_score(testy, test_pred)
    
    results = {
        'train-rmse': train_rmse,
        'train-r2': train_r2,
        'val-rmse': val_rmse,
        'val-r2': val_r2,
        'test-rmse': test_rmse,
        'test-r2': test_r2
    }
    
    return results
    
if __name__ == '__main__':
    train_data,val_data,test_data=preprocess_data(train_path,val_path,test_path)
    
    trainx, trainy = split_features_target(train_data)
    valx, valy = split_features_target(val_data)
    testx, testy = split_features_target(test_data)
    
    
    results1=eval_linear1(trainx, trainy, valx, valy, testx, testy)
    print(results1)
    
    results2=eval_linear2(trainx, trainy, valx, valy, testx, testy)
    print(results2)
    
    results_ridge=eval_ridge(trainx, trainy, valx, valy, testx, testy, alpha=10.0)
    print(results_ridge)
    
    results_lasso=eval_lasso(trainx, trainy, valx, valy, testx, testy, alpha=2.0)
    print(results_lasso)
    
