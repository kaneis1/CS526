import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

train_path='HW1\energydata\energy_train.csv'
val_path='HW1\energydata\energy_val.csv'
test_path='HW1\energydata\energy_test.csv'


def load_data():
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    return train_data,val_data,test_data


def split_features_target(data):
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%y %H:%M')
    
    data['hour'] = data['date'].dt.hour
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data = data.drop(columns=['date'])
    if data.isnull().values.any():
        data = data.fillna(data.mean())  
    X = data.drop(columns=['Appliances']).to_numpy() 
    y = data['Appliances'].to_numpy()
    
    return X, y


def feature_extraction(trainx):
    train_temp_in = trainx[:, [1, 3, 5, 7, 9, 13, 15, 17]]
    train_temp_out = trainx[:, [11, 19, 24]]
    train_humid_in = trainx[:, [2, 4, 6, 8, 10, 14, 16, 18]]
    train_humid_out = trainx[:, [12, 21]]
    train_avg_temp_out = np.mean(train_temp_out, axis=1).reshape(-1, 1)
    train_avg_humid_out = np.mean(train_humid_out, axis=1).reshape(-1, 1)
    train_avg_temp_in = np.mean(train_temp_in, axis=1).reshape(-1, 1)
    train_avg_humid_in = np.mean(train_humid_in, axis=1).reshape(-1, 1)
    trainx[:, [1, 3, 5, 7, 9, 13, 15, 17]]=train_avg_temp_in
    trainx[:, [11, 19, 24]]=train_avg_temp_out
    trainx[:, [2, 4, 6, 8, 10, 14, 16, 18]]=train_avg_humid_in
    trainx[:, [12, 21]]=train_avg_humid_out
    return trainx

def preprocess_data(trainx,valx,testx):
     
    
         
    trainx=trainx.astype(np.int64)
    
    valx=valx.astype(np.int64)
            
            
    testx=testx.astype(np.int64)
    
    return trainx,valx,testx


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
    train_data,val_data,test_data=load_data()
    
   
    
    trainx, trainy = split_features_target(train_data)
    valx, valy = split_features_target(val_data)
    testx, testy = split_features_target(test_data)
    
    trainx,valx,testx=preprocess_data(trainx,valx,testx)
    

    results1=eval_linear1(trainx, trainy, valx, valy, testx, testy)
    print(results1)
    
    results2=eval_linear2(trainx, trainy, valx, valy, testx, testy)
    print(results2)
    
    results_ridge=eval_ridge(trainx, trainy, valx, valy, testx, testy, alpha=10.0)
    print(results_ridge)
    
    results_lasso=eval_lasso(trainx, trainy, valx, valy, testx, testy, alpha=2.0)
    print(results_lasso)
    
