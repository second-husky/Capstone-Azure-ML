# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 10:15:40 2021

@author: user
"""

import argparse
import os
import joblib
import tarfile
from six.moves import urllib

import numpy as np
import pandas as pd


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklean.metrics import metrics


from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):
    """
    clean input housing data using one-hot encoding and standardscaler, return dataframes X and y
    """
    cleaned_data = data.to_pandas_dataframe().dropna().reset_index(drop=True)
    #one-hot encoding of ocean_proximity column
    one_hot_data = pd.concat([cleaned_data.drop("ocean_proximity",axis = 1), pd.get_dummies(cleaned_data["ocean_proximity"],prefix = "ocean_proximity")], axis = 1)
    SS = StandardScaler()
    scaled_data = pd.DataFrame(SS.fit_transform(one_hot_data.drop("median_house_value",axis = 1)),columns = one_hot_data.drop("median_house_value",axis = 1).columns)
    X_data = scaled_data
    y_data = cleaned_data["median_house_value"]
    return X_data, y_data

# TODO: Create TabularDataset using TabularDatasetFactory
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    ds = TabularDatasetFactory.from_delimited_files(csv_path, separator = ',')
    return ds

ds = fetch_housing_data()
X, y = clean_data(ds)

# TODO: Split data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

run = Run.get_context()    

def main():
    # Add arguments to script
#    parser = argparse.ArgumentParser()
#
#    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
#    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
#
#    args = parser.parse_args()
#
#    run.log("Regularization Strength:", np.float(args.C))
#    run.log("Max iterations:", np.int(args.max_iter))
#
#    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
#
#    r2 = metrics.r2_score(y_test, y_predict)
#    run.log("Accuracy", np.float(accuracy))
#
#    os.makedirs('./outputs', exist_ok = True)
#    joblib.dump(value = model, filename='./outputs/model.joblib')

#===========================================================

    linear_model = LinearRegression()
    linear_model.fit(X_train,y_train)


    y_predict = linear_model.predict(X_test)
#    r2 = metrics.r2_score(y_test, y_predict)

if __name__ == '__main__':
    main()