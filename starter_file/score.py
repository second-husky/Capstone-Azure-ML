# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:26:05 2021

@author: user
"""

import json
import numpy as np
import os
from sklearn.externals import joblib


def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'hyper-model.joblib')
    model = joblib.load(model_path)

def run(data):
    try:
        
        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
