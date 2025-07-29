from train_model import X_train, y_train
from ml.model import train_model

import os
import pandas
import pytest

def test_algorithm():
    """
    Checks that the model used in model.py is AdaBoost
    """
    model = train_model(X_train, y_train)
    algorithm_expected = 'AdaBoostClassifier'
    assert type(model).__name__ == algorithm_expected, f'Algorithm\nExpected: {algorithm_expected}\n Actual: {type(model).__name__}'

def test_model_exists():
    """
    Checks if necessary model pickle file exists
    """
    file_path = './model/model.pkl'
    assert os.path.isfile(file_path)
    
def test_training_data_size():
    """
    Checks that the training set is 80% of the total dataset from a fresh read-in of the csv.
    """
    df = pandas.read_csv('./data/census.csv')
    all_rows = df.shape[0]
    train_rows = X_train.shape[0]
    assert train_rows == int(all_rows * 0.8)