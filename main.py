import torch
from utils import *
import polars as pl
from typing import Type
from model import Regressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

stats = { 'mean': 0.0 , 'std': 1.0 }
process_engine = StandardScaler()


def load_process(file):
    data = pl.read_csv(file)
    data = data.drop(data.columns[0])
    global process_engine
    process_engine, processed, mean, std = preprocessor(process_engine, data, False)
    global stats
    stats['mean'] = mean
    stats['std'] = std
    X = processed[:, 0].unsqueeze(dim=1)
    Y = processed[:, 1].unsqueeze(dim=1)
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def postprocess(test_output, pred_output):
    print("Pred-output: ", pred_output)
    global stats
    y_pred = pred_output*stats.get('std')[1] + stats.get('mean')[1]
    print("Y_pred: ", y_pred.round())
    print("Test_oout: ", test_output)
    print(torch.eq(test_output, pred_output))

def run():
    train_input, test_input, train_output, test_output = load_process('dataset.csv')
    setup: Type[Setup] = Setup
    setup.model = Regressor()
    model = trainer(setup, train_input, train_output)
    global stats
    evaluator(model, stats['mean'], stats['std'], test_input, test_output, postprocess)

if __name__ == '__main__':
    run()
