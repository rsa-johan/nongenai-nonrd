from typing import Callable, Type
from numpy import sqrt, test
from tqdm import tqdm
import torch
from torch.nn import L1Loss, Module as Neuron
from torch.optim import SGD, Adam, Adagrad, Optimizer
from datatypes import Setup, Losses, Optimizers


def get_optimizer(typein: Optimizers, params, lr):
    match typein:
        case Optimizers.SGD:
            return SGD(params=params, lr=lr)
        case _:
            return SGD(params=params, lr=lr)


def get_loss_fn(typein: Losses):
    match typein:
        case Losses.L1:
            return L1Loss()
        case _:
            return L1Loss()


def preprocessor(engine, data, skip_fit):
    stdscaler = engine
    mean = 0
    std = 1 
    if not skip_fit:
        fit = stdscaler.fit(data)
        mean, std = fit.mean_ if fit.mean_ is not None else 1, sqrt(fit.var_ if fit.var_ is not None else 1)
    return stdscaler, torch.Tensor(stdscaler.transform(data)), mean, std



def trainer(setup: Type[Setup], train_input, train_output):
    device = torch.device('cpu')
    if setup.cuda:
        device = torch.device('cuda')

    model = setup.model.to(device)
    loss_func = get_loss_fn(setup.loss)
    optimizer_func = get_optimizer(setup.optimizer, model.parameters(), setup.lr)
    train_input, train_output = train_input.to(device), train_output.to(device)
    epochs = tqdm(range(setup.epochs)) if setup.show_progress else range(setup.epochs)

    for _ in epochs:
        model.train()

        pred_output = model(train_input)

        loss = loss_func(train_output, pred_output)
        optimizer_func.zero_grad()

        loss.backward()

        optimizer_func.step()

    return model

def evaluator(model: Neuron, mean, std, test_input, test_output, postprocessor: Callable):
    model.eval()
    with torch.inference_mode():
        inputs = (test_input - mean[0])/std[0]
        pred_output = model(inputs)
        postprocessor(test_output, pred_output)
