from torch import Tensor, inference_mode
from torch.optim import Adam, SGD
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss, Module
from sklearn.model_selection import train_test_split

def trainer(model: Module, X, Y, test_perc: float, epochs: int, lr: float, loss_fn: str, optim_fn: str):
    train_input, test_input, train_output, test_output = train_test_split(X, Y, test_size=test_perc, random_state=42)
    train_input, test_input, train_output, test_output = Tensor(train_input), Tensor(test_input), Tensor(train_output), Tensor(test_output)

    _loss_fn = get_loss_fn(loss_fn)
    _optim_fn = get_optim_fn(optim_fn, lr, model.parameters())

    model.train(True)
    if _loss_fn and _optim_fn:
        for _ in range(epochs):
            _optim_fn.zero_grad()

            output = model(train_input)
            loss = _loss_fn(test_output, output)

            loss.backward()
            _optim_fn.step()

    return model, test_input, test_output

def tester(model: Module, test_input, test_output, preprocess, postprocess):
    model.eval()
    with inference_mode():
        output = model(test_input)
        output = postprocess(output.numpy(force=True))
        return output
        

def get_loss_fn(loss_fn: str):
    _loss_fn = None
    match loss_fn:
        case "mse":
            _loss_fn = MSELoss()
        case "ce":
            _loss_fn = CrossEntropyLoss()
        case "bce":
            _loss_fn = BCELoss()
        case "bcel":
            _loss_fn = BCEWithLogitsLoss
        case _:
            _loss_fn = L1Loss()

    return _loss_fn

def get_optim_fn(optim_fn: str, lr: float, params):
    _optim_fn = None
    match optim_fn:
        case "adam":
            _optim_fn = Adam(lr=lr, params=params)
        case "sgd":
            _optim_fn = SGD(lr=lr, params=params)

    return _optim_fn
