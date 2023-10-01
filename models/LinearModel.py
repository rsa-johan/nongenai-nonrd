from torch.nn import Linear, Module, Tanh

class BaseLinearModel(Module):
    layer1: Module
    layer2: Module
    layer3: Module
    layer4: Module
    layer5: Module

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(BaseLinearModel, self).__init__()

        self.layer1 = Linear(input_size, hidden_size)
        self.layer2 = Linear(hidden_size, hidden_size*2)
        self.layer3 = Linear(hidden_size*2, hidden_size)
        self.layer4 = Linear(hidden_size, output_size)
        self.layer5 = Tanh()

    def forward(self, data):
        output = self.layer1(data)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
