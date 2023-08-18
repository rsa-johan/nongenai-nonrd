from torch.nn import Module as Neuron, Linear
from torch.utils.data.dataloader import DataLoader

class Regressor(Neuron):
    input_layer: Neuron
    hidden_layer: Neuron
    hidden_layer_1: Neuron
    hidden_layer_2: Neuron
    hidden_layer_3: Neuron
    output_layer: Neuron

    def __init__(self):
        super(Regressor, self).__init__()

        self.input_layer = Linear(in_features=1, out_features=16)
        #self.hidden_layer_1 = Linear(in_features=16, out_features=128)
        #self.hidden_layer_2 = Linear(in_features=128, out_features=128)
        self.hidden_layer_3 = Linear(in_features=16, out_features=16)
        self.output_layer = Linear(in_features=16, out_features=1)


    def forward(self, inputs):
        output = self.input_layer(inputs)
        #output = self.hidden_layer_1(output)
        #output = self.hidden_layer_2(output)
        output = self.hidden_layer_3(output)
        output = self.output_layer(output)

        return output
