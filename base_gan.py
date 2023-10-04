from utils.config import get_config
from utils.processor import processor
from utils.train_test import trainer, tester
from utils.data import read_file, read_folder, get_as_df
from models.LinearModel import BaseLinearModel

# Get Input & Output from csv file
filecode = 'sd'
input_cols, output_cols = ['YearsExperience'], ['Salary']
X, Y = read_file(read_folder(), filecode, input_cols, output_cols)

#Preprocess the Input & Output
X, Y, preprocess, postprocess = processor(X, Y)

#Model creation
input_size = len(input_cols)
hidden_size = input_size*4
output_size = len(output_cols)
model = BaseLinearModel(input_size, hidden_size, output_size)
print(f"Model params: {input_size}, {hidden_size}, {output_size}")

#Model training
model, test_input, test_output = trainer(model, X, Y, 0.2, 12, 0.01, "mse", "adam")

#Model tester
output = tester(model, test_input, test_output, preprocess, postprocess)


#Model Trial
trial_input = get_as_df({"YearsExperience": [10, 20]}) 
_ ,out = postprocess(None, model(preprocess(trial_input, None)[0]).detach().numpy())
print(out)
