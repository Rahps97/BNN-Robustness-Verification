import os
import json
import torch
from get_args import args
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.autograd import Function
from utils import to_spin
import numpy as np
from bnn_as_qubo import setup_optim_model
torch.set_printoptions(threshold=torch.inf) 

# Load the inputs
Sizes = [5, 7, 11, 28]
InputDataSize = {5: 31, 7: 63, 11: 127, 28: 1023}
PetrubSize = {5: 16, 7: 32, 11: 64, 28: 256}
PetrubSizeBound = {5: 8, 7: 32, 11: 32, 28: 128}

# Variables
InputSize = Sizes[0]
InputDataSize = InputDataSize[InputSize]
args.pixels_to_perturb_len = PetrubSize[InputSize]
args.LAMBDA = {
    'sum_taus': 0.1,
    'output': 1,
    'hard_constraints': 1,
    'perturbation_bound_constraint': 1,
    'epsilon': 1
    }# Relative to the other lambdas. 0.1 means 10% of the other lambdas.
args.objective = 'zero'
args.epsilon = PetrubSizeBound[InputSize]
args.include_perturbation_bound_constraint = True
args.selected_targets = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

#Folder for Data
DataFolder = f"Dataset/{InputSize}x{InputSize}/"

# Loading Data
train_dataloader = torch.load(DataFolder + f'Train.txt', weights_only=False)
test_dataloader = torch.load(DataFolder + f'Test.txt', weights_only=False)

# Functions for Layers
class Binarize(Function):
    clip_value = 1

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        output = inp.new(inp.size())  # output = inp.sign() It doesn't work because of 0.
        output[inp >= 0] = 1
        output[inp < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp: Tensor = ctx.saved_tensors[0]
        clipped = inp.abs() <= Binarize.clip_value
        output = torch.zeros(inp.size()).to(grad_output.device)
        output[clipped] = 1
        output[~clipped] = 0
        return output * grad_output

binarize = Binarize.apply

# Layers
class BinaryLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias=False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, inp: Tensor) -> Tensor:
        weight = binarize(self.weight)
        out = F.linear(inp, weight)
        out = binarize(out)
        return out.to(inp.device)

class LastLayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias=False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, inp: Tensor) -> Tensor:
        weight = binarize(self.weight)
        out = F.linear(inp, weight)
        return out.to(inp.device)

class QUBONet(nn.Module):
    def __init__(self):
        super(QUBONet, self).__init__()
        self.fc1 = BinaryLinear(InputDataSize, 7)
        # self.fc2 = BinaryLinear(511, 127)
        # self.fc3 = BinaryLinear(127, 31)
        self.fc4 = LastLayer(7, 10)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = self.fc4(x)
        x = torch.argmax(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
qubonet = QUBONet().to(device)

name = f"{InputDataSize}"
for _ in list(qubonet.named_children()):
    a = _[1].out_features
    name = name+f"x{a}"
ResultFolder = f"TrainedNN/{InputSize}x{InputSize}/{name}/"

qubonet.load_state_dict(torch.load(ResultFolder + f'{InputDataSize}.pth', weights_only=True))
qubonet.eval()

args.pixels_to_perturb = list(
    train_dataloader.dataset.tensors[0][:, 0:InputSize*InputSize] # Rahul : This brings the perturbation in non-padded region
    .mean(axis=0).abs()
    .topk(min(args
              .pixels_to_perturb_len,
              train_dataloader.dataset.tensors[0][:, 0:InputSize*InputSize] # Rahul : This brings the perturbation in non-padded region
              .shape[1]), largest=False).indices.numpy())


with torch.no_grad():
    for image_index in range(len(train_dataloader)):
        sample_input_item = train_dataloader.dataset[image_index]
        sample_input_boolean = sample_input_item[0]
        sample_input_spin = to_spin(sample_input_boolean)
        sample_input_target = sample_input_item[1]
        shape = sample_input_spin.shape[-2:]
        # if the model already predict the sample
        # with an incorrect class; skip.
        nn_output = qubonet(torch.
                            Tensor(np.array([np.array(sample_input_spin)]))
                            .to(device))
        if nn_output != sample_input_target:
            continue
        else:
            break

H, ordered_variables = setup_optim_model(sample_input_spin, sample_input_target, qubonet, args)
qubo = H.to_qubo().Q

QUBOFolder = f"QUBO/{InputSize}x{InputSize}/{name}/"
try:
    os.makedirs(QUBOFolder)
    print(f"Folder '{QUBOFolder}' created successfully!")
except FileExistsError:
    print(f"Folder '{QUBOFolder}' already exists!")


temp_x, temp_y = map(max, zip(*qubo))
res = [[qubo.get((j, i), 0) for i in range(temp_y + 1)
        ]for j in range(temp_x + 1)]

np.savetxt(QUBOFolder + 'QUBO_W.txt', res)

with open(QUBOFolder + "Variables.json", "w") as f:
    json.dump(ordered_variables, f, indent="\t")

# Only needed for Outsiders 
file = open(QUBOFolder+"Info_Relevant.txt", "w")
file.write(f"Minimum Energy : {-H.to_qubo()[()]} \n")
file.write(f"Total Variables : {len(H.variables)} \n")
file.write(f"Minimum elemenet of the QUBO : {min(qubo.values())} \n")
file.write(f"Maximum elemenet of the QUBO : {max(qubo.values())} \n")
file.close()


file = open(QUBOFolder+"Info.txt", "w")
file.write(f"Minimum Energy : {-H.to_qubo()[()]} \n")
file.write(f"Total Variables : {len(H.variables)} \n")
file.write(f"Minimum elemenet of the QUBO : {min(qubo.values())} \n")
file.write(f"Maximum elemenet of the QUBO : {max(qubo.values())} \n")
file.write(f"Total Pixels to perturb : {args.pixels_to_perturb_len} \n")
file.write(f"Pixels perturbed : {args.pixels_to_perturb} \n")
file.write(f"Input to perturb : {sample_input_boolean} \n")
file.write(f"Label to perturb : {sample_input_target} \n")
file.write(f"Epsilon : {args.epsilon} \n")
file.write(f"Perturbation Bound Constriant included : {args.include_perturbation_bound_constraint} \n")
file.close()

