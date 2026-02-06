import os
import torch
from prepare_dataset import dataloaders
from get_args import args
from utils import to_spin
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function 
from torch import Tensor, argmax
from torch.optim.optimizer import Optimizer

Sizes = [5, 7, 11, 28]
DataSize = {5: 31, 7: 63, 11: 127, 28: 1023}

#Parameters
n_epochs = 1000
random_seed = 12345
learning_rate = 0.01
InputSize = Sizes[3]
InputDatasize = DataSize[InputSize]
log_interval = 10
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)
    

"""
Changes:
Cuda 0
Change cuda if you change
Folder Name for saving Model
Model Parameters
"""
    
DataFolder = f"Dataset/{InputSize}x{InputSize}/"


train_dataloader = torch.load(DataFolder + "Train.txt", weights_only=False)
test_dataloader = torch.load(DataFolder + "Test.txt", weights_only=False)


class Binarize(Function):
    clip_value = 1
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        # output = inp.sign()
        output = inp.new(inp.size())
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

class BinaryLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=False
    ):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, inp: Tensor) -> Tensor:
        weight = binarize(self.weight)
        out = F.linear(inp, weight)
        out = binarize(out)
        return out.to(inp.device)

class LastLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=False
    ):
        super().__init__(in_features, out_features, bias=bias)
    
    def forward(self, inp: Tensor) -> Tensor:
        weight = binarize(self.weight)
        out = F.linear(inp, weight)
        return out.to(inp.device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = BinaryLinear(InputDatasize, 7)
        # self.fc2 = BinaryLinear(511, 127)
        # self.fc3 = BinaryLinear(127, 31)
        self.fc4 = LastLayer(7, 10)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = self.fc4(x)
        softmax = nn.Softmax(dim=1)
        x = softmax(x)
        return x

network = Net()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adadelta(network.parameters(), lr=learning_rate)
# optimizer = optim.Adam(network.parameters(), lr=learning_rate)

device = device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
network.to(device)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_dataloader.dataset) for i in range(n_epochs + 1)]

name = f"{InputDatasize}"
for _ in list(network.named_children()):
    a = _[1].out_features
    name = name+f"x{a}"

ResultFolder = f"TrainedNN/{InputSize}x{InputSize}/{name}/"

def train(epoch, device):
    network.train()
    batch_idx = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = to_spin(data).to(device)
        target = target.type(torch.LongTensor)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_dataloader.dataset)))
            torch.save(network.state_dict(), ResultFolder + f"{InputDatasize}.pth")

def test(out=False, file=None):
    network.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_dataloader:
            data = to_spin(data).to(device)
            target = target.type(torch.LongTensor)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_dataloader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
    if out==True:
        file.write('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_dataloader.dataset),100. * correct / len(test_dataloader.dataset)))


try:
    os.makedirs(ResultFolder)
    print(f"Result Folder created successfully!")
except FileExistsError:
    print(f"Result Folder already exists!")

try:
    PATH = ResultFolder + f"{InputDatasize}.pth"
    network.load_state_dict(torch.load(PATH))
    network.eval()
except FileNotFoundError:
    pass

for epoch in range(1, n_epochs + 1):
    train(epoch, device)
    test()

file = open(ResultFolder+"/Info.txt", "w")
print(network, file=file)
test(out=True, file=file)
