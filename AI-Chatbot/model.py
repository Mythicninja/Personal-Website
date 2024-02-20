import torch
import torch.nn as nn

'''
using a Nerual net, the bag of words, represented as an array of arrays of 0s and 1s, will be passed through three nets
(between each net the data is passed trough an activation that makes them non-linear layers) and the output
will be pushed through a softmax (not shown in this code). In 'train.py', it will include a section to demonstrate
the epochs (when a complete dataset is cycled forward and backward through a net) as well as the loss rate. 

Being that this is the first time I work with neural nets, there is still I am learning from this project, but it has opened my
eyes to how doable it is to train a model using data and being able to get my outcome.
'''

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        #no activation and no softmax
        return out



