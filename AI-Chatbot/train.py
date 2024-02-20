import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet
#-------getting the data ready-------

#open the jsom file
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

#grab the patterns from the jsom file and tokenizes each sentence.
#Then, it then breaks adds it to an all words set
for intent in intents['intents']: 
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

#removing symbols and duplicates as well as sorting
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words)) # this function sorts words and removes duplicates at the same time
tags = sorted(set(tags))

#create arr for bag of words and tags
x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

#creating the datasets as arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

class chatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
#hyperparameters, first two and last two can be adjusted
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

#---------training the model using the dataset-----------

dataset = chatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.int64)

        #forward
        output = model(words)
        loss = criterion(output, labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#*****check the loss with every epoch*****
    #if (epoch +1) % 100 == 0:
        #print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
#print(f'final loss, loss={loss.item():.4f}')
        
#save the data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. File saved to {FILE}')