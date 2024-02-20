import random
import json
import torch 
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

'''
This is the main chat file that uses all of the data from the training file and applies it to a chatbot created later 
in this file. The data comes from the neural net model created in the model file which then processes the data in the
training file. This file completes the cycle by being able to put the chatbot to use and connecting the chatbot to 
the front end where many will be able to use it. 

This was a very enjoyable project, and there are still many things that I learned here that I will have to revisit,
but it was a fun experience for my first true AI project! 

Skills learned: JSON, Pytorch basics, NLTK, Neural-Nets, Numpy (arrays and sets)
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Merlin"
print("Let's chat! type 'quit' to exit")
while True:
    sentence = input('you: ')
    if sentence == 'quit':
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probability = torch.softmax(output, dim=1)
    prob = probability[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
           if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        fallback = [intent["responses"] for intent in intents["intents"] if intent["tag"] == "fallback"]
        print(f"{bot_name}: {random.choice(fallback)}")

    


