# In this problem you will use a popular RNN model 
# called the Gated Recurrent Units (RNN) to learn to 
# predict the sentiment of a film, television, etc. review.
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader
import torch
from string import punctuation
import torch.nn as nn

# =============================== Part 1: pre-processing ================================
# Each word is represented by a single number index in a vocabulary. 
# Remove all punctuation from the sentences. 
# Build a vocabulary from the unique words collected from text file so that each word is mapped to a number.
def pre_process(file):
    index_to_words = []
    word_to_index = {}
    dicts = {i:'' for i in punctuation}
    punc_table = str.maketrans(dicts)
    for line in open(file, "rb"):
        line = str(line).replace("<br />", " ")
        new_line = line.translate(punc_table)
        words = new_line.split()
        for w in words:
            if w not in index_to_words:
                index_to_words.append(w)
    for i, w in enumerate(index_to_words):
        word_to_index[w] = i
    print(len(index_to_words))
    return index_to_words, word_to_index

# Convert the data to a matrix where each row is a review. Pad or truncate. Use 400 as the fixed length.
def data_to_matrix(file1, file2, word_to_index, fixed_len):
    review_matrix = []
    train_y = []
    dicts = {i:'' for i in punctuation}
    punc_table = str.maketrans(dicts)
    for k in range(2):
        if k == 0:
            file = file1
        else:
            file = file2
        for line in open(file, "rb"):
            line = str(line).replace("<br />", " ")
            new_line = line.translate(punc_table)
            words = new_line.split()
            review = np.zeros(fixed_len, dtype=int)
            pad = max(0, fixed_len-len(words))
            for i, w in enumerate(words):
                if i >= fixed_len:
                    break
                review[i+pad] = word_to_index[w]
            review_matrix.append(review)
            if k == 0:
                train_y.append(1)
            else:
                train_y.append(0)

    return np.array(review_matrix), np.array(train_y)
# ==============================================================================================

# =================== Part 2 - Build A Binary Prediction RNN with RNN ==========================
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 50
batch_size = 100
learning_rate = 0.001

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # # Embedding layers
        # self.emb = nn.Embedding(num_emb, emb_dim)

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0
        )

        # # Fully connected layer
        # self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # x = self.emb(x)

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, 1, self.hidden_dim, dtype=torch.long).to(device)
        x = x.type(torch.LongTensor)
        x=x.view(-1,1,2)
        h0 = h0.type(torch.LongTensor)
        # Forward propagation by passing in the input and hidden state into the model
        out, hid = self.rnn(x)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        # out = self.fc(out)
        if out > 0:
            out = 1
        else:
            out = 0

        return out

def train_step(model, optimizer, criterion, x, y):
    # Sets model to train mode
    model.train()

    # Makes predictions
    yhat = model(x)
    # yhat = yhat.view(yhat.shape[0])

    # Computes loss
    loss = criterion(yhat, y)

    # Computes gradients
    loss.backward()

    # Updates parameters and zeroes gradients
    optimizer.step()
    optimizer.zero_grad()

    # Returns the loss
    return loss.item()


# ==============================================================================================

def main():
    input_data = torch.tensor([[[1,0],[1,1],[1,0],[0,0],[0,1],[1,1],[0,0]]],dtype=torch.long)
    y = torch.tensor([[1,0,0,1,1,0,1]],dtype=torch.long)
    model = RNNModel(input_dim=2, hidden_dim=3, layer_dim=1, output_dim=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss() # (logsoftmax + NLLLoss) or (Crossentropy)
    # train(model, optimizer, criterion, input_data)
    model.train()

    # Makes predictions
    yhat = model(input_data)
    # yhat = yhat.view(yhat.shape[0])

    # Computes loss
    loss = criterion(yhat, y)

    # Computes gradients
    loss.backward()

    # Updates parameters and zeroes gradients
    optimizer.step()
    optimizer.zero_grad()

    # Returns the loss
    return loss.item()

if __name__ == "__main__":
    main()

import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))


'''
Add two binary numbers

Each column is a time step.

00110100

01001101

10000001
'''

X = np.array([[0,1],
             [0,0],
             [1,1],
             [0,1],
             [1,0],
             [1,0],
             [0,1],
             [0,0]])

Y = np.array([[1],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [1]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((1,2)) - 1
syn1 = 2*np.random.random((1,1)) - 1
l1_t = 2*np.random.random((1,1)) - 1 # past example
syn2 = 2*np.random.random((1,1)) - 1

for x, y in zip(X, Y):

    # Feed forward through layers 0, 1, and 2
    l0 = x.reshape(2, 1)
    l1 = nonlin(np.add(np.dot(syn0, l0), np.dot(syn1, l1_t)))
    l2 = nonlin(np.dot(syn2, l1))

    # how much did we miss the target value?
    l2_error = y - l2

    print("Error:" + str(np.mean(np.abs(l2_error))))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_t_error = l2_delta.dot(syn1.T)
    l1_t_delta = l1_t_error *nonlin(l1,deriv=True)
    l1_error = l2_delta.dot(syn0)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn2 += l2.T.dot(l2_delta)
    syn1 += l1.T.dot(l1_t_delta)
    syn0 += l0.T.dot(l1_delta)

    # Save l1 for next iteration
    l1_t = l1