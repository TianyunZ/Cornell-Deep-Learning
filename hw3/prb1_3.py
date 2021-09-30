import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from string import punctuation
import numpy as np
import torch.optim as optim

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

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_dim, out_dim):
        super(MLP, self).__init__()
        
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.type(torch.FloatTensor)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        pred_y = self.softmax(x)
        
        return pred_y

class MLPModel():
    """ Maximum entropy model for classification.

    Attributes:

    """
    def __init__(self):
        # Initialize the parameters of the model.
        # TODO: Implement initialization of this model.
        input_size = 400
        hidden_dim = 64

        out_dim = 2
        lr = 0.000005
        num_epochs = 100
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mlp_model = MLP(input_size, hidden_dim, out_dim)

        self.criteria = nn.CrossEntropyLoss()
        self.device = device
        mlp_model.to(device)

        self.model = mlp_model
        optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train(self, train_loader, test_loader):
        train_losses = []
        test_acc = []
        self.model.train()
        for epoch in range(self.num_epochs):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.type(torch.LongTensor)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(x_batch)
                loss = self.criteria(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_losses.append(loss.item())
            training_loss = np.mean(batch_losses)
            train_losses.append(training_loss)

            with torch.no_grad():
                batch_test_acc = []
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.view([batch_size, -1]).to(self.device)
                    y_batch = y_batch.type(torch.LongTensor)
                    y_batch = y_batch.to(self.device)
                    self.model.eval()
                    outputs = self.model(x_batch)
    
                    _, pred = torch.max(outputs.data, 1)
                    correct_counts = pred.eq(y_batch.data.view_as(pred))
    
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))
                    batch_test_acc.append(acc)

                avg_test_acc = np.mean(batch_test_acc)
                test_acc.append(avg_test_acc)

            print(
                f"[{epoch}] Training loss: {training_loss:.4f}\t Test Acc: {avg_test_acc:.4f}"
            )

if __name__ == "__main__":
    batch_size = 100
    # train_data, dev_data, test_data, data_type = load_data(sys.argv)
    ALL_FILE = "sentiment_data/data/all_merged.txt"
    TRAIN_FILE_1 = "sentiment_data/data/train_pos_merged.txt"
    TRAIN_FILE_2 = "sentiment_data/data/train_neg_merged.txt"
    # TRAIN_FILE_1 = "sentiment_data/data/demo_pos.txt"
    # TRAIN_FILE_2 = "sentiment_data/data/demo_neg.txt"
    TEST_FILE_1 = "sentiment_data/data/test_pos_merged.txt"
    TEST_FILE_2 = "sentiment_data/data/test_neg_merged.txt"
    fixed_length = 400
    index_to_words, word_to_index = pre_process(ALL_FILE)
    review_matrix, train_y = data_to_matrix(TRAIN_FILE_1, TRAIN_FILE_2, word_to_index, fixed_length)
    train_data = TensorDataset(torch.from_numpy(review_matrix), torch.from_numpy(train_y).type(torch.FloatTensor))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    
    test_review_matrix, y = data_to_matrix(TEST_FILE_1, TEST_FILE_2, word_to_index, fixed_length)
    test_data = TensorDataset(torch.from_numpy(test_review_matrix), torch.from_numpy(y).type(torch.FloatTensor))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

    # num_emb = 42484
    model = MLPModel()
    model.train(train_loader, test_loader)

