# In this problem you will use a popular RNN model 
# called the Gated Recurrent Units (GRU) to learn to 
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

# =================== Part 2 - Build A Binary Prediction RNN with GRU ==========================
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 50
batch_size = 100
learning_rate = 0.001

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, num_emb, emb_dim, dropout_prob):
        super(GRUModel, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        self.emb = nn.Embedding(num_emb, emb_dim)

        self.gru = nn.GRU(
            emb_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = self.emb(x)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device).requires_grad_()
        out, hid = self.gru(x, h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)

        return out

def train_step(model, optimizer, criterion, x, y):
    model.train()
    yhat = model(x)
    loss = criterion(yhat, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def train(model, optimizer, criterion, train_loader, test_loader, batch_size=batch_size, n_epochs=num_epochs):
    model_path = f'models/{model}_1'

    train_losses = []
    test_acc = []

    for epoch in range(1, n_epochs + 1):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.view([batch_size, -1]).to(device)
            y_batch = y_batch.to(device)
            loss = train_step(model, optimizer, criterion, x_batch, y_batch)
            batch_losses.append(loss)
        training_loss = np.mean(batch_losses)
        train_losses.append(training_loss)

        with torch.no_grad():
            batch_test_acc = []
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.view([batch_size, -1]).to(device)
                y_batch = y_batch.to(device)
                model.eval()
                outputs = model(x_batch)
 
                _, pred = torch.max(outputs.data, 1)
                correct_counts = pred.eq(y_batch.data.view_as(pred))
 
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                batch_test_acc.append(acc)

            avg_test_acc = np.mean(batch_test_acc)
            test_acc.append(avg_test_acc)

        print(
            f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Test Acc: {avg_test_acc:.4f}"
        )

    torch.save(model.state_dict(), model_path)
# ==============================================================================================

def main():
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
    train_data = TensorDataset(torch.from_numpy(review_matrix), torch.from_numpy(train_y).type(torch.LongTensor))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    
    test_review_matrix, y = data_to_matrix(TEST_FILE_1, TEST_FILE_2, word_to_index, fixed_length)
    test_data = TensorDataset(torch.from_numpy(test_review_matrix), torch.from_numpy(y).type(torch.LongTensor))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

    # num_emb = 42484
    model = GRUModel(input_dim=fixed_length, hidden_dim=50, layer_dim=1, output_dim=2, num_emb=len(index_to_words), emb_dim=500, dropout_prob=0.2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss() # (logsoftmax + NLLLoss) or (Crossentropy)
    train(model, optimizer, criterion, train_loader, test_loader)

if __name__ == "__main__":
    main()