from sklearn.metrics import f1_score
from torch import nn
from utils import *
import numpy as np
import random
import torch

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

DIM = 300
BATCH_SIZE = 32


class MyLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.hidden2tag = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_lens):
        lstm_out_padding, _ = self.lstm(x)

        # concatenating all the non-padded words h states vectors to a single matrix
        lstm_out = torch.cat([matrix[torch.arange(x_len)] for matrix, x_len in zip(lstm_out_padding, x_lens)])

        output = self.hidden2tag(lstm_out)
        probability_preds = self.sigmoid(output)

        return torch.squeeze(probability_preds)


def train_model_3(model, train_loader, f_loss, f_optim, device, epochs):
    """Trains a given model on the same batch where each sample is assigned to a random binary label,
    and returns a list of train + test loss (according to loss function 'f_loss') in every epoch."""

    for epoch in range(1, epochs+1):

        # print current epoch
        if epoch == 1 or epoch % 2 == 0:
            print(f"Processing epoch [{epoch}/{epochs}]")

        for train_x, train_x_lens, train_y in train_loader:
            # upload train batch to GPU if available
            train_x, train_x_lens, train_y = train_x.to(device), train_x_lens.to(device), train_y.to(device)

            # # get non-padded word's indices for each sentences
            # indices = [np.arange(x_len) for x_len in train_x_lens]

            # get probability predications
            train_y_prob_pred = model(train_x, train_x_lens)

            # flat train labels
            train_y_flatten = torch.cat([sen_labels[torch.arange(sen_len)] for sen_labels, sen_len in zip(train_y, train_x_lens)])

            # get loss
            train_y_flatten, train_y_prob_pred = train_y_flatten.to(device), train_y_prob_pred.to(device)
            train_loss = f_loss(train_y_prob_pred, train_y_flatten)

            # back prop
            f_optim.zero_grad()
            train_loss.backward()
            f_optim.step()

def get_preds_model_3(model, data_loader, device):
    prob_pred_lst = []

    model.eval()

    with torch.no_grad():
        for x, x_lens in data_loader:
            # upload train batch to GPU if available
            x, x_lens = x.to(device), x_lens.to(device)

            prob_pred = model(x, x_lens)
            prob_pred_lst += prob_pred.tolist()

    # convert probability predictions into binary labels
    y_pred_lst = np.round(np.array(prob_pred_lst))
    return y_pred_lst

def train_and_evaluate_model_3(train_x, train_x_lens, train_y, test_x, test_x_lens, test_y):
    train_loader = get_data_loader(x=train_x, x_lens=train_x_lens, y=train_y, is_flatten=False, is_double=False, is_test=False,
                                   batch_size=BATCH_SIZE, shuffle=True)

    train_loader_untagged = get_data_loader(x=train_x, x_lens=train_x_lens, y=train_y, is_flatten=False, is_double=False, is_test=True,
                                   batch_size=BATCH_SIZE, shuffle=False)

    test_loader = get_data_loader(x=test_x, x_lens=test_x_lens, y=test_y, is_flatten=False, is_double=False, is_test=True,
                                  batch_size=BATCH_SIZE, shuffle=False)

    # get device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # get model
    lstm = MyLSTM(embedding_dim=DIM, hidden_dim=100).to(device)

    # get loss and optimization functions
    bce_loss = nn.BCELoss()
    adam_optim = torch.optim.Adam(lstm.parameters(), lr=0.01)

    train_model_3(lstm, train_loader, bce_loss, adam_optim, device, epochs=10)

    # evaluate on 'train.tagged'
    train_y_preds = get_preds_model_3(lstm, train_loader_untagged, device)
    train_y = np.concatenate([sen_labels[np.arange(sen_len)] for sen_labels, sen_len in zip(train_y, train_x_lens)])
    train_f1 = f1_score(train_y, train_y_preds)
    print(f"\nF1 Score for Model 3 on 'train.tagged': {round(train_f1, 3)}\n")

    # evaluate on 'dev.tagged'
    test_y_preds = get_preds_model_3(lstm, test_loader, device)
    test_y = np.concatenate([sen_labels[np.arange(sen_len)] for sen_labels, sen_len in zip(test_y, test_x_lens)])
    test_f1 = f1_score(test_y, test_y_preds)
    print(f"\nF1 Score for Model 3 on 'dev.tagged': {round(test_f1, 3)}\n")
