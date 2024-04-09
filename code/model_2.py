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


class MyFCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # initialize layers
        self.layers = nn.Sequential(
            nn.Linear(DIM, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

        # create sigmoid
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # flattened_input = self.flatten(input_layer)

        output_layer = self.layers(x)
        probability_preds = self.sigmoid(output_layer)

        return torch.squeeze(probability_preds)


def train_model_2(model, train_loader, f_loss, f_optim, device, epochs):
    """Trains a given model on the same batch where each sample is assigned to a random binary label,
    and returns a list of train + test loss (according to loss function 'f_loss') in every epoch."""

    for epoch in range(1, epochs+1):

        # print current epoch
        if epoch == 1 or epoch % 2 == 0:
            print(f"Processing epoch [{epoch}/{epochs}]")

        for train_x, train_y in train_loader:
            # upload train batch to GPU if available
            train_x, train_y = train_x.to(device), train_y.to(device)

            # get probability predications
            train_y_prob_pred = model(train_x)
            train_y_prob_pred = train_y_prob_pred.to(device)

            # get loss
            train_loss = f_loss(train_y_prob_pred, train_y)

            # back prop
            f_optim.zero_grad()
            train_loss.backward()
            f_optim.step()


def get_preds_model_2(model, data_loader, device):
    prob_pred_lst = []

    model.eval()

    with torch.no_grad():
        for test_x in data_loader:
            # upload train batch to GPU if available
            test_x = test_x.to(device)

            prob_pred = model(test_x)
            prob_pred_lst += prob_pred.tolist()

    # convert probability predictions into binary labels
    y_pred_lst = np.round(np.array(prob_pred_lst))
    return y_pred_lst


def train_and_evaluate_model_2(train_x, train_y, test_x, test_y):
    train_loader = get_data_loader(x=train_x, y=train_y, is_flatten=True, is_double=False, is_test=False,
                                   batch_size=BATCH_SIZE, shuffle=True)

    train_loader_untagged = get_data_loader(x=train_x, y=train_y, is_flatten=True, is_double=False, is_test=True,
                                   batch_size=BATCH_SIZE, shuffle=False)

    test_loader = get_data_loader(x=test_x, y=test_y, is_flatten=True, is_double=False, is_test=True,
                                  batch_size=BATCH_SIZE, shuffle=False)

    # get device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # get model
    fc_nn = MyFCNN().to(device)

    # get loss and optimization functions
    bce_loss = nn.BCELoss()
    adam_optim = torch.optim.Adam(fc_nn.parameters(), lr=0.01)

    # train and save the model, and the random labels
    train_model_2(fc_nn, train_loader, bce_loss, adam_optim, device, epochs=10)


    # evaluate on 'train.tagged'
    train_y_preds = get_preds_model_2(fc_nn, train_loader_untagged, device)
    train_f1 = f1_score(train_y, train_y_preds)
    print(f"\nF1 Score for Model 2 on 'train.tagged': {round(train_f1, 3)}\n")

    # evaluate on 'dev.tagged'
    test_y_preds = get_preds_model_2(fc_nn, test_loader, device)
    test_f1 = f1_score(test_y, test_y_preds)
    print(f"\nF1 Score for Model 2 on 'dev.tagged': {round(test_f1, 3)}\n")
