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

W2V_DIM, GLOVE_DIM = 300, 200

# hyperparameters
LR = 0.001
NUM_EPOCHS = 8
BATCH_SIZE = 64
HIDDEN_SIZE = 128


class MyLSTM(nn.Module):
    def __init__(self, w2v_dim, glove_dim, hidden_dim):
        super().__init__()
        self.lstm_w2v = nn.LSTM(w2v_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm_glove = nn.LSTM(glove_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        self.lstm_total = nn.LSTM(2*hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.hidden2tag = nn.Sequential(nn.ReLU(), nn.Linear(2*hidden_dim, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_w2v, x_glove, x_lens):
        lstm_out_w2v, _ = self.lstm_w2v(x_w2v)
        lstm_out_glove, _ = self.lstm_glove(x_glove)

        # concatenate the results 'lstm_w2v' and 'lstm_glove'
        lstms_concat = torch.cat((lstm_out_w2v, lstm_out_glove), dim=2)

        lstm_out_total, _ = self.lstm_total(lstms_concat)

        # concatenate all the non-padded words h states vectors to a single matrix
        lstm_out_total = torch.cat([matrix[torch.arange(x_len)] for matrix, x_len in zip(lstm_out_total, x_lens)])

        output = self.hidden2tag(lstm_out_total)

        prediction_probs = self.sigmoid(output)

        return torch.squeeze(prediction_probs)

def train_model_comp(model, train_loader, device):
    """Trains a given model on the same batch where each sample is assigned to a random binary label,
    and returns a list of train + test loss in every epoch."""

    # get loss and optimization functions
    f_loss = nn.BCELoss()
    f_optim = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS+1):

        # print current epoch
        if epoch == 1 or epoch % 2 == 0:
            print(f"Processing epoch [{epoch}/{NUM_EPOCHS}]")

        for train_x_w2v, train_x_glove, train_x_lens, train_y in train_loader:
            # upload train batch to GPU if available
            train_x_w2v, train_x_glove = train_x_w2v.to(device), train_x_glove.to(device)
            train_x_lens, train_y = train_x_lens.to(device) ,train_y.to(device)

            # get probability predications
            train_y_prob_pred = model(train_x_w2v, train_x_glove, train_x_lens)

            # flat train labels
            train_y_flatten = torch.cat([sen_labels[torch.arange(sen_len)] for sen_labels, sen_len in zip(train_y, train_x_lens)])


            # get loss
            train_y_flatten, train_y_prob_pred = train_y_flatten.to(device), train_y_prob_pred.to(device)
            train_loss = f_loss(train_y_prob_pred, train_y_flatten)

            # back prop
            f_optim.zero_grad()
            train_loss.backward()
            f_optim.step()


def get_preds_model_comp(model, data_loader, device):
    prob_pred_lst = []

    model.eval()

    with torch.no_grad():
        for x_w2v, x_glove, x_lens in data_loader:
            # upload train batch to GPU if available
            x_w2v, x_glove, x_lens = x_w2v.to(device), x_glove.to(device), x_lens.to(device)

            prob_pred = model(x_w2v, x_glove, x_lens)
            prob_pred_lst += prob_pred.tolist()

    # convert probability predictions into binary labels
    y_pred_lst = np.round(np.array(prob_pred_lst))
    return y_pred_lst


def remove_O_sentences_manually(sentences, tags):
    clean_sentences = []
    clean_tags = []

    for sentence, sentence_tags in zip(sentences, tags):
        # check if all tags are 0 (or -1 for padding)
        if not 1 in sentence_tags:
            clean_sentences.append(sentence)
            clean_tags.append(sentence_tags)

    return clean_sentences, clean_tags


def concatenate_arrays(arr1, arr2, dtype):
    concatenated_list = list(arr1) + list(arr2)
    return np.array(concatenated_list, dtype=dtype)

def cross_validation():
    # extract train sentences + labels, and REMOVE "O" sentences
    train_sentences, train_tags = extract_data("data/train.tagged", is_tagged=True,
                                                               remove_O_sentences=True)

    # extract dev sentences + labels, and KEEP "O" sentences
    dev_sentences, dev_tags = extract_data("data/dev.tagged", is_tagged=True,
                                                           remove_O_sentences=False)

    split_idx = len(dev_sentences) // 2

    dev_first_half_sentences, dev_second_half_sentences = dev_sentences[:split_idx], dev_sentences[split_idx:]
    dev_first_half_tags, dev_second_half_tags = dev_tags[:split_idx], dev_tags[split_idx:]

    ### fold 1:
    # in this fold we:
    # 1. train on 'train.tagged' + the first half of 'dev.tagged'
    # 2. test on the second half of 'dev.tagged'

    dev_first_half_sentences_no_O, dev_first_half_tags_no_O = remove_O_sentences_manually(dev_first_half_sentences, dev_first_half_tags)

    fold1_train_sentences = train_sentences + dev_first_half_sentences_no_O
    fold1_train_tags = train_tags + dev_first_half_tags_no_O

    fold1_test_sentences = dev_second_half_sentences
    fold1_test_tags = dev_second_half_tags

    # convert train sentences to embeddings
    fold1_train_x_w2v, fold1_train_x_glove, fold1_train_x_lens, fold1_train_y = double_embedding(fold1_train_sentences, fold1_train_tags,
                                                                                                 is_tagged=True, label_type="binary",
                                                                                                 flatten=False)

    # convert dev sentences to embeddings
    fold1_test_x_w2v, fold1_test_x_glove, fold1_test_x_lens, fold1_test_y = double_embedding(fold1_test_sentences, fold1_test_tags,
                                                                                             is_tagged=True, label_type="binary",
                                                                                             flatten=False)

    # run the model on fold 1
    fold1_f1 = train_and_evaluate_model_comp(fold1_train_x_w2v, fold1_train_x_glove, fold1_train_x_lens, fold1_train_y,
                                             fold1_test_x_w2v, fold1_test_x_glove, fold1_test_x_lens, fold1_test_y)

    ### fold 2:
    # in this fold we:
    # 1. train on 'train.tagged' + the second half of 'dev.tagged'
    # 2. test on the first half of 'dev.tagged'

    dev_second_half_sentences_no_O, dev_second_half_tags_no_O = remove_O_sentences_manually(dev_second_half_sentences, dev_second_half_tags)

    fold2_train_sentences = train_sentences + dev_second_half_sentences_no_O
    fold2_train_tags = train_tags + dev_second_half_tags_no_O

    fold2_test_sentences = dev_first_half_sentences
    fold2_test_tags = dev_first_half_tags

    # convert train sentences to embeddings
    fold2_train_x_w2v, fold2_train_x_glove, fold2_train_x_lens, fold2_train_y = double_embedding(fold2_train_sentences,
                                                                                                 fold2_train_tags,
                                                                                                 is_tagged=True,
                                                                                                 label_type="binary",
                                                                                                 flatten=False)

    # convert dev sentences to embeddings
    fold2_test_x_w2v, fold2_test_x_glove, fold2_test_x_lens, fold2_test_y = double_embedding(fold2_test_sentences,
                                                                                             fold2_test_tags,
                                                                                             is_tagged=True,
                                                                                             label_type="binary",
                                                                                             flatten=False)

    # run the model on fold 1
    fold2_f1 = train_and_evaluate_model_comp(fold2_train_x_w2v, fold2_train_x_glove, fold2_train_x_lens, fold2_train_y,
                                             fold2_test_x_w2v, fold2_test_x_glove, fold2_test_x_lens, fold2_test_y)

    # return average f1 score across the two folds
    return (fold1_f1 + fold2_f1)/2

def initialize_model_comp(train_x_w2v, train_x_glove, train_x_lens, train_y, test_x_w2v,  test_x_glove, test_x_lens, test_y):

    train_loader = get_data_loader(x_w2v=train_x_w2v, x_glove=train_x_glove, x_lens=train_x_lens, y=train_y,
                                   is_flatten=False, is_double=True, is_test=False, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = get_data_loader(x_w2v=test_x_w2v, x_glove=test_x_glove, x_lens=test_x_lens, y=test_y,
                                  is_flatten=False, is_double=True, is_test=True, batch_size=BATCH_SIZE, shuffle=False)

    # get device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # get model
    lstm = MyLSTM(W2V_DIM, GLOVE_DIM, hidden_dim=HIDDEN_SIZE).to(device)

    return lstm, train_loader, test_loader, device

def evaluate_model_comp(test_y, test_y_preds, file_name):
    f1 = f1_score(test_y, test_y_preds)
    print(f"\nF1 Score for the Competition Model on '{file_name}': {round(f1, 3)}\n")
    return f1

def train_and_evaluate_model_comp(train_x_w2v, train_x_glove, train_x_lens, train_y, test_x_w2v,  test_x_glove, test_x_lens, test_y,
                                  test_file="dev.tagged", perform_cv=False):

    if perform_cv:
        avg_f1 = cross_validation()
        print(f"\nAverage F1 Score in Cross Validation: {round(avg_f1, 3)}\n")
        return avg_f1

    lstm, train_loader, test_loader, device = initialize_model_comp(train_x_w2v, train_x_glove, train_x_lens, train_y, test_x_w2v,
                                                                    test_x_glove, test_x_lens, test_y)

    train_model_comp(lstm, train_loader, device)

    # evaluate
    test_y_preds = get_preds_model_comp(lstm, test_loader, device)
    test_y = np.concatenate([sen_labels[np.arange(sen_len)] for sen_labels, sen_len in zip(test_y, test_x_lens)])
    test_f1 = evaluate_model_comp(test_y, test_y_preds, test_file)

    return test_f1

