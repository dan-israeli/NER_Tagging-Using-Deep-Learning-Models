from torch.utils.data import Dataset
from string import punctuation
import gensim.downloader
from abc import ABC
import numpy as np
import random
import torch

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

TOTAL_DIM = 500
W2V_DIM = 300
GLOVE_DIM = 200


TAG2NUM = { 'O': 0,
            'I-product': 1,
            'B-group': 2,
            'B-person': 3,
            'B-corporation': 4,
            'I-person': 5,
            'I-corporation': 6,
            'I-creative-work': 7,
            'B-product': 8,
            'I-group': 9,
            'B-creative-work': 10,
            'B-location': 11,
            'I-location': 12 }


class MyDataset(Dataset, ABC):
    def __init__(self, x, x_w2v, x_glove, x_lens, y, is_flatten, is_double, is_test):

        self.is_flatten = is_flatten
        self.is_double = is_double
        self.is_test = is_test

        if not is_double and is_flatten:
            self.x = x
            self.y = y

        if not is_double and not is_flatten:
            self.x = x
            self.x_lens = x_lens
            self.y = y


        elif is_double:
            self.x_w2v = x_w2v
            self.x_glove = x_glove
            self.x_lens = x_lens
            self.y = y

    def __len__(self):
        if not self.is_double:
            return self.x.shape[0]

        return self.x_w2v.shape[0]

    def __getitem__(self, item):

        if not self.is_double and self.is_flatten:
            if self.is_test:
                return self.x[item]
            else:
                return self.x[item], self.y[item]

        if not self.is_double and not self.is_flatten:
            if self.is_test:
                return self.x[item], self.x_lens[item]
            else:
                return self.x[item], self.x_lens[item], self.y[item]

        if self.is_double:
            if self.is_test:
                return self.x_w2v[item], self.x_glove[item], self.x_lens[item]
            else:
                return self.x_w2v[item], self.x_glove[item], self.x_lens[item], self.y[item]

def get_data_loader(x=None, x_w2v=None, x_glove=None, x_lens=None, y=None, is_flatten=True, is_double=False, is_test=False, batch_size=32, shuffle=True):

    dataset = MyDataset(x, x_w2v, x_glove, x_lens, y, is_flatten, is_double, is_test)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def flatten_list(deep_lst):
    flat_lst = []
    for inner_lst in deep_lst:
        flat_lst += list(inner_lst)

    return np.array(flat_lst)

def get_w2v():
    w2v_dict = gensim.downloader.load('word2vec-google-news-300')
    return w2v_dict


def get_glove():
    glove_dict = gensim.downloader.load('glove-twitter-200')
    return glove_dict

def extract_data(path, is_tagged, remove_O_sentences):
    sentences_words = []
    sentences_tags = []

    sentence_words = []
    sentence_tags = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]

            if is_tagged and (line == '\t' or line == ''):
                # only include sentences which have a "non O" tag
                if remove_O_sentences:
                    if not (len(set(sentence_tags)) == 1 and "O" in sentence_tags):
                        sentences_words.append(sentence_words)
                        sentences_tags.append(sentence_tags)
                else:
                    sentences_words.append(sentence_words)
                    sentences_tags.append(sentence_tags)

                sentence_words, sentence_tags = [], []
                continue

            elif not is_tagged and line == '':
                sentences_words.append(sentence_words)

                sentence_words = []
                continue

            if is_tagged:
                word, tag = line.split("\t")
                sentence_words.append(word)
                sentence_tags.append(tag)

            else:
                word = line
                sentence_words.append(word)

    return sentences_words, sentences_tags

def get_max_sentence_len(sentences_words):
    max_len = 0
    for sen in sentences_words:
        max_len = max(max_len, len(sen))

    return max_len

def get_onehot_vec(idx, dim):
    onehot_vec = np.zeros(dim)
    onehot_vec[idx] = 1

    return onehot_vec

def get_structured_embedding(embedding_model):

    structured_embedding = {}

    structured_embedding_type = ['starts_with_@', 'starts_with_#', 'link', 'contains_only_letters', 'contains_only_letters_and_punc',
                                  'contains_letters_punc_and_nums', 'contains_only_letters_and_nums', 'contains_only_punc',
                                  'contains_only_punc_and_nums', "contains_no_letters_punc_and_nums"]
    if embedding_model == "w2v":
        structured_embedding = {t: get_onehot_vec(i, W2V_DIM) for i, t in
                                enumerate(structured_embedding_type, start=1)}

    elif embedding_model == "glove":
        structured_embedding = {t: get_onehot_vec(i, GLOVE_DIM) for i, t in
                                enumerate(structured_embedding_type, start=1)}


    return structured_embedding


def get_embedding_vector(word, embedding_dict, structured_embeddings_dict):
    if word in embedding_dict:
        return embedding_dict[word]

    if structured_embeddings_dict is None:
        embedding_dim = len(embedding_dict["the"])
        return np.zeros(embedding_dim)

    if word[0] == '@' and len(word) > 1:
        return structured_embeddings_dict["starts_with_@"]

    if word[0] == '#' and len(word) > 1:
        return structured_embeddings_dict["starts_with_#"]

    if word[:3] == 'www' or word[:4] == 'http':
        return structured_embeddings_dict["link"]

    if str.lower(word) != str.upper(word) and all(c not in punctuation for c in word) and all(not c.isdigit() for c in word):
        return structured_embeddings_dict["contains_only_letters"]

    if str.lower(word) != str.upper(word) and any(c in punctuation for c in word) and all(not c.isdigit() for c in word):
        return structured_embeddings_dict["contains_only_letters_and_punc"]

    if str.lower(word) != str.upper(word) and any(c in punctuation for c in word) and any(c.isdigit() for c in word):
        return structured_embeddings_dict["contains_letters_punc_and_nums"]

    if str.lower(word) != str.upper(word) and all(c not in punctuation for c in word) and any(c.isdigit() for c in word):
        return structured_embeddings_dict["contains_only_letters_and_nums"]

    if str.lower(word) == str.upper(word) and all(not c.isdigit() for c in word):
        return structured_embeddings_dict["contains_only_punc"]

    if str.lower(word) == str.upper(word) and any(c.isdigit() for c in word):
        return structured_embeddings_dict["contains_only_punc_and_nums"]

    if str.lower(word) == str.upper(word) and all(c not in punctuation for c in word) and all(not c.isdigit() for c in word):
        return structured_embeddings_dict["contains_no_letters_punc_and_nums"]

    embedding_dim = embedding_dict["the"].shape[0]
    return np.zeros(embedding_dim)

def pad_sentence(sentence, max_len, input_type, dim=300):
    # the padding of a word in a sentence are zeros vector
    if input_type == "word":
        padded_sentence = np.zeros((max_len, dim))
        padded_sentence[:len(sentence)] = sentence

    # the padding of a label in a sentence are -1's
    if input_type == "label":
        padded_sentence = np.zeros(max_len) - 1
        padded_sentence[:len(sentence)] = sentence

    return padded_sentence

def get_label_from_tag(tag, label_type):
    if label_type == "binary":
        if tag == 'O':
            label = 0
        else:
            label = 1

    elif label_type == "multiclass":
        label = TAG2NUM[tag]

    return label


def get_word_embeddings(sentences_words, embedding_dict, structured_embeddings_dict, flatten, max_len):
    x, x_lens = [], []

    # flat all the words into a single list (remove "sentence context" between them)
    if flatten:
        # flatten list
        flatten_sentences_words = flatten_list(sentences_words)

        for word in flatten_sentences_words:
            w_vector = get_embedding_vector(word, embedding_dict, structured_embeddings_dict)
            x.append(w_vector)

    # keep sentence context between the words
    else:
        # get embedding dim
        embedding_dim = len(embedding_dict["the"])

        for sen in sentences_words:
            sen_vectors = []

            for w in sen:
                w_vector = get_embedding_vector(w, embedding_dict, structured_embeddings_dict)
                sen_vectors.append(w_vector)

            sen_matrix = np.array(sen_vectors, dtype=np.float32)

            # get current sentence length
            x_lens.append(sen_matrix.shape[0])

            # pad sentence to the length of the longest sentence in 'sentences_words'
            padded_sen_matrix = pad_sentence(sen_matrix, max_len, input_type="word", dim=embedding_dim)
            x.append(padded_sen_matrix)

    return np.array(x, dtype=np.float32), np.array(x_lens, dtype=np.intc)


def get_labels(sentences_tags, is_tagged, label_type, flatten, max_len):
    y = []

    if not is_tagged:
        return y

    if flatten:
        flatten_sentences_tags = flatten_list(sentences_tags)
        for tag in flatten_sentences_tags:
            label = get_label_from_tag(tag, label_type)
            y.append(label)

    else:
        for sen_tags in sentences_tags:
            sen_labels = []
            for tag in sen_tags:
                label = get_label_from_tag(tag, label_type)
                sen_labels.append(label)

            padded_sen_labels = pad_sentence(sen_labels, max_len, input_type="label")
            y.append(padded_sen_labels)

    return np.array(y, dtype=np.float32)



def single_embedding(sentences_words, sentences_tags, embedding_model, is_tagged, label_type, flatten):

    embedding_dict, structured_embedding_dict = {}, {}
    max_len = get_max_sentence_len(sentences_words)

    if embedding_model == "w2v":
        embedding_dict = get_w2v()
        # structured_embedding_dict = get_structured_embedding(embedding_model="w2v")

    elif embedding_model == "glove":
        embedding_dict = get_glove()
        # structured_embedding_dict = get_structured_embedding(embedding_model="glove")

    structured_embedding_dict = None

    x, x_lens = get_word_embeddings(sentences_words, embedding_dict, structured_embedding_dict, flatten, max_len)
    y = get_labels(sentences_tags, is_tagged, label_type, flatten, max_len)

    return x, x_lens, y


def double_embedding(sentences_words, sentences_tags, is_tagged, label_type, flatten):

    w2v_embedding_dict = get_w2v()
    w2v_structured_embedding_dict = get_structured_embedding(embedding_model="w2v")

    glove_embedding_dict = get_glove()
    glove_structured_embedding_dict = get_structured_embedding(embedding_model="glove")

    max_len = get_max_sentence_len(sentences_words)

    x_w2v, x_lens = get_word_embeddings(sentences_words, w2v_embedding_dict, w2v_structured_embedding_dict, flatten, max_len)
    x_glove, _ = get_word_embeddings(sentences_words, glove_embedding_dict, glove_structured_embedding_dict, flatten, max_len)
    y = get_labels(sentences_tags, is_tagged, label_type, flatten, max_len)

    return x_w2v, x_glove, x_lens, y


def prepare_data(path, embedding_model, is_tagged, remove_O_sentences, label_type, flatten):

    # extract data
    sentences_words, sentences_tags = extract_data(path, is_tagged, remove_O_sentences)

    if embedding_model in ["w2v", "glove"]:
        x, x_lens, y = single_embedding(sentences_words, sentences_tags, embedding_model, is_tagged, label_type, flatten)
        return x, x_lens, y

    if embedding_model == "w2v+glove":
        x_w2v, x_glove, x_lens, y = double_embedding(sentences_words, sentences_tags, is_tagged, label_type, flatten)
        return x_w2v, x_glove, x_lens, y
