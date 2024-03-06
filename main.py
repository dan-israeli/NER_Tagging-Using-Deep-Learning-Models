from utils import *
from model_1 import train_and_evaluate_model_1
from model_2 import train_and_evaluate_model_2
from model_3 import train_and_evaluate_model_3
from model_comp import train_and_evaluate_model_comp


def model_1():
    train_x, _, train_y = prepare_data(path='data/train.tagged', embedding_model="w2v", is_tagged=True,
                                       remove_O_sentences=False, label_type="binary", flatten=True)

    test_x, _, test_y = prepare_data(path='data/dev.tagged', embedding_model="w2v", is_tagged=True,
                                     remove_O_sentences=False, label_type="binary", flatten=True)

    # train model 1 on 'train.tagged' and evaluate it on both 'train.tagged' and 'dev.tagged'
    train_and_evaluate_model_1(train_x, train_y, test_x, test_y, k=1)


def model_2():
    train_x, _, train_y = prepare_data(path='data/train.tagged', embedding_model="w2v", is_tagged=True,
                                       remove_O_sentences=False, label_type="binary", flatten=True)

    test_x, _, test_y = prepare_data(path='data/dev.tagged', embedding_model="w2v", is_tagged=True,
                                     remove_O_sentences=False, label_type="binary", flatten=True)

    # train model 2 on 'train.tagged' and evaluate it on both 'train.tagged' and 'dev.tagged'
    train_and_evaluate_model_2(train_x, train_y, test_x, test_y)


def model_3():
    train_x, train_x_lens, train_y = prepare_data(path='data/train.tagged', embedding_model="w2v", is_tagged=True,
                                                  remove_O_sentences=False, label_type="binary", flatten=False)

    test_x, test_x_lens, test_y = prepare_data(path='data/dev.tagged', embedding_model="w2v", is_tagged=True,
                                               remove_O_sentences=False, label_type="binary", flatten=False)

    # train model 3 on 'train.tagged' and evaluate it on both 'train.tagged' and 'dev.tagged'
    train_and_evaluate_model_3(train_x, train_x_lens, train_y, test_x, test_x_lens, test_y)


def model_comp():
    train_x_w2v, train_x_glove, train_x_lens, train_y = prepare_data(path='data/train.tagged', embedding_model="w2v+glove",
                                                                     is_tagged=True, remove_O_sentences=True, label_type="binary",
                                                                     flatten=False)

    train_x_w2v_no_O, train_x_glove_no_0, train_x_lens_no_0, train_y_no_0 = prepare_data(path='data/train.tagged', embedding_model="w2v+glove",
                                                                                         is_tagged=True, remove_O_sentences=False, label_type="binary",
                                                                                         flatten=False)

    test_x_w2v, test_x_glove, test_x_lens, test_y = prepare_data(path='data/dev.tagged', embedding_model="w2v+glove",
                                                                 is_tagged=True, remove_O_sentences=False, label_type="binary",
                                                                 flatten=False)

    # train model_comp on 'train.tagged' and evaluate it on 'train.tagged'
    train_and_evaluate_model_comp(train_x_w2v, train_x_glove, train_x_lens, train_y,
                                  train_x_w2v_no_O, train_x_glove_no_0, train_x_lens_no_0, train_y_no_0,
                                  test_file='train.tagged')

    # train model_comp on 'train.tagged' and evaluate it on  'dev.tagged'
    train_and_evaluate_model_comp(train_x_w2v, train_x_glove, train_x_lens, train_y,
                                  test_x_w2v, test_x_glove, test_x_lens, test_y,
                                  test_file='dev.tagged')


def main():
    # model_1()  # uncomment to train model_1 on 'train.tagged' and evaluate it on 'train.tagged' and 'dev.tagged'
    # model_2()  # uncomment to train model_2 on 'train.tagged' and evaluate it on 'train.tagged' and 'dev.tagged'
    # model_3()  # uncomment to train model_3 on 'train.tagged' and evaluate it on 'train.tagged' and 'dev.tagged'
    # model_comp()  # uncomment to train model_comp on 'train.tagged' and evaluate it on 'train.tagged' and 'dev.tagged'
    pass

if __name__ == '__main__':
    main()
