from model_comp import *
from utils import prepare_data

def merge_train_and_dev():
    with open("data/train.tagged", 'r', encoding='utf-8') as f1:
        content1 = f1.read()

    with open("data/dev.tagged", 'r', encoding='utf-8') as f2:
        content2 = f2.read()

    with open("data/full_train.tagged", 'w', encoding='utf-8') as output_file:
        output_file.write(content1)
        output_file.write(content2)


def create_comp_preds_file(preds):
    cur_idx = 0

    output_file = open('comp.tagged', 'w', encoding='utf-8')

    with open('data/test.untagged', 'r', encoding='utf-8') as f:
        for line in f:

            if line[-1:] == "\n":
                line = line[:-1]

            new_line = ''

            if line != '':
                new_line = line + '\t' + str(preds[cur_idx])
                cur_idx += 1

            new_line += '\n'
            output_file.write(new_line)

    output_file.close()


def main():
    merge_train_and_dev()

    train_x_w2v, train_x_glove, train_x_lens, train_y = prepare_data(path='data/train.tagged',
                                                                     embedding_model="w2v+glove", is_tagged=True,
                                                                     remove_O_sentences=True, label_type="binary",
                                                                     flatten=False)

    test_x_w2v, test_x_glove, test_x_lens, test_y = prepare_data(path='data/test.untagged', embedding_model="w2v+glove",
                                                                 is_tagged=False,
                                                                 remove_O_sentences=False, label_type="binary",
                                                                 flatten=False)

    comp_model, train_loader, test_loader, device = initialize_model_comp(train_x_w2v, train_x_glove, train_x_lens, train_y,
                                                                          test_x_w2v, test_x_glove, test_x_lens, test_y)

    train_model_comp(comp_model, train_loader, device)

    pred_labels = get_preds_model_comp(comp_model, test_loader, device)

    pred_tags = ["O" if pred_label == 0 else "N" for pred_label in pred_labels]

    create_comp_preds_file(pred_tags)


if __name__ == '__main__':
    main()
