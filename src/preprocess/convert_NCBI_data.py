import os
import json
from tqdm import trange
from sklearn.model_selection import train_test_split, KFold


def save_info(data_dir, data, desc):
    with open(os.path.join(data_dir, f'{desc}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def convert_data_to_json(base_dir, save_data=False, save_dict=False):
    stack_examples = []
    pseudo_examples = []
    dev_examples = []
    test_examples = []

    # stack_dir = os.path.join(base_dir, 'BC5CDR')
    # pseudo_dir = os.path.join(base_dir, 'pseudo')
    # test_dir = os.path.join(base_dir, 'BC5CDR')
    # dev_dir = os.path.join(base_dir, 'BC5CDR')

    text_data = []
    label_data = []
    with open(os.path.join(base_dir, 'train.tsv'), 'r') as f:
        text = ''
        label = []
        flag = 0
        for line in f.readlines():
            if len(line.split('\t')) == 2:
                line = line.replace('\n', '')
                if text == '':
                    text = line.split('\t')[0]
                else:
                    text += ' ' + line.split('\t')[0]
                label.append(line.split('\t')[1])
                flag += 1
            else:
                text_data.append(text)
                label_data.append(' '.join(label))
                text = ''
                label = []

    X = text_data
    Y = label_data

    for idx, x in enumerate(Y):
        B_index = []
        E_index = []
        S_index = []
        label = x.split(' ')
        for ids, l in enumerate(label):
            if l == 'B':
                if ids == len(label) - 1 or label[ids + 1] == 'O' or label[ids + 1] == 'B':
                    S_index.append(ids)
                    continue
                if label[ids + 1] == 'I':
                    B_index.append(ids)
                    continue
            if l == 'I':
                if ids == len(label) - 1:
                    E_index.append(ids)
                    continue
                if label[ids + 1] == 'O' or label[ids + 1] == 'B':
                    E_index.append(ids)
        text = X[idx]
        labels = []
        for index, B in enumerate(B_index):
            labels.append(
                ['T' + str(index)] + ['Disease'] + [B] + [E_index[index] + 1] + [
                    ' '.join(text.split(' ')[B:E_index[index] + 1])])
        for index, S in enumerate(S_index):
            labels.append(
                ['T' + str(index + len(B_index))] + ['Disease'] + [S] + [S + 1] + [
                    ' '.join(text.split(' ')[S:S + 1])])
        stack_examples.append({'id': idx,
                               'text': text,
                               'labels': labels,
                               'pseudo': 0})

    kf = KFold(10, random_state=626)
    entities = set()
    ent_types = set()
    for _now_id, _candidate_id in kf.split(stack_examples):
        now = [stack_examples[_id] for _id in _now_id]
        candidate = [stack_examples[_id] for _id in _candidate_id]
        now_entities = set()

        for _ex in now:
            for _label in _ex['labels']:
                ent_types.add(_label[1])

                if len(_label[-1]) > 1:
                    now_entities.add(_label[-1])
                    entities.add(_label[-1])
        # print(len(now_entities))
        for _ex in candidate:
            text = _ex['text']
            candidate_entities = []

            for _ent in now_entities:
                if _ent in text:
                    candidate_entities.append(_ent)

            _ex['candidate_entities'] = candidate_entities
    print(len(ent_types))
    # # process test examples predicted by the preliminary model
    # text_data = []
    # label_data = []
    # B_index = []
    # E_index = []
    # S_index = []
    # times = 0
    # with open(os.path.join(pseudo_dir, 'pseudo.txt'), 'r', encoding='utf-8') as f:
    #     text = ''
    #     label = []
    #     B = []
    #     E = []
    #     S = []
    #     flag = 0
    #     bio = []
    #     for line in f.readlines():
    #         if len(line.split(' ')) == 2:
    #             line = line.replace('\n', '')
    #             text += line.split(' ')[0]
    #             bio.append(line.split(' ')[1])
    #             if line.split(' ')[1].split('-')[0] == 'B':
    #                 B.append(flag)
    #                 label.append(line.split(' ')[1])
    #             if line.split(' ')[1].split('-')[0] == 'E':
    #                 E.append(flag)
    #             if line.split(' ')[1].split('-')[0] == 'S':
    #                 S.append(flag)
    #                 label.append(line.split(' ')[1])
    #             flag += 1
    #         else:
    #             times += 1
    #             text_data.append(text)
    #             label_data.append(label)
    #
    #             B_index.append(B)
    #             E_index.append(E)
    #             S_index.append(S)
    #             text = ''
    #             label = []
    #             flag = 0
    #             B = []
    #             E = []
    #             S = []
    #             bio = []
    # times = 0
    # last = 0
    # for ids, text in enumerate(text_data):
    #     labels = []
    #     if len(B_index[ids]) == len(E_index[ids]):
    #         for index, B in enumerate(B_index[ids]):
    #             labels.append(
    #                 ['T' + str(index)] + [label_data[ids][index].split('-')[1]] + [B] + [E_index[ids][index] + 1] + [
    #                     text[B:E_index[ids][index] + 1]])
    #         for index, S in enumerate(S_index[ids]):
    #             labels.append(['T' + str(index + len(B_index[ids]) + 1)] + [label_data[ids][index].split('-')[1]]
    #                           + [S] + [S + 1] + [text[S:S + 1]])
    #         candidate_entities = []
    #         for _ent in entities:
    #             if _ent in text:
    #                 candidate_entities.append(_ent)
    #         pseudo_examples.append({'id': last,
    #                                 'text': text,
    #                                 'labels': labels,
    #                                 'candidate_entities': candidate_entities,
    #                                 'pseudo': 1})
    #         last+=1
    #     else:
    #         times+=1
    # print(times)
    text_data = []
    label_data = []
    with open(os.path.join(base_dir, 'devel.tsv'), 'r') as f:
        text = ''
        label = []
        flag = 0
        for line in f.readlines():
            if len(line.split('\t')) == 2:
                line = line.replace('\n', '')
                if text == '':
                    text = line.split('\t')[0]
                else:
                    text += ' ' + line.split('\t')[0]
                label.append(line.split('\t')[1])
                flag += 1
            else:
                text_data.append(text)
                label_data.append(' '.join(label))
                text = ''
                label = []

    X = text_data
    Y = label_data

    for idx, x in enumerate(Y):
        B_index = []
        E_index = []
        S_index = []
        label = x.split(' ')
        for ids, l in enumerate(label):
            if l == 'B':
                if ids == len(label) - 1 or label[ids + 1] == 'O' or label[ids + 1] == 'B':
                    S_index.append(ids)
                    continue
                if label[ids + 1] == 'I':
                    B_index.append(ids)
                    continue
            if l == 'I':
                if ids == len(label) - 1:
                    E_index.append(ids)
                    continue
                if label[ids + 1] == 'O' or label[ids + 1] == 'B':
                    E_index.append(ids)
        text = X[idx]
        labels = []
        for index, B in enumerate(B_index):
            labels.append(
                ['T' + str(index)] + ['Disease'] + [B] + [E_index[index] + 1] + [
                    ' '.join(text.split(' ')[B:E_index[index] + 1])])
        for index, S in enumerate(S_index):
            labels.append(
                ['T' + str(index + len(B_index))] + ['Disease'] + [S] + [S + 1] + [
                    ' '.join(text.split(' ')[S:S + 1])])
        candidate_entities = []
        for _ent in entities:
            if _ent in text:
                candidate_entities.append(_ent)
        dev_examples.append({'id': idx,
                               'text': text,
                               'labels': labels,
                                'candidate_entities': candidate_entities,
                               'pseudo': 0})
    # process test examples
    text_data = []
    label_data = []
    with open(os.path.join(base_dir, 'test.tsv'), 'r') as f:
        text = ''
        label = []
        flag = 0
        for line in f.readlines():
            if len(line.split('\t')) == 2:
                line = line.replace('\n', '')
                if text == '':
                    text = line.split('\t')[0]
                else:
                    text += ' ' + line.split('\t')[0]
                label.append(line.split('\t')[1])
                flag += 1
            else:
                text_data.append(text)
                label_data.append(' '.join(label))
                text = ''
                label = []

    X = text_data
    Y = label_data

    for idx, x in enumerate(Y):
        B_index = []
        E_index = []
        S_index = []
        label = x.split(' ')
        for ids, l in enumerate(label):
            if l == 'B':
                if ids == len(label) - 1 or label[ids + 1] == 'O' or label[ids + 1] == 'B':
                    S_index.append(ids)
                    continue
                if label[ids + 1] == 'I':
                    B_index.append(ids)
                    continue
            if l == 'I':
                if ids == len(label) - 1:
                    E_index.append(ids)
                    continue
                if label[ids + 1] == 'O' or label[ids + 1] == 'B':
                    E_index.append(ids)
        text = X[idx]
        labels = []
        for index, B in enumerate(B_index):
            labels.append(
                ['T' + str(index)] + ['Disease'] + [B] + [E_index[index] + 1] + [
                    ' '.join(text.split(' ')[B:E_index[index] + 1])])
        for index, S in enumerate(S_index):
            labels.append(
                ['T' + str(index + len(B_index))] + ['Disease'] + [S] + [S + 1] + [
                    ' '.join(text.split(' ')[S:S + 1])])
        candidate_entities = []
        for _ent in entities:
            if _ent in text:
                candidate_entities.append(_ent)
        test_examples.append({'id': idx,
                             'text': text,
                             'labels': labels,
                              'candidate_entities': candidate_entities,
                             'pseudo': 0})

    train = stack_examples
    dev = dev_examples
    # 记得解开注释
    if save_data:
        save_info(base_dir, stack_examples, 'stack')
        save_info(base_dir, train, 'train')
        save_info(base_dir, dev, 'dev')
        save_info(base_dir, test_examples, 'test')

        # save_info(base_dir, pseudo_examples, 'pseudo')

    if save_dict:
        ent_types = list(ent_types)
        span_ent2id = {_type: i+1 for i, _type in enumerate(ent_types)}


        mid_data_dir = os.path.join(base_dir, 'mid_data')
        if not os.path.exists(mid_data_dir):
            os.mkdir(mid_data_dir)

        save_info(mid_data_dir, span_ent2id, 'span_ent2id')



if __name__ == '__main__':
    convert_data_to_json('../../data/NCBI-disease', save_data=True, save_dict=True)
