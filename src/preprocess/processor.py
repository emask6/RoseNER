import os
import re
import json
import logging
from transformers import BertTokenizer
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

ENTITY_TYPES = ['Chemical', 'Disease']
class InputExample:
    def __init__(self,
                 set_type,
                 text,
                 labels=None,
                 pseudo=None,
                 distant_labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels
        self.pseudo = pseudo
        self.distant_labels = distant_labels


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class SpanFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 start_ids=None,
                 end_ids=None,
                 pseudo=None):
        super(SpanFeature, self).__init__(token_ids=token_ids,
                                          attention_masks=attention_masks,
                                          token_type_ids=token_type_ids)
        self.start_ids = start_ids
        self.end_ids = end_ids
        # pseudo
        self.pseudo = pseudo

class NERProcessor:
    def __init__(self, cut_sent_len=256):
        self.cut_sent_len = cut_sent_len

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = json.load(f)
        return raw_examples

    @staticmethod
    def _refactor_labels(sent, labels, distant_labels, start_index):
        """
        分句后需要重构 labels 的 offset
        :param sent: 切分并重新合并后的句子
        :param labels: 原始文档级的 labels
        :param distant_labels: 远程监督 label
        :param start_index: 该句子在文档中的起始 offset
        :return (type, entity, offset)
        """
        new_labels, new_distant_labels = [], []
        end_index = start_index + len(sent.split(' '))
        for _label in labels:
            if start_index <= _label[2] <= _label[3] <= end_index:
                new_offset = _label[2] - start_index
                # print('#########')
                # print(sent)
                # print(start_index,_label[2],_label[3],end_index)
                # print(new_offset, new_offset + len(_label[-1].split(' ')))
                # print(len(sent.split(' ')))
                # print(' '.join(sent.split(' ')[new_offset: new_offset + len(_label[-1].split(' '))] ))
                # print(_label[-1])
                assert ' '.join(sent.split(' ')[new_offset: new_offset + len(_label[-1].split(' '))] )== _label[-1]

                new_labels.append((_label[1], _label[-1], new_offset))
            # label 被截断的情况
            elif _label[2] < end_index < _label[3]:
                continue

        for _label in distant_labels:
            if _label in sent:
                new_distant_labels.append(_label)

        return new_labels, new_distant_labels

    def get_examples(self, raw_examples, set_type):
        examples = []

        for i, item in enumerate(raw_examples):
            text = item['text']
            distant_labels = item['candidate_entities']
            pseudo = item['pseudo']

            sentences = cut_sent(text, self.cut_sent_len)

            start_index = 0
            # print(len(sentences))

            for sent in sentences:
                labels, tmp_distant_labels = self._refactor_labels(sent, item['labels'], distant_labels, start_index)

                start_index += len(sent.split(' '))

                examples.append(InputExample(set_type=set_type,
                                             text=sent,
                                             labels=labels,
                                             pseudo=pseudo,
                                             distant_labels=tmp_distant_labels))

        return examples


def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens


# def cut_sentences_v1(sent):
#     """
#     the first rank of sentence cut
#     """
#     sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", sent)  # 单字符断句符
#     sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号
#     sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
#     sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
#     # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
#     return sent.split("\n")
#
#
# def cut_sentences_v2(sent):
#     """
#     the second rank of spilt sentence, split '；' | ';'
#     """
#     sent = re.sub('([；;])([^”’])', r"\1\n\2", sent)
#     return sent.split("\n")


def cut_sent(text, max_seq_len):
    # 将句子分句，细粒度分句后再重新合并
    sentences = []
    # sentences.append(text)
    if len(text.split(' ')) > max_seq_len-2:
        sentences_size = len(text.split(' '))
        start_index = 0
        while sentences_size > 0:
            sentences.append(' '.join(text.split(' ')[start_index:start_index+max_seq_len-1]))
            start_index += max_seq_len-1
            sentences_size = sentences_size - max_seq_len-1
    else:
        sentences.append(text)
    # 细粒度划分
    # sentences_v1 = cut_sentences_v1(text)
    # print(text)
    # print(sentences_v1)
    # print('---------------')
    # for sent_v1 in sentences_v1:
    #     if len(sent_v1.split(' ')) > max_seq_len - 2:
    #         sentences_v2 = cut_sentences_v2(sent_v1)
    #         sentences.extend(sentences_v2)
    #     else:
    #         sentences.append(sent_v1)
    #
    # assert ''.join(sentences) == text

    # 合并
    merged_sentences = []
    start_index_ = 0
    # print(len(sentences))
    while start_index_ < len(sentences):
        tmp_text = sentences[start_index_]

        end_index_ = start_index_ + 1

        while end_index_ < len(sentences) and \
                len(tmp_text.split(' ')) + len(sentences[end_index_]) <= max_seq_len - 2:
            tmp_text += sentences[end_index_]
            end_index_ += 1

        start_index_ = end_index_

        merged_sentences.append(tmp_text)
    # print(len(merged_sentences))
    # print('-----')
    # print(merged_sentences)
    # if len(merged_sentences)==2:
    #     print(merged_sentences[0])
    #     print(merged_sentences[1])
    #     print('---------')
    return merged_sentences


def convert_span_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                         max_seq_len, ent2id):
    set_type = example.set_type
    raw_text = example.text
    entities = example.labels
    pseudo = example.pseudo

    tokens = fine_grade_tokenize(raw_text, tokenizer)
    assert len(tokens) == len(raw_text)

    callback_labels = {x: [] for x in ENTITY_TYPES}

    for _label in entities:
        callback_labels[_label[0]].append((_label[1], _label[2]))

    callback_info = (raw_text, callback_labels,)

    start_ids, end_ids = None, None

    if set_type == 'train':
        start_ids = [0] * len(tokens)
        end_ids = [0] * len(tokens)

        for _ent in entities:

            ent_type = ent2id[_ent[0]]
            ent_start = _ent[-1]
            ent_end = ent_start + len(_ent[1]) - 1

            start_ids[ent_start] = ent_type
            end_ids[ent_end] = ent_type

        if len(start_ids) > max_seq_len - 2:
            start_ids = start_ids[:max_seq_len - 2]
            end_ids = end_ids[:max_seq_len - 2]

        start_ids = [0] + start_ids + [0]
        end_ids = [0] + end_ids + [0]

        # pad
        if len(start_ids) < max_seq_len:
            pad_length = max_seq_len - len(start_ids)

            start_ids = start_ids + [0] * pad_length  # CLS SEP PAD label都为O
            end_ids = end_ids + [0] * pad_length

        assert len(start_ids) == max_seq_len
        assert len(end_ids) == max_seq_len

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']


    feature = SpanFeature(token_ids=token_ids,
                          attention_masks=attention_masks,
                          token_type_ids=token_type_ids,
                          start_ids=start_ids,
                          end_ids=end_ids,
                          pseudo=pseudo)

    return feature, callback_info


def convert_examples_to_features(task_type, examples, max_seq_len, bert_dir, ent2id):
    assert task_type in ['crf', 'span', 'mrc', 'lstm_crf']
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))

    features = []

    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')
    type2id = {x: i for i, x in enumerate(ENTITY_TYPES)}

    for i, example in enumerate(examples):

        if task_type == 'span':
            feature, tmp_callback = convert_span_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                ent2id=ent2id,
                tokenizer=tokenizer
            )

        if feature is None:
            continue

        if task_type == 'mrc':
            features.extend(feature)
            callback_info.extend(tmp_callback)
        else:
            features.append(feature)
            callback_info.append(tmp_callback)

    logger.info(f'Build {len(features)} features')

    out = (features, )

    if not len(callback_info):
        return out

    type_weight = {}  # 统计每一类的比例，用于计算 micro-f1
    for _type in ENTITY_TYPES:
        type_weight[_type] = 0.

    count = 0.

    if task_type == 'mrc':
        for _callback in callback_info:
            type_weight[_callback[-2]] += len(_callback[-1])
            count += len(_callback[-1])
    else:
        for _callback in callback_info:
            for _type in _callback[1]:
                type_weight[_type] += len(_callback[1][_type])
                count += len(_callback[1][_type])

    for key in type_weight:
        type_weight[key] /= count

    out += ((callback_info, type_weight), )

    return out


if __name__ == '__main__':
    pass
