import os
import json
import torch
from collections import defaultdict
from transformers import BertTokenizer
from src.utils.model_utils import SpanModel
from src.utils.evaluator import span_decode
from src.utils.functions_utils import load_model_and_parallel, ensemble_vote
from src.preprocess.processor import cut_sent, fine_grade_tokenize

MID_DATA_DIR = "data/BC5CDR/mid_data"
RAW_DATA_DIR = "data/BC5CDR"
SUBMIT_DIR = "./result"
GPU_IDS = "0"

LAMBDA = 0.3
THRESHOLD = 0.9
MAX_SEQ_LEN = 512

TASK_TYPE = "span"  # choose crf or span
VOTE = True  # choose True or False
VERSION = "single"  # choose single or ensemble or mixed ; if mixed  VOTE and TAST_TYPE is useless.

# single_predict
BERT_TYPE = "bert"  # roberta_wwm / ernie_1 / uer_large

BERT_DIR = f"./bert/torch_{BERT_TYPE}"
with open('./best_ckpt_path.txt', 'r', encoding='utf-8') as f:
    CKPT_PATH = f.read().strip()



def prepare_info():
    info_dict = {}
    with open(os.path.join(MID_DATA_DIR, f'{TASK_TYPE}_ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)

    with open(os.path.join(RAW_DATA_DIR, 'test.json'), encoding='utf-8') as f:
        info_dict['examples'] = json.load(f)

    info_dict['id2ent'] = {ent2id[key]: key for key in ent2id.keys()}

    info_dict['tokenizer'] = BertTokenizer(os.path.join(BERT_DIR, 'vocab.txt'))

    return info_dict




def base_predict(model, device, info_dict, ensemble=False, mixed=''):
    labels = defaultdict(list)

    tokenizer = info_dict['tokenizer']
    id2ent = info_dict['id2ent']

    with torch.no_grad():
        for _ex in info_dict['examples']:
            ex_idx = _ex['id']
            raw_text = _ex['text']

            if not len(raw_text):
                labels[ex_idx] = []
                print('{}为空'.format(ex_idx))
                continue

            sentences = cut_sent(raw_text, MAX_SEQ_LEN)

            start_index = 0

            for sent in sentences:

                sent_tokens = fine_grade_tokenize(sent, tokenizer)

                encode_dict = tokenizer.encode_plus(text=sent_tokens,
                                                    max_length=MAX_SEQ_LEN,
                                                    is_pretokenized=True,
                                                    pad_to_max_length=False,
                                                    return_tensors='pt',
                                                    return_token_type_ids=True,
                                                    return_attention_mask=True)

                model_inputs = {'token_ids': encode_dict['input_ids'],
                                'attention_masks': encode_dict['attention_mask'],
                                'token_type_ids': encode_dict['token_type_ids']}

                for key in model_inputs:
                    model_inputs[key] = model_inputs[key].to(device)

                if ensemble:
                    if TASK_TYPE == 'crf':
                        if VOTE:
                            decode_entities = model.vote_entities(model_inputs, sent, id2ent, THRESHOLD)
                        else:
                            pred_tokens = model.predict(model_inputs)[0]
                            # decode_entities = crf_decode(pred_tokens, sent, id2ent)
                    else:
                        if VOTE:
                            decode_entities = model.vote_entities(model_inputs, sent, id2ent, THRESHOLD)
                        else:
                            start_logits, end_logits = model.predict(model_inputs)
                            start_logits = start_logits[0].cpu().numpy()[1:1 + len(sent)]
                            end_logits = end_logits[0].cpu().numpy()[1:1 + len(sent)]

                            decode_entities = span_decode(start_logits, end_logits, sent, id2ent)

                else:

                    if mixed:

                        start_logits, end_logits = model(**model_inputs)

                        start_logits = start_logits[0].cpu().numpy()[1:1 + len(sent)]
                        end_logits = end_logits[0].cpu().numpy()[1:1 + len(sent)]

                        decode_entities = span_decode(start_logits, end_logits, sent, id2ent)

                    else:

                        start_logits, end_logits = model(**model_inputs)

                        start_logits = start_logits[0].cpu().numpy()[1:1+len(sent)]
                        end_logits = end_logits[0].cpu().numpy()[1:1+len(sent)]

                        decode_entities = span_decode(start_logits, end_logits, sent, id2ent)


                for _ent_type in decode_entities:
                    for _ent in decode_entities[_ent_type]:
                        tmp_start = _ent[1] + start_index
                        tmp_end = tmp_start + len(_ent[0])

                        assert raw_text[tmp_start: tmp_end] == _ent[0]

                        labels[ex_idx].append((_ent_type, tmp_start, tmp_end, _ent[0]))

                start_index += len(sent)

                if not len(labels[ex_idx]):
                    labels[ex_idx] = []

    return labels

def single_predict():
    save_dir = os.path.join(SUBMIT_DIR, VERSION)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    info_dict = prepare_info()


    model = SpanModel(bert_dir=BERT_DIR, num_tags=len(info_dict['id2ent'])+1)

    print(f'Load model from {CKPT_PATH}')
    model, device = load_model_and_parallel(model, GPU_IDS, CKPT_PATH)
    model.eval()

    labels = base_predict(model, device, info_dict)

    for key in labels.keys():
        with open(os.path.join(save_dir, f'{key}.txt'), 'w', encoding='utf-8') as f:
            if not len(labels[key]):
                print(key)
                f.write("")
            else:
                for idx, _label in enumerate(labels[key]):
                    f.write(str(_label[1])+'#'+str(_label[2])+'#'+_label[0]+'#'+_label[3]+'\n')
    f.close()




if __name__ == '__main__':
    assert VERSION in ['single'], 'VERSION mismatch'

    if VERSION == 'single':
        single_predict()


