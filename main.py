import time
import os
import json
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.utils.trainer import train
from src.utils.options import Args
from src.utils.model_utils import build_model
from src.utils.dataset_utils import NERDataset
from src.utils.evaluator import  span_evaluation
from src.utils.functions_utils import set_seed, get_model_path_list, load_model_and_parallel, get_time_dif
from src.preprocess.processor import NERProcessor, convert_examples_to_features

ENTITY_TYPES = ['Disease', 'Chemical']
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def train_base(opt, train_examples, dev_examples=None, test_examples=None):
    with open(os.path.join(opt.mid_data_dir, f'{opt.task_type}_ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)

    train_features = convert_examples_to_features(opt.task_type, train_examples,
                                                  opt.max_seq_len, opt.bert_dir, ent2id)[0]
    train_dataset = NERDataset(opt.task_type, train_features, 'train', use_type_embed=opt.use_type_embed)

    if opt.task_type == 'span':
        model = build_model('span', opt.bert_dir, num_tags=len(ent2id)+1,
                            dropout_prob=opt.dropout_prob,
                            loss_type=opt.loss_type)


    train(opt, model, train_dataset)

    if dev_examples is not None:

        dev_features, dev_callback_info = convert_examples_to_features(opt.task_type, dev_examples,
                                                                       opt.max_seq_len, opt.bert_dir, ent2id)

        dev_dataset = NERDataset(opt.task_type, dev_features, 'dev', use_type_embed=opt.use_type_embed)

        dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size,
                                shuffle=False, num_workers=0)

        dev_info = (dev_loader, dev_callback_info)

        model_path_list = get_model_path_list(opt.output_dir)

        metric_str = ''

        max_f1 = 0.
        max_f1_step = 0

        max_f1_path = ''

        for idx, model_path in enumerate(model_path_list):

            tmp_step = model_path.split('/')[-2].split('-')[-1]


            model, device = load_model_and_parallel(model, opt.gpu_ids[0],
                                                    ckpt_path=model_path)


            if opt.task_type == 'span':
                tmp_metric_str, tmp_f1 = span_evaluation(model, dev_info, device, ent2id)


            logger.info(f'In step {tmp_step}:\n {tmp_metric_str}')

            metric_str += f'In step {tmp_step}:\n {tmp_metric_str}' + '\n\n'

            if tmp_f1 > max_f1:
                max_f1 = tmp_f1
                max_f1_step = tmp_step
                max_f1_path = model_path

        max_metric_str = f'Max f1 is: {max_f1}, in step {max_f1_step}'

        logger.info(max_metric_str)

        metric_str += max_metric_str + '\n'

        eval_save_path = os.path.join(opt.output_dir, 'eval_metric.txt')

        with open(eval_save_path, 'a', encoding='utf-8') as f1:
            f1.write(metric_str)

        with open('./best_ckpt_path.txt', 'a', encoding='utf-8') as f2:
            f2.write(max_f1_path + '\n')

        del_dir_list = [os.path.join(opt.output_dir, path.split('/')[-2])
                        for path in model_path_list if path != max_f1_path]

        import shutil
        for x in del_dir_list:
            shutil.rmtree(x)
            logger.info('{}已删除'.format(x))

    if test_examples is not None:

        test_features, test_callback_info = convert_examples_to_features(opt.task_type, test_examples,
                                                                       opt.max_seq_len, opt.bert_dir, ent2id)

        test_dataset = NERDataset(opt.task_type, test_features, 'test', use_type_embed=opt.use_type_embed)

        test_loader = DataLoader(test_dataset, batch_size=opt.eval_batch_size,
                                shuffle=False, num_workers=0)

        test_info = (test_loader, test_callback_info)
        with open(eval_save_path, 'r', encoding='utf-8') as f1:
            model_path = f1.readline()
        metric_str = ''

        max_f1 = 0.

        tmp_step = model_path.split('/')[-2].split('-')[-1]

        model, device = load_model_and_parallel(model, opt.gpu_ids[0],
                                                ckpt_path=model_path)

        if opt.task_type == 'span':
            tmp_metric_str, tmp_f1 = span_evaluation(model, test_info, device, ent2id)


        logger.info(f'In step {tmp_step}:\n {tmp_metric_str}')

        metric_str += f'In step {tmp_step}:\n {tmp_metric_str}' + '\n\n'

        max_metric_str = f'Test Max f1 is: {max_f1}, in step {tmp_step}'

        logger.info(max_metric_str)

        metric_str += max_metric_str + '\n'

        eval_save_path = os.path.join(opt.output_dir, 'eval_metric.txt')

        with open(eval_save_path, 'a', encoding='utf-8') as f1:
            f1.write(metric_str)


def training(opt):

    processor = NERProcessor(opt.max_seq_len)

    train_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'train.json'))

    # add pseudo data to train data
    # pseudo_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'pseudo.json'))# 将预测好的测试集加入训练集
    # train_raw_examples = train_raw_examples + pseudo_raw_examples

    train_examples = processor.get_examples(train_raw_examples, 'train')

    dev_examples = None
    if opt.eval_model:
        dev_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'dev.json'))
        dev_examples = processor.get_examples(dev_raw_examples, 'dev')
        test_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'test.json'))
        test_examples = processor.get_examples(test_raw_examples, 'test')
    train_base(opt, train_examples, dev_examples, test_examples)


if __name__ == '__main__':
    start_time = time.time()
    logging.info('----------------开始计时----------------')
    logging.info('----------------------------------------')

    args = Args().get_parser()

    assert args.mode in ['train', 'stack'], 'mode mismatch'
    assert args.task_type in ['crf', 'span', 'mrc', 'lstm_crf']

    args.output_dir = os.path.join(args.output_dir, args.bert_type)

    set_seed(args.seed)

    if args.attack_train != '':
        args.output_dir += f'_{args.attack_train}'

    if args.weight_decay:
        args.output_dir += '_wd'

    if args.use_fp16:
        args.output_dir += '_fp16'

    if args.task_type == 'span':
        args.output_dir += f'_{args.loss_type}'

    if args.task_type == 'mrc':
        if args.use_type_embed:
            args.output_dir += f'_embed'
        args.output_dir += f'_{args.loss_type}'

    args.output_dir += f'_{args.task_type}'

    if args.mode == 'stack':
        args.output_dir += '_stack'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f'{args.mode} {args.task_type} in max_seq_len {args.max_seq_len}')

    if args.mode == 'train':
        training(args)


    time_dif = get_time_dif(start_time)
    logging.info("----------本次容器运行时长：{}-----------".format(time_dif))
