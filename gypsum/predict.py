import logging
import os
import pickle
import subprocess

from torch.utils.data import sampler
from tqdm import tqdm
from transformers import (RobertaConfig, RobertaModel)
from transformers import RobertaTokenizer

import c2nl.inputters.bert_dataset as data
import c2nl.inputters.bert_vector as vector
from cogypsum.bert_model import Code2NL
from cogypsum.bert_utils import *
from cogypsum.data_utils import read_examples, convert_examples_to_features
from c2nl.inputters.bert_dataloader import DataLoaderX, DataFetcher
from c2nl.inputters.timer import AverageMeter, Timer
from code2seq import c2q_utils
from code_summary.metrics import init_nmt_config
from utils.utils import try_gpu

# 设置模型参数
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
# 设置logger
logger = logging.getLogger()


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


# 初始化设置
def set_defaults(m_config):
    # 创建保存模型的目录
    subprocess.call(['mkdir', '-p', m_config.model_dir])

    # 设置模型名称
    if not m_config.model_name:
        import uuid
        import time
        m_config.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    suffix = '_test' if m_config.only_test else ''
    m_config.model_file = os.path.join(m_config.model_dir, m_config.model_name + '.mdl')
    m_config.log_file = os.path.join(m_config.model_dir, m_config.model_name + suffix + '.txt')
    m_config.pred_file = os.path.join(m_config.model_dir, m_config.model_name + suffix + '.json')

    # 加载pretrain模型
    if m_config.pretrained:
        m_config.pretrained = os.path.join(m_config.model_dir, m_config.pretrained + '.mdl')

    return m_config


# ------------------------------------------------------------------------------
# Train loop
# ------------------------------------------------------------------------------
def train(m_config, tokenizer, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss = AverageMeter()
    perplexity = AverageMeter()
    epoch_time = Timer()
    c_model = m_config.model
    current_epoch = global_stats['epoch']
    progress_bar = tqdm(range(m_config.train_len // m_config.batch_size))

    progress_bar.set_description("%s" % 'Epoch = %d [perplexity = 0.00, ml_loss = 0.00]' %
                                 current_epoch)
    pre_fetcher = DataFetcher(data_loader)
    # 开始训练
    for _ in progress_bar:
        ex = pre_fetcher.next()
        bsz = ex['batch_size']
        # 设置learning rate
        if c_model.optimizer in ['sgd', 'adam'] and current_epoch <= c_model.warmup_epochs:
            cur_lr = global_stats['warmup_factor'] * (model.updates + 1)
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = cur_lr

        net_loss = model.update(ex, tokenizer)
        ml_loss.update(net_loss['ml_loss'], bsz)
        perplexity.update(net_loss['perplexity'], bsz)
        log_info = 'Epoch = %d [perplexity = %.2f, ml_loss = %.2f]' % \
                   (current_epoch, perplexity.avg, ml_loss.avg)

        progress_bar.set_description("%s" % log_info)

    logger.info('train: Epoch %d | perplexity = %.2f | ml_loss = %.2f | '
                'Time for epoch = %.2f (s)' %
                (current_epoch, perplexity.avg, ml_loss.avg, epoch_time.time()))

    # Checkpoint
    if m_config.checkpoint:
        model.checkpoint(m_config.model_file + '.checkpoint', current_epoch + 1)
    return ml_loss.avg

# ------------------------------------------------------------------------------
# Validation loops
# ------------------------------------------------------------------------------
def validate_official(m_config, data_loader, model, tokenizer, global_stats, mode='dev'):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    # Run through examples
    examples = 0
    sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()
    with torch.no_grad():
        progress_bar = tqdm(range(m_config.dev_len // m_config.batch_size))
        pre_fetcher = DataFetcher(data_loader)
        for idx in progress_bar:
            ex = pre_fetcher.next()
            batch_size = ex['batch_size']
            ex_ids = list(range(idx * batch_size, (idx * batch_size) + batch_size))
            predictions, targets, copy_info = model.predict(ex, tokenizer, replace_unk=True)
            # print("=" * 49)
            # print(predictions)
            # print(targets)
            src_sequences = [code for code in ex['code_texts']]
            examples += batch_size
            for key, src, pred, tgt in zip(ex_ids, src_sequences, predictions, targets):
                # if 'ip address' in pred:
                #     print(pred)
                #     print(tgt)
                #     print(src)
                #     print(key)
                #     exit(0)
                hypotheses[key] = [pred]
                references[key] = tgt if isinstance(tgt, list) else [tgt]
                sources[key] = src

            if copy_info is not None:
                flag = isinstance(copy_info, list)
                if not flag:
                    copy_info = copy_info.cpu().numpy().astype(int).tolist()
                for key, cp in zip(ex_ids, copy_info):
                    if not flag:
                        copy_dict[key] = cp
                    else:
                        copy_dict[key] = cp.cpu().numpy().astype(int)

            progress_bar.set_description("%s" % 'Epoch = %d [validating ... ]' % global_stats['epoch'])

    copy_dict = None if len(copy_dict) == 0 else copy_dict
    bleu, rouge_l, meteor, precision, recall, f1 = eval_accuracies(m_config,
                                                                   hypotheses,
                                                                   references,
                                                                   copy_dict,
                                                                   sources=sources,
                                                                   filename=m_config.pred_file,
                                                                   print_copy_info=m_config.print_copy_info,
                                                                   mode=mode)
    result = dict()
    result['bleu'] = bleu
    result['rouge_l'] = rouge_l
    result['meteor'] = meteor
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1

    if mode == 'test':
        logger.info('test valid official: '
                    'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                    (bleu, rouge_l, meteor) +
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | '
                    'examples = %d | ' %
                    (precision, recall, f1, examples) +
                    'test time = %.2f (s)' % eval_time.time())

    else:
        logger.info('dev valid official: Epoch = %d | ' %
                    (global_stats['epoch']) +
                    'bleu = %.2f | rouge_l = %.2f | '
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | examples = %d | ' %
                    (bleu, rouge_l, precision, recall, f1, examples) +
                    'valid time = %.2f (s)' % eval_time.time())

    return result


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main(m_config):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load pretrained model')
    # 加载模型
    config_class, model_class, tokenizer_class = MODEL_CLASSES[nmt_config.model_type]
    config = config_class.from_pretrained(
        nmt_config.config_name if nmt_config.config_name else nmt_config.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        nmt_config.tokenizer_name if nmt_config.tokenizer_name else nmt_config.model_name_or_path,
        do_lower_case=nmt_config.do_lower_case)
    code_bert = model_class.from_pretrained(nmt_config.model_name_or_path, config=config)

    c_model = m_config.model
    train_exs = []
    include_ast = m_config.include_ast
    vocab_sub_token = None
    vocab_nodes = None
    vocab_target = None
    if include_ast:
        code_seq_config = m_config.model.code2seq
        data_dir = os.path.join(code_seq_config.data.home)
        dict_file = os.path.join(data_dir, code_seq_config.data.dict)
        # load vocab dict
        with open(dict_file, 'rb') as file:
            sub_token_to_count = pickle.load(file)
            node_to_count = pickle.load(file)
            target_to_count = pickle.load(file)

        # making vocab dicts for terminal sub_token, non_terminal node and target.
        vocab_sub_token = c2q_utils.Vocab(word2id=c2q_utils.word2id)
        vocab_nodes = c2q_utils.Vocab(word2id=c2q_utils.word2id)
        vocab_target = c2q_utils.Vocab(word2id=c2q_utils.word2id)

        vocab_sub_token.build_vocab(list(sub_token_to_count.keys()), min_count=0)
        vocab_nodes.build_vocab(list(node_to_count.keys()), min_count=0)
        vocab_target.build_vocab(list(target_to_count.keys()), min_count=0)

        vocab_size_sub_token = len(vocab_sub_token.id2word)
        vocab_size_nodes = len(vocab_nodes.id2word)
        vocab_size_target = len(vocab_target.id2word)

        m_config.model.code2seq.hyper.set('vocab_size_sub_token', vocab_size_sub_token)
        m_config.model.code2seq.hyper.set('vocab_size_nodes', vocab_size_nodes)
        m_config.model.code2seq.hyper.set('vocab_size_target', vocab_size_target)

        print("vocab_size_sub_token: " + str(vocab_size_sub_token))
        print("vocab_size_nodes: " + str(vocab_size_nodes))
        print("vocab_size_target: " + str(vocab_size_target))
        print(m_config.model.code2seq.hyper.to_vanilla_())

    m_config.dataset_weights = dict()
    src_examples = read_examples(m_config, 'train')
    m_config.dataset_weights[m_config.lang] = len(src_examples)
    train_exs = convert_examples_to_features(src_examples, tokenizer, m_config, stage='train', logger=logger)
    logger.info('Num train examples = %d' % len(train_exs))
    m_config.num_train_examples = len(train_exs)
    for lang in m_config.dataset_weights.keys():
        weight = (1.0 * m_config.dataset_weights[lang]) / len(train_exs)
        m_config.dataset_weights[lang] = round(weight, 2)
    logger.info('Dataset weights = %s' % str(m_config.dataset_weights))

    dev_examples = read_examples(m_config, 'test')
    dev_exs = convert_examples_to_features(dev_examples, tokenizer, m_config, stage='dev', logger=logger)
    logger.info('Num dev examples = %d' % len(dev_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 1
    if m_config.only_test:
        if m_config.pretrained:
            model = Code2NL.load(m_config.pretrained, code_bert, tokenizer, m_config)
        else:
            if not os.path.isfile(m_config.model_file):
                raise IOError('No such file: %s' % m_config.model_file)
            model = Code2NL.load(m_config.model_file, code_bert, tokenizer, m_config)

    # Use the GPU?
    if m_config.cuda:
        try_gpu(model)

    if m_config.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')

    train_loader = None
    dev_dataset = data.CommentDataset(dev_exs, tokenizer, m_config)
    if include_ast:
        dev_dataset.add_vocab(vocab_subtoken=vocab_sub_token, vocab_nodes=vocab_nodes,
                              vocab_target=vocab_target)
    dev_sampler = sampler.SequentialSampler(dev_dataset)

    dev_loader = DataLoaderX(
        dev_dataset,
        batch_size=m_config.test_batch_size,
        sampler=dev_sampler,
        num_workers=m_config.data_workers,
        collate_fn=vector.batch_handle,
        pin_memory=m_config.cuda,
        drop_last=m_config.parallel
    )
    m_config.dev_len = len(dev_dataset)

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(m_config.to_vanilla_(), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST
    stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
    validate_official(m_config, dev_loader, model, tokenizer, stats, mode='test')


if __name__ == '__main__':

    # Parse cmdline args and setup environment
    nmt_config = init_nmt_config()
    nmt_config = set_defaults(nmt_config)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if nmt_config.log_file:
        if nmt_config.checkpoint:
            logfile = logging.FileHandler(nmt_config.log_file, 'a')
        else:
            logfile = logging.FileHandler(nmt_config.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)

    # Run!
    main(nmt_config)
