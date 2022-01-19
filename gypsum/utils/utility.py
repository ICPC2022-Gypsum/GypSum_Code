import argparse
import json
import os
import logging
import copy

import collections
import math
import subprocess
from transformers import (RobertaConfig, RobertaModel)
from transformers import RobertaTokenizer

from collections import defaultdict
import numpy as np
import torch

_GPUS_EXIST = True
device = None

PAD_INDEX = 0
BOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3

PAD = '<pad>'
BOS = '<s>'
EOS = '</s>'
UNK = '<unk>'


def long_t(x):
    return torch.tensor(x, dtype=torch.long)


def float_t(x):
    return torch.tensor(x, dtype=torch.float)


# 初始化设置
def get_default_config():

    config = init_nmt_config()

    # 创建保存模型的目录
    subprocess.call(['mkdir', '-p', config.model_dir])

    # 设置模型名称
    if not config.model_name:
        import uuid
        import time
        config.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    suffix = '_test' if config.only_test else ''
    config.model_file = os.path.join(
        config.model_dir, config.model_name + '.mdl')
    config.log_file = os.path.join(
        config.model_dir, config.model_name + suffix + '.txt')
    config.pred_file = os.path.join(
        config.model_dir, config.model_name + suffix + '.json')

    # 加载pretrain模型
    if config.pretrained:
        config.pretrained = os.path.join(
            config.model_dir, config.pretrained + '.mdl')

    return config


def init_nmt_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--config-gpu',
                        help='gpu')
    parser.add_argument('-c', '--config-string',
                        help='Additional config (as JSON)')
    parser.add_argument('configs', nargs='+',
                        help='Config JSON or YAML files')
    # TODO: greedy sampling is still buggy!
    # parser.add_argument('--sample_method', default='random', choices=['random', 'greedy'])
    # configs = ['config/java_graph_test.yml', 'config/general_config.yml']

    configs = parser.parse_args().configs
    gpu = [int(configs[0])]
    configs = configs[1:]
    config = {}
    for path in configs:
        new_config = load_config(path)
        merge_configs(config, new_config)

    # if args.config_string:
    #     new_config = json.loads(args.config_string)
    #     merge_configs(config, new_config)
    config = ConfigDict(config)
    nmt_config = config.nmt
    general_config = config.general
    gpu_list = general_config.gpu.to_vanilla_()
    gpu_list = gpu

    ngpus = len(gpu_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
    global device
    device = torch.device(
        f'cuda:{gpu_list[0]}' if torch.cuda.is_available() else 'cpu')

    nmt_config.cuda = True if ngpus > 0 else False
    nmt_config.parallel = general_config.parallel
    if ngpus <= 1:
        torch.cuda.set_device('cuda:' + str(gpu_list[0]))
    # seed the RNG
    seed = config.general.seed
    torch.manual_seed(seed)
    if general_config.use_cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    return nmt_config


def set_logging(logger, config):
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if config.log_file:
        if config.checkpoint:
            logfile = logging.FileHandler(config.log_file, 'a')
        else:
            logfile = logging.FileHandler(config.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)


def load_bert(config):
    # 设置模型参数
    MODEL_CLASSES = {'roberta': (
        RobertaConfig, RobertaModel, RobertaTokenizer)}

    config_class, model_class, tokenizer_class = MODEL_CLASSES[config.model_type]
    output_hidden_states = True
    bert_config = config_class.from_pretrained(
        config.config_name if config.config_name else config.model_name_or_path, output_attentions=True)

    tokenizer = tokenizer_class.from_pretrained(
        config.tokenizer_name if config.tokenizer_name else config.model_name_or_path,
        do_lower_case=config.do_lower_case)

    # tokenize_token = '_<SplitNode>_'
    # javalang_special_tokens = ['CompilationUnit', 'Import', 'Documented', 'Declaration', 'TypeDeclaration', 'PackageDeclaration',
    #                            'ClassDeclaration', 'EnumDeclaration', 'InterfaceDeclaration', 'AnnotationDeclaration', 'Type',
    #                            'BasicType', 'ReferenceType', 'TypeArgument', 'TypeParameter', 'Annotation', 'ElementValuePair',
    #                            'ElementArrayValue', 'Member', 'MethodDeclaration', 'FieldDeclaration', 'ConstructorDeclaration',
    #                            'ConstantDeclaration', 'ArrayInitializer', 'VariableDeclaration', 'LocalVariableDeclaration',
    #                            'VariableDeclarator', 'FormalParameter', 'InferredFormalParameter', 'Statement', 'IfStatement',
    #                            'WhileStatement', 'DoStatement', 'ForStatement', 'AssertStatement', 'BreakStatement', 'ContinueStatement',
    #                            'ReturnStatement', 'ThrowStatement', 'SynchronizedStatement', 'TryStatement', 'SwitchStatement',
    #                            'BlockStatement', 'StatementExpression', 'TryResource', 'CatchClause', 'CatchClauseParameter',
    #                            'SwitchStatementCase', 'ForControl', 'EnhancedForControl', 'Expression', 'Assignment', 'TernaryExpression',
    #                            'BinaryOperation', 'Cast', 'MethodReference', 'LambdaExpression', 'Primary', 'Literal', 'This',
    #                            'MemberReference', 'Invocation', 'ExplicitConstructorInvocation', 'SuperConstructorInvocation',
    #                            'MethodInvocation', 'SuperMethodInvocation', 'SuperMemberReference', 'ArraySelector', 'ClassReference',
    #                            'VoidClassReference', 'Creator', 'ArrayCreator', 'ClassCreator', 'InnerClassCreator', 'EnumBody',
    #                            'EnumConstantDeclaration', 'AnnotationMethod', 'Modifier', tokenize_token]

    # special_tokens_dict = {
    #     'additional_special_tokens': javalang_special_tokens}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    code_bert = model_class.from_pretrained(
        config.model_name_or_path, config=bert_config)

    return code_bert, tokenizer


'''
CONFIG CODE
'''

try:
    import yaml
except ImportError:
    logging.warning('yaml is not supported')


def load_config(filename):
    if filename.endswith('.json'):
        with open(filename) as fin:
            return json.load(fin)
    elif filename.endswith('.yaml') or filename.endswith('.yml'):
        with open(filename) as fin:
            return yaml.load(fin, Loader=yaml.FullLoader)
    else:
        raise ValueError('Unknown file type: {}'.format(filename))


def dump_config(config, filename):
    if isinstance(config, ConfigDict):
        config = config.to_vanilla_()
    print('Writing config to {}'.format(filename))
    with open(filename, 'w') as fout:
        json.dump(config, fout, indent=2)
        fout.write('\n')


def merge_configs(base, new, wildcard_key='XXX'):
    """
    Merge the new config (dict) into the base config (dict).
    This modifies base but not new.

    Rules:
    - Look at each key k in the new config.
    - If base[k] does not exist, set base[k] = new[k]
    - If base[k] exists:
        - If base[k] and new[k] are both dicts, do recursive merge.
        - If new[k] is null, remove key k from base.
        - Otherwise, set base[k] = new[k].

    Special Rule:
    - If k is wildcard_key, merge new[k] with base[k'] for all k'
    """
    for key in new:
        base_keys = list(base) if key == wildcard_key else [key]
        for base_key in base_keys:
            if base_key not in base:
                base[base_key] = copy.deepcopy(new[key])
            elif isinstance(base[base_key], dict) and isinstance(new[key], dict):
                merge_configs(base[base_key], new[key], wildcard_key)
            elif new[key] is None:
                del base[base_key]
            else:
                base[base_key] = copy.deepcopy(new[key])


class ConfigDict(object):
    """
    Allow the config to be accessed with dot notation:
    config['epochs'] --> config.epochs
    """

    def __init__(self, data):
        assert isinstance(data, dict)
        self._data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                value = ConfigDict(value)
            elif isinstance(value, list):
                value = ConfigList(value)
            self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, key):
        return self._data[key]

    def set(self, key, value):
        self._data[key] = value

    def __iter__(self):
        for key in self._data:
            yield key

    def get_(self, key, value=None):
        return self._data.get(key, value)

    def to_vanilla_(self):
        data = {}
        for key, value in self._data.items():
            if isinstance(value, (ConfigDict, ConfigList)):
                value = value.to_vanilla_()
            data[key] = value
        return data


class ConfigList(object):

    def __init__(self, data):
        assert isinstance(data, list)
        self._data = []
        for value in data:
            if isinstance(value, dict):
                value = ConfigDict(value)
            elif isinstance(value, list):
                value = ConfigList(value)
            self._data.append(value)

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for value in self._data:
            yield value

    def expand(self, num):
        self._data = self._data * num

    def to_vanilla_(self):
        data = []
        for value in self._data:
            if isinstance(value, (ConfigDict, ConfigList)):
                value = value.to_vanilla_()
            data.append(value)
        return data


def try_gpu(x, cuda=True):
    global device

    if x is None:
        return x
    """Try to put x on a GPU."""
    global _GPUS_EXIST
    if _GPUS_EXIST and cuda and device:
        try:
            return x.to(device)
        except (AssertionError, RuntimeError):
            print('No GPUs detected. Sticking with CPUs.')
            _GPUS_EXIST = False
    return x


def get_inputs(src, vocab):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """
    word_ids = word2id(src, vocab)
    padded_ids, masks = pad_sequence(word_ids)
    return try_gpu(torch.LongTensor(padded_ids))


def pad_sequence(src):
    """
    padding input函数，返回mask和padding_input

    Args:
        src: 输入数据
    Returns:
        padded_word
        masks
    """
    max_len = max(len(s) for s in src)
    batch_size = len(src)

    padded_word = []
    masks = []
    for i in range(max_len):
        padded_word.append(
            [src[k][i] if len(src[k]) > i else PAD_INDEX for k in range(batch_size)])
        masks.append([1 if len(src[k]) > i else 0 for k in range(batch_size)])

    return padded_word, masks


def word2id(inputs, vocab):
    """

    Args:
        inputs:  输入数据
        vocab:  词典
    Returns:
        input_indices
    """
    if type(inputs[0]) == list:
        return [[vocab[w] for w in s] for s in inputs]
    else:
        return [vocab[w] for w in inputs]


def tensor_transform(linear, x):
    # X is a 3D tensor
    return linear(x.contiguous().view(-1, x.size(2))).view(x.size(0), x.size(1), -1)


def read_corpus(file_path, source):
    data = []
    for line in open(file_path, encoding='utf-8'):
        sent = line.strip().split(' ')
        sent = [w for w in sent]
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = [BOS] + sent + [EOS]
        data.append(sent)

    return data


def read_corpus_for_dsl(file_path, source):
    data = []
    lm_scores = []
    scores_path = file_path + '.score'
    for line, score in zip(open(file_path), open(scores_path)):
        sent = line.strip().split(' ')
        if source != 'tgt':
            lm_scores.append(float(score))
        data.append(sent)

    return data, lm_scores


def batch_slice(data, batch_size, sort=True):
    batched_data = []
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - \
            1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0]
                     for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1]
                     for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(
                src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        batched_data.append((src_sents, tgt_sents))

    return batched_data


def batch_slice_for_dsl(data, batch_size, sort=True):
    batched_data = []
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - \
            1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0]
                     for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1]
                     for b in range(cur_batch_size)]
        src_scores = [data[i * batch_size + b][2]
                      for b in range(cur_batch_size)]
        tgt_scores = [data[i * batch_size + b][3]
                      for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(
                src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]
            src_scores = [src_scores[src_id] for src_id in src_ids]
            tgt_scores = [tgt_scores[src_id] for src_id in src_ids]

        batched_data.append((src_sents, tgt_sents, src_scores, tgt_scores))

    return batched_data


def get_new_batch(batch_data):
    cur_batch_size = len(batch_data[0])
    src_sents, tgt_sents, src_scores, tgt_scores = batch_data[
        0], batch_data[1], batch_data[2], batch_data[3]
    src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(
        tgt_sents[src_id]), reverse=True)
    src_sents = [src_sents[src_id] for src_id in src_ids]
    tgt_sents = [tgt_sents[src_id] for src_id in src_ids]
    src_scores = [src_scores[src_id] for src_id in src_ids]
    tgt_scores = [tgt_scores[src_id] for src_id in src_ids]

    batch_data = (src_sents, tgt_sents, src_scores, tgt_scores)
    return batch_data


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """
    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle:
            np.random.shuffle(tuples)
        batched_data.extend(batch_slice(tuples, batch_size))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch


def data_iter_for_dual(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """

    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle:
            np.random.shuffle(tuples)
        batched_data.extend(batch_slice_for_dsl(tuples, batch_size))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch
