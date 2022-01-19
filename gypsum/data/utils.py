import copy
import json
import pickle as pkl
import torch

from .vocabulary import Vocabulary
import operator
from functools import reduce


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 nodes=None,
                 edges=None,
                 edge_attrs=None,
                 graph_tokens=None,
                 graph_ids=None,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.nodes = nodes
        self.edges = edges
        self.edge_attrs = edge_attrs
        self.graph_tokens = graph_tokens
        self.graph_ids = graph_ids


def read_examples(config, task):
    """Read examples from filename."""
    import os
    assert task in ['train', 'dev', 'test', 'test_remove']

    base_dir = os.path.join(config.data_dir, config.lang, task)
    train_src = os.path.join(base_dir, config.src)
    train_tgt = os.path.join(base_dir, config.tgt)
    examples = []
    src_lines = open(train_src, encoding='utf-8').readlines()
    tgt_lines = open(train_tgt, encoding='utf-8').readlines()

    graph_config = config.model.graph
    include_graph = config.include_graph
    include_token = config.include_token

    graph_data = None
    nodes = None
    edge_index = None
    edge_attrs = None
    token_list = None
    token_ids = None

    '''
    Load Graph Data
    '''
    if include_graph:
        graph_data = pkl.load(
            open(os.path.join(base_dir, graph_config.data), 'rb'))

    for idx, (code, nl) in enumerate(zip(src_lines, tgt_lines)):
        code = code.replace('\n', ' ')
        code = ' '.join([i for i in code.split(' ') if i.strip()]).lower()
        nl = nl.replace('\n', ' ')
        nl = ' '.join([i for i in nl.split(' ') if i.strip()]).lower()
        if include_graph:
            graph = graph_data[idx]
            nodes, edge_index, edge_attrs, _, token_list, token_ids = graph

        # if not include_token:
        #     code = None

        # TODO Filter nodes num > 2000
        if not include_graph or len(nodes) < 2000:
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                    nodes=nodes,
                    edges=edge_index,
                    edge_attrs=edge_attrs,
                    graph_tokens=token_list,
                    graph_ids=token_ids,
                )
            )

    print('task: %s, num_examples: %d' % (task, len(examples)))
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_tokens,
                 target_tokens,
                 source_texts,
                 target_texts,
                 source_mask,
                 target_mask,
                 source_len,
                 target_len,
                 # graph
                 nodes,
                 edges,
                 edge_attrs,
                 graph_tokens,
                 graph_ids):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.source_len = source_len
        self.target_len = target_len
        self.nodes = nodes
        self.edges = edges
        self.edge_attrs = edge_attrs
        self.graph_tokens = graph_tokens
        self.graph_ids = graph_ids

        self.src_vocab = None
        self.graph_vocab = None

    # TODO 检查是否影响copy
    def form_src_vocab(cls, tokenizer, source):
        src_vocab = Vocabulary(tokenizer)
        # TODO 删除eos和bos
        src_vocab.tok2ind = {tokenizer.pad_token: 0,
                             tokenizer.unk_token: 1}
        src_vocab.ind2tok = {0: tokenizer.pad_token,
                             1: tokenizer.unk_token}
        # print(self.src_vocab.tok2ind)
        # print(self.src_vocab.ind2tok)
        # assert self.src_vocab.remove(tokenizer.cls_token)
        # assert self.src_vocab.remove(tokenizer.sep_token)
        src_vocab.add_tokens(source)
        # print(self.src_vocab.tok2ind)
        # print(self.src_vocab.ind2tok)
        # quit(-1)
        return src_vocab


def convert_examples_to_features(examples, tokenizer, nmt_config, stage=None, logger=None):
    features = []
    c_model = nmt_config.model

    include_token = nmt_config.include_token
    include_graph = nmt_config.include_graph

    source_tokens = None
    source_ids = None
    source_mask = None
    source_texts = None
    source_len = None

    for example_index, example in enumerate(examples):
        # source
        # TODO source token 不使用bos和eos
        source_tokens = tokenizer.tokenize(example.source)[
            :c_model.max_source_length]

        # source_tokens = [tokenizer.cls_token] + \
        #     source_tokens + [tokenizer.sep_token]

        if include_token:
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

            # padding  1 for valid 0 for pad
            source_len = len(source_ids)
            source_mask = [1] * (source_len)

            padding_length = c_model.max_source_length - source_len

            source_ids += [tokenizer.pad_token_id] * padding_length
            source_mask += [0] * padding_length

            source_texts = example.source

        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            # TODO source token 切割方式改变
            target_tokens = tokenizer.tokenize(example.target)[
                :c_model.max_target_length - 2]
            # target_tokens =

        target_tokens = [tokenizer.cls_token] + \
            target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        target_len = len(target_ids)
        target_mask = [1] * target_len
        padding_length = c_model.max_target_length - target_len

        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        target_texts = example.target

        '''
        No Need To Process Graph Data
        '''
        if example_index < 5 and logger:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                if include_token:
                    logger.info("source_tokens: {}".format(
                        [x.replace('\u0120', '_') for x in source_tokens]))
                    logger.info("source_ids: {}".format(
                        ' '.join(map(str, source_ids))))
                    logger.info("source_mask: {}".format(
                        ' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format(
                    [x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(
                    ' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(
                    ' '.join(map(str, target_mask))))

        nodes = None
        edges = None
        edge_attrs = None
        graph_tokens = None
        graph_ids = None

        if include_graph:
            max_node_len = c_model.graph.max_node_len
            graph_tokens = example.graph_tokens[:max_node_len]
            graph_ids = example.graph_ids[:max_node_len]
            nodes = example.nodes
            edges = example.edges
            edge_attrs = example.edge_attrs

        feature = InputFeatures(
            example.idx,
            source_ids,
            target_ids,
            source_tokens,
            target_tokens,
            source_texts,
            target_texts,
            source_mask,
            target_mask,
            source_len,
            target_len,
            nodes=nodes,
            edges=edges,
            edge_attrs=edge_attrs,
            graph_tokens=graph_tokens,
            graph_ids=graph_ids)

        '''
        Construct Local dict for src sentence
        '''
        if include_token:
            feature.src_vocab = feature.form_src_vocab(
                tokenizer, source_tokens)

        if include_graph:
            feature.graph_vocab = feature.form_src_vocab(
                tokenizer, graph_tokens)

        features.append(feature)

    return features


def sort_sequence(inputs, sequence_length, batch_first):
    sorted_seq_lengths, indices = torch.sort(sequence_length, descending=True)

    _, desorted_indices = torch.sort(indices, descending=False)

    if batch_first:
        inputs = inputs[indices]
    else:
        inputs = inputs[:, indices]

    return inputs, sorted_seq_lengths, desorted_indices
