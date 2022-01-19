import torch
import numpy as np
from .c2s_dict import sentence_to_ids, pad_seq
import random
from gypsum.utils.utility import long_t, float_t
from torch_geometric.data import Data


def vectorize(ex, config, args):
    """Vectorize a single example."""
    vectorized_ex = dict()

    # code 信息
    vectorized_ex['code_ids'] = ex.source_ids
    vectorized_ex['code_mask'] = ex.source_mask
    vectorized_ex['code_tokens'] = ex.source_tokens
    vectorized_ex['code_lens'] = ex.source_len
    vectorized_ex['src_vocab'] = ex.src_vocab
    vectorized_ex['code_texts'] = ex.source_texts

    # summary信息
    vectorized_ex['summary_ids'] = None
    vectorized_ex['summary_mask'] = None
    vectorized_ex['summary_lens'] = None
    vectorized_ex['summary_texts'] = None
    vectorized_ex['summary_tokens'] = None
    vectorized_ex['target_seq'] = None

    # if not config.only_test:
    vectorized_ex['summary_ids'] = ex.target_ids
    vectorized_ex['summary_mask'] = ex.target_mask

    vectorized_ex['summary_tokens'] = ex.target_tokens
    vectorized_ex['summary_lens'] = ex.target_len
    vectorized_ex['summary_texts'] = ex.target_texts

    # target is only used to compute loss during training
    vectorized_ex['target_seq'] = ex.target_ids

    '''
    Handle Graph
    '''
    if ex.nodes is not None and config.include_graph:
        graph_config = config.model.graph
        # TODO Do we need to minus one for cls token ??

        max_node_len = graph_config.max_node_len
        graph_tokens = ex.graph_tokens

        node_len = min(len(graph_tokens), max_node_len)
        graph_tokens = graph_tokens[:node_len]
        graph_idx = [long_t(ex.graph_ids[:node_len])]
        # print(ex.graph_ids[:node_len])
        # print(graph_tokens)
        # print(ex.nodes)
        # print([ex.nodes[i] for i in ex.graph_ids[:node_len]])
        # print(ex.graph_ids)
        # print('='*50)

        x = long_t(ex.nodes)
        edge = long_t(ex.edges)
        edge_attr = long_t(ex.edge_attrs)
        node_len = node_len
        x_len = len(x)

        graph_data = Data(x=x, edge_index=edge, edge_attr=edge_attr,
                          indice=graph_idx, node_len=node_len, x_len=x_len)

        vectorized_ex['graph_data'] = graph_data
        vectorized_ex['graph_tokens'] = graph_tokens
        vectorized_ex['graph_vocab'] = ex.graph_vocab
        vectorized_ex['graph_idx'] = x[ex.graph_ids[:node_len]]

    return vectorized_ex


def batch_handle(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)

    has_token = 'code_ids' in batch[0].keys() and batch[0]['code_ids']

    if has_token:
        # Batch Code Representations
        max_code_len = np.max([ex['code_lens'] for ex in batch])
        code_ids = torch.cat(
            [long_t(ex['code_ids'][:max_code_len]).unsqueeze(0) for ex in batch], dim=0)
        code_mask = torch.cat(
            [long_t(ex['code_mask'][:max_code_len]).unsqueeze(0) for ex in batch], dim=0)
        code_lens = long_t([ex['code_lens'] for ex in batch])

        source_maps = []
        src_vocabs = []

        for i in range(batch_size):
            context = batch[i]['code_tokens']
            vocab = batch[i]['src_vocab']
            src_vocabs.append(vocab)

            src_map = long_t([vocab[w] for w in context])
            source_maps.append(src_map)

        code_tokens = [ex['code_tokens'] for ex in batch]

    else:
        code_ids = None
        code_mask = None
        code_lens = None
        code_tokens = None
        src_vocabs = None
        source_maps = None

    code_texts = [ex['code_texts'] for ex in batch]

    has_graph = 'graph_data' in batch[0].keys() and batch[0]['graph_data']

    if not has_graph:
        graph_datas = []
        graph_tokens = None
        graph_maps = None
        graph_vocabs = None
        graph_idx = None
    else:
        graph_datas = [ex['graph_data'] for ex in batch]
        graph_tokens = [ex['graph_tokens'] for ex in batch]

        node_lens = [ex['graph_data'].node_len for ex in batch]
        max_node_len = max(node_lens)
        graph_idx = torch.cat([torch.cat([ex['graph_idx'], torch.zeros(max_node_len-node_lens[idx])]).unsqueeze(0)
                               for idx, ex in enumerate(batch)], 0)
        graph_maps = []
        graph_vocabs = []

        for i in range(batch_size):
            context = batch[i]['graph_tokens']
            vocab = batch[i]['graph_vocab']
            graph_vocabs.append(vocab)

            graph_map = long_t([vocab[w] for w in context])
            graph_maps.append(graph_map)

    # --------- Prepare Summary tensors ---------
    no_summary = batch[0]['summary_ids'] is None

    if no_summary:
        summary_lens = None
        summary_ids = None
        summary_mask = None
        target_seq = None
        alignments = None
    else:
        max_summary_len = np.max([ex['summary_lens'] for ex in batch])
        summary_ids = torch.cat(
            [long_t(ex['summary_ids'][: max_summary_len]).unsqueeze(0) for ex in batch], dim=0)
        summary_mask = torch.cat(
            [long_t(ex['summary_mask'][: max_summary_len]).unsqueeze(0) for ex in batch], dim=0)
        target_seq = torch.cat(
            [long_t(ex['target_seq'][: max_summary_len]).unsqueeze(0) for ex in batch], dim=0)
        summary_lens = long_t([ex['summary_lens'] for ex in batch])

        alignments = []
        if has_token:
            for i in range(batch_size):
                target = batch[i]['summary_tokens']
                align_mask = long_t([src_vocabs[i][w] for w in target])
                alignments.append(align_mask)

        graph_alignments = []
        if has_graph:
            for i in range(batch_size):
                target = batch[i]['summary_tokens']
                graph_align_mask = long_t([graph_vocabs[i][w] for w in target])
                graph_alignments.append(graph_align_mask)

    return {
        'batch_size': batch_size,
        'code_ids': code_ids,
        'code_mask': code_mask,
        'code_lens': code_lens,
        'summary_ids': summary_ids,
        'summary_mask': summary_mask,
        'summary_lens': summary_lens,
        'target_seq': target_seq,
        'code_texts': code_texts,
        'summary_texts': [ex['summary_texts'] for ex in batch],
        'code_tokens': code_tokens,
        'summary_tokens': [ex['summary_tokens'] for ex in batch],
        'src_vocab': src_vocabs,
        'src_map': source_maps,
        'alignment': alignments,
        'graph_data': graph_datas,
        'graph_tokens': graph_tokens,
        'graph_vocab': graph_vocabs,
        'graph_map': graph_maps,
        'graph_alignment': graph_alignments,
        'graph_idx': graph_idx
    }
