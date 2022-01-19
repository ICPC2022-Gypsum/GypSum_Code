import torch
from nltk import bleu_score
import os
import pickle

PAD = 1
BOS = 0
EOS = 2
UNK = 3

PAD_TOKEN = '<PAD>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'
UNK_TOKEN = '<UNK>'

word2id = {
    PAD_TOKEN: PAD,
    BOS_TOKEN: BOS,
    EOS_TOKEN: EOS,
    UNK_TOKEN: UNK,
}


class Vocab(object):
    def __init__(self, word2id={}):

        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def build_vocab(self, sentences, min_count=1):
        word_counter = {}
        for word in sentences:
            word_counter[word] = word_counter.get(word, 0) + 1

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word


def sentence_to_ids(vocab, sentence):
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    ids += [EOS]
    return ids


def ids_to_sentence(vocab, ids):
    return [vocab.id2word[_id] for _id in ids]


def trim_eos(ids):
    if EOS in ids:
        return ids[:ids.index(EOS)]
    else:
        return ids


def pad_seq(seq, max_length):
    # pad tail of sequence to extend sequence length up to max_length
    res = seq + [PAD for i in range(max_length - len(seq))]
    return res


def load_code2seq_dict(c2s_config):
    data_dir = os.path.join(c2s_config.data.home)
    dict_file = os.path.join(data_dir, c2s_config.data.dict)

    # load vocab dict
    with open(dict_file, 'rb') as file:
        sub_token_to_count = pickle.load(file)
        node_to_count = pickle.load(file)
        target_to_count = pickle.load(file)

    # making vocab dicts for terminal sub_token, non_terminal node and target.
    vocab_sub_token = Vocab(word2id=word2id)
    vocab_nodes = Vocab(word2id=word2id)
    vocab_target = Vocab(word2id=word2id)

    vocab_sub_token.build_vocab(list(sub_token_to_count.keys()), min_count=0)
    vocab_nodes.build_vocab(list(node_to_count.keys()), min_count=0)
    vocab_target.build_vocab(list(target_to_count.keys()), min_count=0)

    vocab_size_sub_token = len(vocab_sub_token.id2word)
    vocab_size_nodes = len(vocab_nodes.id2word)
    vocab_size_target = len(vocab_target.id2word)

    c2s_config.hyper.set('vocab_size_sub_token', vocab_size_sub_token)
    c2s_config.hyper.set('vocab_size_nodes', vocab_size_nodes)
    c2s_config.hyper.set('vocab_size_target', vocab_size_target)

    print("vocab_size_sub_token: " + str(vocab_size_sub_token))
    print("vocab_size_nodes: " + str(vocab_size_nodes))
    print("vocab_size_target: " + str(vocab_size_target))

    return vocab_sub_token, vocab_nodes, vocab_target
