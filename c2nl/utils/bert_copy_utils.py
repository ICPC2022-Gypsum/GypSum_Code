import torch
from c2nl.inputters import constants


def collapse_copy_scores(tokenizer, src_vocabs):
    """
    Given scores from an expanded dictionary
    corresponding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = tokenizer.vocab_size
    blank_arr, fill_arr = [], []
    for b in range(len(src_vocabs)):
        blank = []
        fill = []
        src_vocab = src_vocabs[b]
        # Starting from 2 to ignore PAD and UNK token TODO
        for i in range(2, len(src_vocab)):
            sw = src_vocab[i]
            ti = tokenizer.convert_tokens_to_ids(sw)
            if ti != tokenizer.unk_token_id:
                blank.append(offset + i)
                fill.append(ti)

        blank_arr.append(blank)
        fill_arr.append(fill)

    return blank_arr, fill_arr


def make_src_map(data):
    """ make src map """
    src_size = max([t.size(0) for t in data])
    src_vocab_size = max([t.max() for t in data]) + 1
    alignment = torch.zeros(len(data), src_size, src_vocab_size)
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[i, j, t] = 1
    return alignment


def align(data):
    """ align """
    tgt_size = max([t.size(0) for t in data])
    alignment = torch.zeros(len(data), tgt_size).long()
    for i, sent in enumerate(data):
        alignment[i, :sent.size(0)] = sent
    return alignment


def replace_unknown(prediction, attn, src_raw, unk_word=None):
    """ ?
        attn: tgt_len x src_len
    """
    tokens = prediction.split()
    # print(attn.size())
    # print(prediction)
    # print(tokens)
    for i in range(len(tokens)):
        if tokens[i] == unk_word:
            _, max_index = attn[i].max(0)
            tokens[i] = src_raw[max_index.item()]
    return ' '.join(tokens)
