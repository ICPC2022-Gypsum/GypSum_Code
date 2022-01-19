# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/copy_generator.py
""" Generator module """
import torch.nn as nn
import torch

from c2nl.inputters import constants
from c2nl.utils.misc import aeq


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source.
    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.
    The copy generator is an extended version of the standard
    generator that computes three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary,
    computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    .. mermaid::
       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O
    Args:
       input_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary
    """

    def __init__(self, input_size, pad_index, generator, eps=1e-20, include_extra=False):
        super(CopyGenerator, self).__init__()
        self.linear = generator
        self.include_extra = include_extra

        if not include_extra:
            self.linear_copy = nn.Linear(input_size, 1)
            self.sigmoid = nn.Sigmoid()
        else:
            self.linear_copy = nn.Linear(input_size, 3)
            self.softmax_copy = nn.Softmax(dim=-1)

        self.pad_index = pad_index
        self.softmax = nn.Softmax(dim=-1)
        self.eps = eps

    def forward(self, hidden, attn, src_map, extra_attn=None, extra_map=None):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.
        Args:
           hidden (`FloatTensor`): hidden outputs `[batch, tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch, tlen, slen]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[batch, src_len, extra_words]`
        """
        # CHECKS
        include_extra = self.include_extra

        batch, tlen, _ = hidden.size()
        batch_, tlen_, slen = attn.size()
        batch, slen_, cvocab = src_map.size()

        aeq(tlen, tlen_)
        aeq(slen, slen_)

        if include_extra:
            batch_, tlen_, glen = extra_attn.size()
            batch, glen_, gvocab = extra_map.size()
            aeq(glen, glen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, :, self.pad_index] = -self.eps
        prob = self.softmax(logits)

        if not include_extra:
            # Probability of copying p(z=1) batch.
            p_copy = self.sigmoid(self.linear_copy(hidden))
            # Probibility of not copying: p_{word}(w) * (1 - p(z))
            out_prob = torch.mul(prob, 1 - p_copy.expand_as(prob))
            mul_attn = torch.mul(attn, p_copy.expand_as(attn))
            # `[batch, tlen, extra_words]`
            copy_prob = torch.bmm(mul_attn, src_map)
            return torch.cat([out_prob, copy_prob], 2)
        else:
            p_copy = self.softmax_copy(self.linear_copy(hidden))

            out_prob = torch.mul(prob, p_copy[:, :, 0:1].expand_as(prob))
            mul_attn_1 = torch.mul(attn, p_copy[:, :, 1:2].expand_as(attn))
            mul_attn_2 = torch.mul(
                extra_attn, p_copy[:, :, 2:3].expand_as(extra_attn))

            copy_prob_1 = torch.bmm(mul_attn_1, src_map)
            copy_prob_2 = torch.bmm(mul_attn_2, extra_map)

            return torch.cat([out_prob, copy_prob_1, copy_prob_2], 2)


class CopyGeneratorCriterion(object):
    """ Copy generator criterion """

    def __init__(self, vocab_size, force_copy, tokenizer=None, eps=1e-20, include_extra=False):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.tokenizer = tokenizer
        self.include_extra = include_extra

    def __call__(self, scores, align, target, extra_align=None, first_offset=0):
        # CHECKS
        batch, tlen, _ = scores.size()
        _, _tlen = target.size()
        aeq(tlen, _tlen)
        _, _tlen = align.size()
        aeq(tlen, _tlen)

        if self.include_extra:
            _, __tlen = extra_align.size()
            aeq(__tlen, _tlen)
            extra_align = extra_align.view(-1)

        align = align.view(-1)
        target = target.view(-1)
        scores = scores.view(-1, scores.size(2))

        # Compute unks in align and target for readability
        # TODO 对于src_vocab来说unk id是1！！！很重要
        align_unk_id = 1
        target_unk_id = self.tokenizer.unk_token_id

        align_unk = align.eq(align_unk_id).float()
        align_not_unk = align.ne(align_unk_id).float()

        target_unk = target.eq(target_unk_id).float()
        target_not_unk = target.ne(target_unk_id).float()

        if self.include_extra:
            extra_align_unk = extra_align.eq(align_unk_id).float()
            extra_align_not_unk = extra_align.ne(align_unk_id).float()
            extra_out = scores.gather(
                1, extra_align.view(-1, 1) + self.offset+first_offset).view(-1)
            extra_out = extra_out.mul(extra_align_not_unk) + self.eps

        # Copy probability of tokens in source
        out = scores.gather(1, align.view(-1, 1) + self.offset).view(-1)
        # Set scores for unk to 0 and add eps
        out = out.mul(align_not_unk) + self.eps
        # Get scores for tokens in target
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # Add score for non-unks in target
            out = out + tmp.mul(target_not_unk)
            # Add score for when word is unk in both align and tgt
            if self.include_extra:
                out = out + \
                    tmp.mul(align_unk).mul(target_unk).mul(extra_align_unk)
            else:
                out = out + tmp.mul(align_unk).mul(target_unk)
        else:
            # Forced copy. Add only probability for not-copied tokens
            if self.include_extra:
                out = out + tmp.mul(align_unk).mul(extra_align_unk)
            else:
                out = out + tmp.mul(align_unk)

        if self.include_extra:
            out += extra_out
        loss = -out.log()
        return loss
