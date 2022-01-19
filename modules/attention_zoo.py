import torch


def dot_prod_attention(h_t, src_outputs, src_for_attn, mask=None):
    """
    :param h_t: (batch_size, hidden_size)
    :param src_outputs: (batch_size, src_sent_len, hidden_size * 2)
    :param src_for_attn: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len) mask is 1 for padding
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_for_attn, h_t.unsqueeze(2)).squeeze(2)
    _b_size, _src_len = att_weight.size()
    if mask:
        att_weight.data.masked_fill_(mask, -float('inf'))
    # 扩张一个time维度
    all_attn = att_weight.view((_b_size, 1, _src_len))
    att_weight = torch.softmax(att_weight, dim=-1)

    # (batch_size, hidden_size*2)
    ctx_vec = torch.bmm(att_weight.view(_b_size, 1, _src_len), src_outputs).squeeze(1)

    return ctx_vec, att_weight, all_attn
