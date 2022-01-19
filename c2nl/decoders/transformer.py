"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from c2nl.decoders.decoder import DecoderBase
from c2nl.modules.multi_head_attn import MultiHeadedAttention
from c2nl.modules.position_ffn import PositionwiseFeedForward
from c2nl.utils.misc import sequence_mask
from c2nl.modules.util_class import LayerNorm
import copy


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
    """

    def __init__(self,
                 d_model,
                 heads,
                 d_k,
                 d_v,
                 d_ff,
                 dropout,
                 max_relative_positions=0,
                 coverage_attn=False,
                 include_graph=False,
                 include_token=True
                 ):
        super(TransformerDecoderLayer, self).__init__()

        self.include_graph = include_graph
        self.include_token = include_token

        self.attention = MultiHeadedAttention(
            heads, d_model, d_k, d_v, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.layer_norm = LayerNorm(d_model)

        if include_token:
            self.context_attn = MultiHeadedAttention(
                heads, d_model, d_k, d_v, dropout=dropout, coverage=coverage_attn)

        if include_graph:
            self.context_attn_1 = MultiHeadedAttention(
                heads, d_model, d_k, d_v, dropout=dropout,
                coverage=coverage_attn)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)
        # if include_ext:
        #     self.fusion_linear = nn.Linear(d_model * 2, d_model)

    def forward(self,
                inputs,
                memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=None,
                step=None,
                coverage=None,
                graph_layer_cache=None,
                graph_out=None,
                graph_pad_mask=None
                ):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``
            layer_cache
            step
            coverage
            graph_out
            graph_pad_mask
            graph_layer_cache
        Returns:
            (FloatTensor, FloatTensor):
            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``
        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        query, _, _ = self.attention(inputs,
                                     inputs,
                                     inputs,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     attn_type="self")

        query_norm = self.layer_norm(self.drop(query) + inputs)
        mid = None
        graph_attn = None
        attn = None

        if self.include_token and memory_bank is not None:
            mid, attn, coverage = self.context_attn(memory_bank,
                                                    memory_bank,
                                                    query_norm,
                                                    mask=src_pad_mask,
                                                    layer_cache=layer_cache,
                                                    attn_type="context",
                                                    step=step,
                                                    coverage=coverage)

        if self.include_graph and graph_out is not None:
            # if self.include_token:
            #     query_norm = self.layer_norm(self.drop(mid) + query_norm)

            graph_mid, graph_attn, graph_coverage = self.context_attn_1(graph_out,
                                                                        graph_out,
                                                                        query_norm,
                                                                        mask=graph_pad_mask,
                                                                        layer_cache=graph_layer_cache,
                                                                        attn_type="context",
                                                                        step=step,
                                                                        coverage=None)
            if self.include_token:
                mid = graph_mid + mid
            else:
                mid = graph_mid

        mid_norm = self.layer_norm(self.drop(mid) + query_norm)

        output = self.feed_forward(mid_norm)

        # attn = attn if self.include_token else graph_attn

        return output, attn, coverage, graph_attn


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O
    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self,
                 num_layers,
                 d_model=512,
                 heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 dropout=0.2,
                 max_relative_positions=0,
                 coverage_attn=False,
                 include_graph=False,
                 include_token=True
                 ):
        super(TransformerDecoder, self).__init__()

        self.include_token = include_token
        self.include_graph = include_graph

        self.num_layers = num_layers

        if isinstance(max_relative_positions, int):
            max_relative_positions = [max_relative_positions] * self.num_layers

        self._coverage = coverage_attn

        self.layer = nn.ModuleList(
            [TransformerDecoderLayer(d_model,
                                     heads,
                                     d_k,
                                     d_v,
                                     d_ff,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     coverage_attn=coverage_attn,
                                     include_graph=include_graph,
                                     include_token=include_token)
             for i in range(num_layers)])

    def init_state(self, src_len=None, max_len=None, graph_len=None, max_graph_len=None):
        """Initialize decoder state."""
        state = dict()
        state["src_len"] = src_len  # [B]
        state["src_max_len"] = max_len  # an integer
        state['graph_len'] = graph_len
        state['graph_max_len'] = max_graph_len
        state["cache"] = None
        return state

    def count_parameters(self):
        params = list(self.layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self,
                tgt_pad_mask,
                emb,
                memory_bank,
                state,
                step=None,
                layer_wise_coverage=None,
                graph_outs=None
                ):
        if step == 0:
            self._init_cache(state)

        include_graph = self.include_graph
        include_token = self.include_token

        graph_pad_mask = None
        src_pad_mask = None

        assert emb.dim() == 3  # batch x len x embedding_dim

        output = emb

        if include_token:
            src_pad_mask = ~sequence_mask(
                state["src_len"], max_len=state["src_max_len"]).unsqueeze(1)

        if include_graph:
            graph_pad_mask = ~sequence_mask(
                state["graph_len"], max_len=state["graph_max_len"]).unsqueeze(1)

        tgt_pad_mask = tgt_pad_mask.unsqueeze(1)  # [B, 1, T_tgt]

        new_layer_wise_coverage = []
        representations = []
        std_attentions = []
        graph_std_attentions = []
        # print(state.keys())
        for i, layer in enumerate(self.layer):
            layer_cache = state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            graph_layer_cache = state["cache"]["graph_layer_{}".format(i)] \
                if step is not None else None

            graph_bank = None
            mem_bank = None

            if include_token:
                mem_bank = memory_bank[i] if isinstance(
                    memory_bank, list) else memory_bank

            if include_graph:
                graph_bank = graph_outs[i] if isinstance(
                    graph_outs, list) else graph_outs

            output, attn, coverage, graph_attn = layer(
                output,
                mem_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                coverage=None if layer_wise_coverage is None
                else layer_wise_coverage[i],
                graph_layer_cache=graph_layer_cache,
                graph_out=graph_bank,
                graph_pad_mask=graph_pad_mask
            )
            representations.append(output)
            std_attentions.append(attn)
            graph_std_attentions.append(graph_attn)
            new_layer_wise_coverage.append(coverage)

        attns = dict()
        attns["std"] = std_attentions[-1]
        attns['graph_std'] = graph_std_attentions[-1]
        attns["coverage"] = None

        if self._coverage:
            attns["coverage"] = new_layer_wise_coverage
        return representations, attns

    def _init_cache(self, state):
        state["cache"] = {}
        for i, layer in enumerate(self.layer):
            layer_cache = dict()
            layer_cache["memory_keys"] = None
            layer_cache["memory_values"] = None
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            state["cache"]["layer_{}".format(i)] = layer_cache
            state["cache"]["graph_layer_{}".format(
                i)] = copy.deepcopy(layer_cache)

    def reorder_state_cache(self, state, index):
        for i, layer in enumerate(self.layer):
            layer_cache = state['cache']["layer_{}".format(i)]
            graph_layer_cache = state["cache"]["graph_layer_{}".format(i)]
            for key in layer_cache:
                if layer_cache[key] is not None:
                    layer_cache[key].data.copy_(
                        layer_cache[key].data.index_select(0, index))
                if graph_layer_cache[key] is not None:
                    graph_layer_cache[key].data.copy_(
                        graph_layer_cache[key].data.index_select(0, index))
            state["cache"]["layer_{}".format(i)] = layer_cache
            state["cache"]["graph_layer_{}".format(i)] = graph_layer_cache

        return state
