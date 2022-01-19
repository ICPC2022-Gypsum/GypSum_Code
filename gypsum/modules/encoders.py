import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from gypsum.utils.utility import try_gpu
from torch.nn.utils.rnn import pack_padded_sequence
try:
    from torch_geometric.nn import GatedGraphConv, GATConv
    from torch_geometric.data import Batch
except:
    pass

from gypsum.data.c2s_dict import PAD
import pickle as pkl


class GraphEncoder(nn.Module):
    def __init__(self, config, out_size):
        super(GraphEncoder, self).__init__()
        self.type_vocab_size = config.node_types
        self.max_node_len = config.max_node_len
        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout
        self.use_gat = config.gat

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        if not self.use_gat:
            self.graph_layer = GatedGraphConv(
                out_channels=self.hidden_size, num_layers=self.num_layers)
        else:
            self.graph_layer = GATConv(
                self.embed_size, self.hidden_size//8, heads=8, dropout=0.3)

        self.out_size = out_size

        self.embed_dropout = nn.Dropout(self.dropout_rate)

        self.out = nn.Linear(self.hidden_size*2, self.out_size)

    def forward(self, graph_data, graph_tokens, graph_idx, encoder, tokenizer=None):
        graph = Batch.from_data_list(graph_data)

        batch_size = graph.num_graphs
        # batch_list = graph.batch.detach().tolist()

        x, x_len, edge_index, edge_attr, indices, node_lens = graph.x, graph.x_len,\
            graph.edge_index, graph.edge_attr, graph.indice, graph.node_len

        x, edge_index, edge_attr = try_gpu(
            x), try_gpu(edge_index), try_gpu(edge_attr)

        encoder_embed = encoder.embeddings(graph_idx)

        x_embed = self.embed_dropout(self.embedding(x))
        attention = None
        if self.use_gat:
            graph_out, attention = self.graph_layer(
                x_embed, edge_index, return_attention_weights=True)
        else:
            graph_out = self.graph_layer(x_embed, edge_index, edge_attr)

        indices = indices[0].detach().numpy()

        x_len = x_len.detach().tolist()

        indices = [indices[sum(node_lens[:i]):sum(node_lens[:i+1])] + sum(x_len[:i])
                   for i in range(batch_size)]

        # TODO Concat with diff cls and pad and len plus 1
        max_node_len = min(self.max_node_len, max(node_lens))
        out = torch.cat([torch.cat([graph_out[indice], torch.zeros(max_node_len-node_lens[idx], self.hidden_size).to(x.device)], dim=0).unsqueeze(0)
                         for idx, indice in enumerate(indices)], 0)

        out = self.out(torch.cat([out, encoder_embed], -1))

        del x_len, indices

        return out, node_lens.to(x.device), attention


class Code2SeqEncoder(nn.Module):
    def __init__(self, input_size_sub_token, input_size_node, token_size, hidden_size, bidirectional=True, num_layers=2,
                 rnn_dropout=0.5, embeddings_dropout=0.25):
        """
        input_size_sub_token : # of unique subtoken
        input_size_node : # of unique node symbol
        token_size : embedded token size
        hidden_size : size of initial state of decoder
        rnn_dropout = 0.5 : rnn drop out ratio
        embeddings_dropout = 0.25 : dropout ratio for context vector
        """

        super(Code2SeqEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.token_size = token_size

        self.embedding_sub_token = nn.Embedding(
            input_size_sub_token, token_size, padding_idx=PAD)
        self.embedding_node = nn.Embedding(
            input_size_node, token_size, padding_idx=PAD)

        self.lstm = nn.LSTM(token_size, token_size, num_layers=num_layers, bidirectional=bidirectional,
                            dropout=rnn_dropout)
        self.out = nn.Linear(token_size * 4, hidden_size)

        self.dropout = nn.Dropout(embeddings_dropout)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

    def forward(self, batch_start, batch_node, batch_end, lengths_k, reverse_index, reverse_length, hidden=None):
        """
        batch_start : (B * k, l) start terminals' sub_token of each ast path
        batch_node : (l, B*k) non_terminals' nodes of each ast path
        batch_end : (B * k, l) end terminals' sub_token of each ast path

        lengths_k : length of k in each example
        reverse_index : index for un_sorting,
        """

        bk_size = batch_node.shape[1]

        # (B * k, l, d)
        encode_start = self.embedding_sub_token(batch_start)
        encode_end = self.embedding_sub_token(batch_end)

        # encode_S (B * k, d) token_representation of each ast path
        encode_start = encode_start.sum(1)
        encode_end = encode_end.sum(1)

        """
        LSTM Outputs: output, (h_n, c_n)
        output (seq_len, batch, num_directions * hidden_size)
        h_n    (num_layers * num_directions, batch, hidden_size) : tensor containing the hidden state for t = seq_len.
        c_n    (num_layers * num_directions, batch, hidden_size)
        """

        # emb_N :(l, B*k, d)
        emb_nodes = self.embedding_node(batch_node)
        packed = pack_padded_sequence(emb_nodes, reverse_length)
        output, (hidden, cell) = self.lstm(packed, hidden)
        # output, _ = pad_packed_sequence(output)

        # hidden (num_layers * num_directions, batch * k, hidden_size)
        # only last layer, (num_directions, batch, hidden_size)
        hidden = hidden[-self.num_directions:, :, :]

        # -> (Bk, num_directions, hidden_size)
        hidden = hidden.transpose(0, 1)

        # -> (Bk, 1, hidden_size * num_directions)
        hidden = hidden.contiguous().view(bk_size, 1, -1)

        # encode_N (Bk, hidden_size * num_directions)
        encode_nodes = hidden.squeeze(1)

        # encode_SNE  : (B*k, hidden_size * num_directions + 2)
        encode_concat = torch.cat(
            [encode_nodes, encode_start, encode_end], dim=1)

        # encode_SNE  : (B*k, d)
        encode_concat = self.out(encode_concat)

        # unsort as example
        # index = torch.tensor(index_N, d_type=torch.long, device=device)
        # encode_SNE = torch.index_select(encode_SNE, dim=0, index=index)
        index = np.argsort(reverse_index)
        encode_concat = encode_concat[[index]]

        # as is in  https://github.com/tech-srl/code2seq/blob/ec0ae309efba815a6ee8af88301479888b20daa9/model.py#L511
        encode_concat = self.dropout(encode_concat)

        # output_bag  : [ B, (k, d) ]
        tmp_bag = torch.split(encode_concat, lengths_k, dim=0)
        max_ast = max(lengths_k)
        outputs = torch.cat(
            [torch.cat([tmp_bag[i], torch.ones(max_ast - lengths_k[i], self.hidden_size).to(encode_concat.device)
                        * PAD], dim=0).unsqueeze(0) for i in range(len(tmp_bag))], dim=0
        )

        # hidden_0  : (1, B, d)
        # for decoder initial state
        hidden_0 = [ob.mean(0).unsqueeze(dim=0) for ob in tmp_bag]
        hidden_0 = torch.cat(hidden_0, dim=0).unsqueeze(dim=0)

        return outputs, hidden_0
