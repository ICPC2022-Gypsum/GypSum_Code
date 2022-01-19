import copy
import torch
import torch.nn as nn
import torch.nn.functional as f

from gypsum.utils.bert_utils import Beam
from gypsum.modules.encoder_decoder import Decoder
from c2nl.modules.bert_copy_generator import CopyGenerator, CopyGeneratorCriterion
from c2nl.modules.global_attention import GlobalAttention
from gypsum.modules.encoders import Code2SeqEncoder, GraphEncoder
from gypsum.utils.utility import try_gpu


class TransformerBert(nn.Module):
    """Module that writes an answer for the question given a passage."""

    def __init__(self, config, code_bert, tokenizer, include_graph=False, include_token=False):
        """"Constructor of the class."""
        super(TransformerBert, self).__init__()

        self.name = 'TransformerBert'
        self.config = config
        self.include_token = include_token
        self.vocab_size = tokenizer.vocab_size
        self.tokenizer = tokenizer

        # TODO Set superparameter t
        self.temperature = 0.5

        self.encoder = code_bert

        self.decoder = Decoder(
            config, config.emsize, include_graph=include_graph, include_token=include_token)

        self.layer_wise_attn = config.layer_wise_attn

        # ========================================================
        # Constrative Learning
        self.use_cons = config.use_cons

        self.generator = nn.Linear(
            self.decoder.input_size, tokenizer.vocab_size)
        # =================================================================

        # =================================================================
        # begin graph network
        self.include_graph = include_graph

        graph_config = config.graph
        self.graph_config = graph_config

        if include_graph:
            self.graph_encoder = GraphEncoder(graph_config, config.emsize)
        # =================================================================

        if config.share_decoder_embeddings:
            self.tie_weights()

        # if self.use_cons:
        #     self.cons_loss = nn.CosineSimilarity(dim=-1)
        #     cons_dim = config.cons_dim
        #     self.graph_linear = nn.Linear(config.emsize, cons_dim)
        #     self.token_linear = nn.Linear(config.emsize, cons_dim)

        self._copy = config.copy_attn
        if self._copy:
            include_extra = include_token and include_graph
            # self.copy_attn = GlobalAttention(dim=self.decoder.input_size,
            #                                  attn_type=config.attn_type)
            self.copy_generator = CopyGenerator(self.decoder.input_size,
                                                tokenizer.pad_token_id,
                                                self.generator, include_extra=include_extra)
            self.criterion = CopyGeneratorCriterion(vocab_size=self.vocab_size,
                                                    force_copy=config.force_copy, tokenizer=tokenizer, include_extra=include_extra)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of wither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.generator,
                                   self.encoder.embeddings.word_embeddings)

    def cal_cons_loss(self, emb1, emb2):
        batch_size, feature_dim = emb1.shape

        anchor_feature = emb1
        contrast_feature = emb2

        anchor_dot_contrast = self.cons_loss(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                             torch.transpose(contrast_feature.expand((batch_size, batch_size, feature_dim)), 0, 1))

        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast,
                                           self.temperature)).diag().sum()

        return loss

    def _run_forward_ml(self,
                        code_ids,
                        code_lens,
                        summary_ids,
                        summary_lens,
                        target_seq,
                        src_map,
                        alignment,
                        **kwargs):

        # embed and encode the source sequence
        # code_emb = self.encoder.embeddings(code_word_rep)
        # source_mask 0->padding
        source_mask = kwargs['code_mask']
        target_mask = kwargs['summary_mask']

        # Encoder
        graph_lens = None
        graph_out = None
        memory_bank = None

        if self.include_token:
            # B x seq_len x h
            outputs = self.encoder(code_ids, attention_mask=source_mask)
            memory_bank = outputs[0]

        if self.include_graph:
            graph_data = kwargs['graph_data']
            graph_tokens = kwargs['graph_tokens']
            graph_idx = kwargs['graph_idx']

            graph_out, graph_lens, g_attn = self.graph_encoder(
                graph_data, graph_tokens, graph_idx=graph_idx, encoder=self.encoder, tokenizer=None)

        # embed and encode the target sequence
        summary_emb = self.encoder.embeddings(summary_ids)
        summary_pad_mask = (1 - target_mask).bool()  # 1 for padding

        # if self.use_cons and self.include_graph and self.include_token:
        #     graph_cls = self.graph_linear(graph_out[:, 0, :])
        #     token_cls = self.token_linear(memory_bank[:, 0, :])

        #     cons_loss = self.cal_cons_loss(graph_cls, token_cls)

        layer_wise_dec_out, attn = self.decoder(memory_bank,
                                                code_lens,
                                                summary_pad_mask,
                                                summary_emb,
                                                graph_out,
                                                graph_lens)

        decoder_outputs = layer_wise_dec_out[-1]

        loss = dict()
        target = target_seq[:, 1:].contiguous()
        if self._copy:

            if self.include_token:
                # copy_score: batch_size, tgt_len, src_len
                copy_score = torch.cat([i.unsqueeze(3)
                                       for i in attn['std']], 3).mean(-1)

            if self.include_graph:
                graph_score = torch.cat([i.unsqueeze(3)
                                         for i in attn['graph_std']], 3).mean(-1)
                graph_map = kwargs['graph_map']
                graph_alignment = kwargs['graph_alignment']

            if self.include_token and self.include_graph:
                scores = self.copy_generator(
                    decoder_outputs, copy_score, src_map, extra_attn=graph_score, extra_map=graph_map)
                scores = scores[:, :-1, :].contiguous()
                first_offset = src_map.size(2)
                ml_loss = self.criterion(
                    scores, alignment[:, 1:].contiguous(), target,
                    extra_align=graph_alignment[:, 1:].contiguous(), first_offset=first_offset)
            elif self.include_token:
                scores = self.copy_generator(
                    decoder_outputs, copy_score, src_map)
                scores = scores[:, :-1, :].contiguous()
                ml_loss = self.criterion(
                    scores, alignment[:, 1:].contiguous(), target)
            elif self.include_graph:
                scores = self.copy_generator(
                    decoder_outputs, graph_score, graph_map)
                scores = scores[:, :-1, :].contiguous()
                ml_loss = self.criterion(
                    scores, graph_alignment[:, 1:].contiguous(), target)

        else:
            # `batch x tgt_len x vocab_size`
            scores = self.generator(decoder_outputs)
            # `batch x tgt_len - 1 x vocab_size`
            scores = scores[:, :-1, :].contiguous()

            ml_loss = self.criterion(scores.view(-1, scores.size(2)),
                                     target=target.view(-1))

        ml_loss = ml_loss.view(*scores.size()[:-1])
        ml_loss = ml_loss.mul(target.ne(self.tokenizer.pad_token_id).float())

        # TODO fix loss
        ml_loss = ml_loss.sum(1)

        loss['ml_loss'] = ml_loss.mean()
        loss['loss_per_token'] = (ml_loss).div(
            (summary_lens - 1).float()).mean()

        return loss

    def forward(self,
                code_ids,
                code_lens,
                summary_ids,
                summary_lens,
                target_seq,
                src_map,
                alignment,
                **kwargs):
        """
        Input:
            - code_ids: ``(batch_size, max_doc_len)``
            - code_lens: ``(batch_size)``
            - summary_ids: ``(batch_size, max_que_len)``
            - summary_lens: ``(batch_size)``
            - target_seq: ``(batch_size, max_len)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        if self.training:
            return self._run_forward_ml(code_ids,
                                        code_lens,
                                        summary_ids,
                                        summary_lens,
                                        target_seq,
                                        src_map,
                                        alignment,
                                        **kwargs)

        else:
            return self.decode(code_ids,
                               code_lens,
                               src_map,
                               alignment,
                               **kwargs)

    def con2src(self, t, src_vocabs, extra_vocabs=None, extra_offset=1e5):
        words = []
        for idx, w in enumerate(t):
            token_id = w[0].item()
            if token_id < self.vocab_size:
                # TODO check if correct
                words.append(self.tokenizer.convert_ids_to_tokens(token_id))
            elif self.vocab_size <= token_id < self.vocab_size + extra_offset:
                token_id = token_id - self.vocab_size
                words.append(src_vocabs[idx][token_id])
            else:
                token_id = token_id - self.vocab_size - extra_offset
                words.append(extra_vocabs[idx][token_id])
        return words

    def __greedy_sequence(self,
                          params,
                          choice='greedy',
                          tgt_words=None):

        memory_bank = params['memory_bank']
        graph_outs = params['graph_outs']
        graph_lens = params['graph_lens']
        src_len = params['src_len']
        src_map = params['src_map']

        batch_size = memory_bank.size(
            0) if memory_bank is not None else graph_outs.size(0)
        device = memory_bank.device if memory_bank is not None else graph_outs.device

        if tgt_words is None:
            tgt_words = torch.LongTensor(
                [self.tokenizer.cls_token_id]).to(device)
            tgt_words = tgt_words.expand(
                batch_size).unsqueeze(1).long()  # B x 1

        dec_predictions = []
        copy_info = []
        attentions = []
        dec_log_prob = []
        acc_dec_outs = []

        max_mem_len = None
        if self.include_token:
            max_mem_len = params['memory_bank'][0].shape[1] \
                if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]

        max_graph_len = None
        if self.include_graph:
            max_graph_len = graph_outs[0].shape[1] \
                if isinstance(graph_outs, list) else graph_outs.shape[1]

        dec_states = self.decoder.init_decoder(
            params['src_len'], max_mem_len, graph_lens, max_graph_len)

        attn = {"coverage": None}

        enc_outputs = memory_bank
        all_attns = {'std': [], 'graph_std': []}
        # +1 for <EOS> token
        for idx in range(self.config.max_len + 1):
            tgt = self.encoder.embeddings(tgt_words)[:, -1:, :]
            tgt_pad_mask = tgt_words.data.eq(self.tokenizer.pad_token_id)
            layer_wise_dec_out, attn = self.decoder.decode(tgt_pad_mask,
                                                           tgt,
                                                           enc_outputs,
                                                           dec_states,
                                                           step=idx,
                                                           layer_wise_coverage=attn['coverage'],
                                                           graph_outs=graph_outs)
            copy_score = None
            graph_score = None

            if attn['std'] is not None:
                copy_score = torch.cat([i.unsqueeze(3)
                                       for i in attn['std']], 3).mean(-1)
                all_attns['std'].append(attn['std'])

            if attn['graph_std'] is not None:
                graph_score = torch.cat([i.unsqueeze(3)
                                        for i in attn['graph_std']], 3).mean(-1)
                all_attns['graph_std'].append(attn['graph_std'])

            decoder_outputs = layer_wise_dec_out[-1]
            acc_dec_outs.append(decoder_outputs.squeeze(1))

            if self._copy:
                include_extra = self.include_token and self.include_graph
                if include_extra:
                    graph_map = params['graph_map']
                    first_offset = src_map.size(2)

                    prediction = self.copy_generator(
                        decoder_outputs, copy_score, src_map, extra_attn=graph_score, extra_map=graph_map)
                    prediction = prediction.squeeze(1)

                    token_blank, token_fill = params['blank'], params['fill']
                    graph_blank, graph_fill = params['graph_blank'], params['graph_fill']

                    for b in range(prediction.size(0)):
                        if token_blank[b]:
                            blank_b = try_gpu(
                                torch.LongTensor(token_blank[b]))
                            fill_b = try_gpu(
                                torch.LongTensor(token_fill[b]))
                            prediction[b].index_add_(0, fill_b,
                                                     prediction[b].index_select(0, blank_b))
                            prediction[b].index_fill_(0, blank_b, 1e-10)
                        if graph_blank[b]:
                            blank_b = try_gpu(
                                torch.LongTensor(graph_blank[b]))+first_offset
                            fill_b = try_gpu(
                                torch.LongTensor(graph_fill[b]))
                            prediction[b].index_add_(0, fill_b,
                                                     prediction[b].index_select(0, blank_b))
                            prediction[b].index_fill_(0, blank_b, 1e-10)
                elif self.include_token:
                    prediction = self.copy_generator(
                        decoder_outputs, copy_score, src_map)
                    prediction = prediction.squeeze(1)
                    token_blank, token_fill = params['blank'], params['fill']
                    for b in range(prediction.size(0)):
                        if token_blank[b]:
                            blank_b = try_gpu(
                                torch.LongTensor(token_blank[b]))
                            fill_b = try_gpu(
                                torch.LongTensor(token_fill[b]))
                            prediction[b].index_add_(0, fill_b,
                                                     prediction[b].index_select(0, blank_b))
                            prediction[b].index_fill_(0, blank_b, 1e-10)
                elif self.include_graph:
                    graph_map = params['graph_map']

                    prediction = self.copy_generator(
                        decoder_outputs, graph_score, graph_map)
                    prediction = prediction.squeeze(1)
                    graph_blank, graph_fill = params['graph_blank'], params['graph_fill']
                    for b in range(prediction.size(0)):
                        if graph_blank[b]:
                            blank_b = try_gpu(
                                torch.LongTensor(graph_blank[b]))
                            fill_b = try_gpu(
                                torch.LongTensor(graph_fill[b]))
                            prediction[b].index_add_(0, fill_b,
                                                     prediction[b].index_select(0, blank_b))
                            prediction[b].index_fill_(0, blank_b, 1e-10)

            else:
                prediction = self.generator(decoder_outputs.squeeze(1))
                prediction = f.softmax(prediction, dim=1)

            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)
            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
            else:
                assert False

            dec_log_prob.append(log_prob.squeeze(1))
            dec_predictions.append(tgt.squeeze(1).clone())
            if "std" in attn and attn['std'] is not None:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attn["std"], dim=1)
                attentions.append(std_attn.squeeze(2))

            if self._copy:
                mask = tgt.gt(self.vocab_size - 1)
                copy_info.append(mask.float().squeeze(1))
            if self.include_token and self.include_graph:
                words = self.con2src(
                    tgt, params['source_vocab'], params['graph_vocab'], src_map.size(2))
            elif self.include_token:
                words = self.con2src(tgt, params['source_vocab'])
            elif self.include_graph:
                words = self.con2src(tgt, params['graph_vocab'])

            words = [self.tokenizer.convert_tokens_to_ids(w) for w in words]
            words = torch.Tensor(words).type_as(tgt).unsqueeze(1)

            tgt_words = torch.cat(
                [tgt_words, words], dim=1)
        return dec_predictions, attentions, copy_info, all_attns

    def __generate_sequence(self,
                            params):
        beam_size = self.config.beam_size
        token_out = params['memory_bank']
        graph_out = params['graph_outs']

        batch_size = token_out.size(
            0) if token_out is not None else graph_out.size(0)
        dec_predictions = []
        acc_dec_outs = []
        all_attentions = []
        all_copy_info = []

        device = token_out.device if token_out is not None else graph_out.device

        max_mem_len = None
        if self.include_token:
            max_mem_len = token_out[0].shape[1] \
                if isinstance(token_out, list) else token_out.shape[1]

        max_graph_len = None
        if self.include_graph:
            max_graph_len = graph_out[0].shape[1] \
                if isinstance(graph_out, list) else graph_out.shape[1]

        for bid in range(batch_size):
            beam = Beam(beam_size, self.tokenizer.cls_token_id,
                        self.tokenizer.eos_token_id)

            tgt_words = beam.getCurrentState().to(device).long()  # 1 x 1

            token_beam, src_len, graph_beam, graph_len = None, None, None, None
            if self.include_token:
                token_beam = token_out[bid:bid +
                                       1].repeat(beam_size, 1, 1)  # 1*seq*emb
                src_len = params['src_len'][bid:bid + 1].repeat(beam_size)
                src_map = params['src_map'][bid:bid +
                                            1].repeat(beam_size, 1, 1)
                token_blank = params['blank'][bid:bid + 1] * beam_size
                token_fill = params['fill'][bid:bid + 1] * beam_size
                source_vocab = params['source_vocab'][bid:bid + 1] * beam_size

            if self.include_graph:
                graph_beam = graph_out[bid:bid + 1].repeat(beam_size, 1, 1)
                graph_len = params['graph_lens'][bid:bid + 1].repeat(beam_size)
                graph_map = params['graph_map'][bid:bid +
                                                1].repeat(beam_size, 1, 1)
                graph_blank = params['graph_blank'][bid:bid + 1] * beam_size
                graph_fill = params['graph_fill'][bid:bid + 1] * beam_size
                graph_vocab = params['graph_vocab'][bid:bid + 1] * beam_size

            attentions = None
            copy_info = None

            dec_states = self.decoder.init_decoder(
                src_len, max_mem_len, graph_len, max_graph_len)

            attn = {"coverage": None}

            for idx in range(self.config.max_len + 1):
                if beam.done():
                    break
                tgt = self.encoder.embeddings(tgt_words)
                tgt_pad_mask = tgt_words.data.eq(
                    self.tokenizer.pad_token_id)

                layer_wise_dec_out, attn = self.decoder.decode(tgt_pad_mask,
                                                               tgt,
                                                               token_beam,
                                                               dec_states,
                                                               step=idx,
                                                               layer_wise_coverage=attn['coverage'],
                                                               graph_outs=graph_beam)

                copy_score = None
                graph_score = None

                if attn['std'] is not None:
                    copy_score = torch.cat([i.unsqueeze(3)
                                            for i in attn['std']], 3).mean(-1)[:, idx:idx + 1, :]

                if attn['graph_std'] is not None:
                    graph_score = torch.cat([i.unsqueeze(3)
                                             for i in attn['graph_std']], 3).mean(-1)[:, idx:idx + 1, :]

                decoder_outputs = layer_wise_dec_out[-1]
                acc_dec_outs.append(decoder_outputs.squeeze(1))

                decoder_outputs = decoder_outputs[:, idx:idx + 1, :]

                if self._copy:
                    include_extra = self.include_token and self.include_graph
                    if include_extra:
                        first_offset = src_map.size(2)

                        prediction = self.copy_generator(
                            decoder_outputs, copy_score, src_map, extra_attn=graph_score, extra_map=graph_map)
                        prediction = prediction.squeeze(1)

                        for b in range(prediction.size(0)):
                            if token_blank[b]:
                                blank_b = try_gpu(
                                    torch.LongTensor(token_blank[b]))
                                fill_b = try_gpu(
                                    torch.LongTensor(token_fill[b]))
                                prediction[b].index_add_(0, fill_b,
                                                         prediction[b].index_select(0, blank_b))
                                prediction[b].index_fill_(0, blank_b, 1e-10)
                            if graph_blank[b]:
                                blank_b = try_gpu(
                                    torch.LongTensor(graph_blank[b]))+first_offset
                                fill_b = try_gpu(
                                    torch.LongTensor(graph_fill[b]))
                                prediction[b].index_add_(0, fill_b,
                                                         prediction[b].index_select(0, blank_b))
                                prediction[b].index_fill_(0, blank_b, 1e-10)
                    elif self.include_token:
                        prediction = self.copy_generator(
                            decoder_outputs, copy_score, src_map)
                        prediction = prediction.squeeze(1)

                        for b in range(prediction.size(0)):
                            if token_blank[b]:
                                blank_b = try_gpu(
                                    torch.LongTensor(token_blank[b]))
                                fill_b = try_gpu(
                                    torch.LongTensor(token_fill[b]))
                                prediction[b].index_add_(0, fill_b,
                                                         prediction[b].index_select(0, blank_b))
                                prediction[b].index_fill_(0, blank_b, 1e-10)
                    elif self.include_graph:
                        prediction = self.copy_generator(
                            decoder_outputs, graph_score, graph_map)
                        prediction = prediction.squeeze(1)
                        for b in range(prediction.size(0)):
                            if graph_blank[b]:
                                blank_b = try_gpu(
                                    torch.LongTensor(graph_blank[b]))
                                fill_b = try_gpu(
                                    torch.LongTensor(graph_fill[b]))
                                prediction[b].index_add_(0, fill_b,
                                                         prediction[b].index_select(0, blank_b))
                                prediction[b].index_fill_(0, blank_b, 1e-10)

                else:
                    prediction = self.generator(decoder_outputs.squeeze(1))
                    prediction = torch.log_softmax(prediction, dim=1)

                beam.advance(prediction.data)
                tgt = beam.getCurrentState()

                tgt_words.data.copy_(
                    tgt_words.data.index_select(0, beam.getCurrentOrigin()))

                dec_states = self.decoder.reorder_state(
                    dec_states, beam.getCurrentOrigin())

                if "std" in attn and attn['std'] is not None:
                    # std_attn: batch_size x num_heads x 1 x src_len
                    std_attn = torch.stack(attn["std"], dim=1)
                    attentions = std_attn.squeeze(2)

                if self._copy and idx > 1:
                    mask = tgt_words.gt(self.vocab_size - 1)
                    copy_info = mask.float().squeeze(1)

                if self.include_token and self.include_graph:
                    words = self.con2src(
                        tgt, source_vocab, graph_vocab, src_map.size(2))
                elif self.include_token:
                    words = self.con2src(tgt, source_vocab)
                elif self.include_graph:
                    words = self.con2src(tgt, graph_vocab)

                words = [self.tokenizer.convert_tokens_to_ids(
                    w) for w in words]
                words = torch.Tensor(words).type_as(tgt).unsqueeze(1)

                tgt_words = torch.cat(
                    [tgt_words, words], dim=1)
            dec_predictions.append(tgt_words[0][1:])
            if attentions is not None:
                all_attentions.append(attentions[0].transpose(0, 1))
            if copy_info is not None:
                all_copy_info.append(copy_info[0])
        return dec_predictions, all_attentions, all_copy_info, attn

    def decode(self,
               code_ids,
               code_lens,
               src_map,
               alignment,
               **kwargs):

        memory_bank = None

        return_attn = kwargs['return_attn']

        c_en_attn = None

        if self.include_token:
            source_mask = kwargs['code_mask'].to(code_ids.device)
            # target_mask = kwargs['summary_mask'].to(code_ids.device)
            # B x seq_len x h
            outputs = self.encoder(code_ids, attention_mask=source_mask)
            # from pdb import set_trace
            # set_trace()
            memory_bank = outputs[0]
            if return_attn:
                c_en_attn = outputs.attentions
            # memory_bank, layer_wise_outputs = self.encoder(code_word_rep)  # B x seq_len x h

        # Encoder
        graph_lens = None
        graph_out = None

        params = dict()

        g_attn = None
        if self.include_graph:
            graph_data = kwargs['graph_data']
            graph_tokens = kwargs['graph_tokens']

            params['graph_map'] = kwargs['graph_map']
            params['graph_blank'] = kwargs['graph_blank']
            params['graph_fill'] = kwargs['graph_fill']
            params['graph_vocab'] = kwargs['graph_vocab']
            params['graph_idx'] = kwargs['graph_idx']

            graph_out, graph_lens, g_attn = self.graph_encoder(
                graph_data, graph_tokens, kwargs['graph_idx'], self.encoder)

        params['memory_bank'] = memory_bank
        params['layer_wise_outputs'] = None
        params['src_len'] = code_lens
        params['source_vocab'] = kwargs['source_vocab']
        params['src_map'] = src_map
        params['src_mask'] = kwargs['code_mask']
        params['fill'] = kwargs['fill']
        params['blank'] = kwargs['blank']
        params['src_words'] = code_ids
        params['graph_outs'] = graph_out
        params['graph_lens'] = graph_lens

        generate_sequence = self.__greedy_sequence if self.config.beam_size <= 1 else self.__generate_sequence
        dec_predictions, attentions, copy_info, attn = generate_sequence(
            params)
        if self.config.beam_size <= 1:
            dec_predictions = torch.stack(dec_predictions, dim=1)
            copy_info = torch.stack(copy_info, dim=1) if copy_info else None
            # attentions: batch_size x tgt_len x num_heads x src_len
            attentions = torch.stack(attentions, dim=1) if attentions else None

        return {
            'predictions': dec_predictions,
            'copy_info': copy_info,
            'memory_bank': memory_bank,
            'attentions': attentions,
            'c_enc_attn': c_en_attn,
            'g_enc_attn': g_attn,
            'dec_attn': attn
        }
