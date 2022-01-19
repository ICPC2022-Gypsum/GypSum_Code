import copy
import logging
import math

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict

from c2nl.models.transformer_bert import TransformerBert
from .utils.parallel import BalancedDataParallel
from c2nl.utils.bert_copy_utils import collapse_copy_scores, replace_unknown, \
    make_src_map, align
from c2nl.utils.bert_misc import tens2sen
from .utils.utility import try_gpu

logger = logging.getLogger(__name__)


class Code2NL(object):
    """High level model that handles initialization the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, config, code_bert=None, tokenizer=None, state_dict=None):
        # Book-keeping.
        c_model = config.model
        self.c_model = c_model
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        self.optimizer = None
        self.include_graph = config.include_graph
        self.include_token = config.include_token
        self.network = TransformerBert(self.c_model, code_bert, tokenizer, self.include_graph,
                                       self.include_token)

        # Load saved state
        if state_dict:
            if config.parallel:
                state_dict = {k.replace('module.', '')
                                        : v for k, v in state_dict.items()}
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer(
                    'fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------
    # training progress
    def update(self, ex, tokenizer):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        source_map, alignment, graph_map, graph_alignment = None, None, None, None
        blank, fill, graph_blank, graph_fill = None, None, None, None

        # To enable copy attn, collect source map and alignment info
        if self.c_model.copy_attn:
            if self.include_token:
                assert 'src_map' in ex and 'alignment' in ex
                source_map = try_gpu(make_src_map(ex['src_map']))
                alignment = try_gpu(align(ex['alignment']))
                blank, fill = collapse_copy_scores(tokenizer, ex['src_vocab'])

            if self.include_graph:
                assert 'graph_map' in ex and 'graph_alignment' in ex
                graph_map = try_gpu(make_src_map(ex['graph_map']))
                graph_alignment = try_gpu(align(ex['graph_alignment']))
                graph_blank, graph_fill = collapse_copy_scores(
                    tokenizer, ex['graph_vocab'])

        code_ids = try_gpu(ex['code_ids'])
        code_mask = try_gpu(ex['code_mask'])
        code_lens = try_gpu(ex['code_lens'])
        summary_ids = try_gpu(ex['summary_ids'])
        summary_mask = try_gpu(ex['summary_mask'])
        summary_lens = try_gpu(ex['summary_lens'])
        target_seq = try_gpu(ex['target_seq'])

        if self.include_graph:
            graph_datas = ex['graph_data']
            graph_tokens = ex['graph_tokens']
            graph_idx = try_gpu(ex['graph_idx'].long())
        else:
            graph_datas = None
            graph_tokens = None
            graph_idx = None

        # Run forward
        net_loss = self.network(code_ids=code_ids,
                                code_lens=code_lens,
                                summary_ids=summary_ids,
                                summary_lens=summary_lens,
                                target_seq=target_seq,
                                src_map=source_map,
                                alignment=alignment,
                                blank=blank,
                                fill=fill,
                                source_vocab=ex['src_vocab'],
                                code_mask=code_mask,
                                summary_mask=summary_mask,
                                graph_data=graph_datas,
                                graph_tokens=graph_tokens,
                                graph_map=graph_map,
                                graph_alignment=graph_alignment,
                                graph_blank=graph_blank,
                                graph_fill=graph_fill,
                                graph_idx=graph_idx
                                )

        loss = net_loss['ml_loss'].mean() if self.parallel \
            else net_loss['ml_loss']

        loss_per_token = net_loss['loss_per_token'].mean() if self.parallel \
            else net_loss['loss_per_token']

        ml_loss = loss.item()
        loss_per_token = loss_per_token.item()
        loss_per_token = 10 if loss_per_token > 10 else loss_per_token
        perplexity = math.exp(loss_per_token)
        self.optimizer.zero_grad()

        loss.backward()

        clip_grad_norm_(self.network.parameters(), self.c_model.grad_clipping)
        self.optimizer.step()

        self.updates += 1
        return {
            'ml_loss': ml_loss,
            'perplexity': perplexity
        }

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, tokenizer, replace_unk=False, return_attn=False):
        """Forward a batch of examples only to get predictions.
        Args:
            tokenizer
            ex: the batch examples
            replace_unk: replace `unk` tokens while generating predictions
        Output:
            predictions: #batch predicted sequences
        """
        # Eval mode
        self.network.eval()

        source_map, alignment, graph_map = None, None, None
        blank, fill, graph_blank, graph_fill = None, None, None, None

        # To enable copy attn, collect source map and alignment info
        if self.c_model.copy_attn:
            if self.include_token:
                assert 'src_map' in ex and 'alignment' in ex
                source_map = try_gpu(make_src_map(ex['src_map']))
                blank, fill = collapse_copy_scores(tokenizer, ex['src_vocab'])

            if self.include_graph:
                assert 'graph_map' in ex and 'graph_alignment' in ex
                graph_map = try_gpu(make_src_map(ex['graph_map']))
                graph_blank, graph_fill = collapse_copy_scores(
                    tokenizer, ex['graph_vocab'])

        code_ids = try_gpu(ex['code_ids'])
        code_mask = try_gpu(ex['code_mask'])
        code_lens = try_gpu(ex['code_lens'])

        if self.include_graph:
            graph_datas = ex['graph_data']
            graph_tokens = ex['graph_tokens']
            graph_idx = try_gpu(ex['graph_idx'].long())
        else:
            graph_datas = None
            graph_tokens = None
            graph_idx = None

        decoder_out = self.network(code_ids=code_ids,
                                   code_lens=code_lens,
                                   summary_ids=None,
                                   summary_lens=None,
                                   target_seq=None,
                                   src_map=source_map,
                                   alignment=alignment,
                                   blank=blank, fill=fill,
                                   source_vocab=ex['src_vocab'],
                                   code_mask=code_mask,
                                   graph_data=graph_datas,
                                   graph_tokens=graph_tokens,
                                   graph_blank=graph_blank, graph_fill=graph_fill,
                                   graph_map=graph_map,
                                   graph_vocab=ex['graph_vocab'],
                                   graph_idx=graph_idx,
                                   return_attn=return_attn)

        if self.include_token and self.include_graph:
            predictions = tens2sen(decoder_out['predictions'],
                                   self.tokenizer,
                                   ex['src_vocab'], ex['graph_vocab'], source_map.size(2))
        elif self.include_token:
            predictions = tens2sen(decoder_out['predictions'],
                                   self.tokenizer,
                                   ex['src_vocab'])
        elif self.include_graph:
            predictions = tens2sen(decoder_out['predictions'],
                                   self.tokenizer,
                                   ex['graph_vocab'])

            # TODO Fix raplace unk
            # if replace_unk:
            #     for i in range(len(predictions)):
            #         enc_dec_attn = decoder_out['attentions'][i]
            #         # assert enc_dec_attn.dim() == 3
            #         enc_dec_attn = enc_dec_attn.mean(1)
            #         predictions[i] = replace_unknown(predictions[i],
            #                                          enc_dec_attn,
            #                                          src_raw=ex['code_tokens'][i],
            #                                          unk_word=self.tokenizer.unk_token)

        last_pred = []
        for i in range(len(predictions)):
            ids = [tokenizer.convert_tokens_to_ids(
                w) for w in predictions[i].split(' ')]
            if tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(tokenizer.eos_token_id)]

            if tokenizer.pad_token_id in ids:
                ids = ids[:ids.index(tokenizer.pad_token_id)]

            if len(ids) == 0:
                ids = [tokenizer.pad_token_id]

            last_pred.append(tokenizer.decode(
                ids, clean_up_tokenization_spaces=False))

        targets = [summary for summary in ex['summary_texts']]
        return last_pred, targets, decoder_out['copy_info'], {'c_enc_attn': decoder_out['c_enc_attn'],
                                                              'g_enc_attn': decoder_out['g_enc_attn'],
                                                              'dec_attn': decoder_out['dec_attn']}

    '''
    Model save and load module
    '''

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'args': self.config.to_vanilla_(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'args': self.config.to_vanilla_(),
            'epoch': epoch,
            'updates': self.updates,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @ staticmethod
    def load(filename, code_bert, tokenizer, config):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        # TODO
        state_dict = saved_params['state_dict']
        model = Code2NL(config, code_bert, tokenizer, state_dict)
        return model

    @ staticmethod
    def load_checkpoint(filename, config, code_bert, tokenizer, use_gpu=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        epoch = saved_params['epoch']
        updates = saved_params['updates']
        optimizer = saved_params['optimizer']
        state_dict = saved_params['state_dict']
        model = Code2NL(config, code_bert, tokenizer, state_dict)
        model.updates = updates
        model.init_optimizer(optimizer, use_gpu)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = try_gpu(self.network)

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = BalancedDataParallel(24, self.network, dim=0)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        if self.c_model.optimizer == 'sgd':
            parameters = [p for p in self.network.parameters()
                          if p.requires_grad]
            self.optimizer = optim.SGD(parameters,
                                       self.c_model.learning_rate,
                                       momentum=self.c_model.momentum,
                                       weight_decay=self.c_model.weight_decay)

        elif self.c_model.optimizer == 'adam':
            parameters = [p for p in self.network.parameters()
                          if p.requires_grad]
            self.optimizer = optim.Adam(parameters,
                                        self.c_model.learning_rate,
                                        weight_decay=self.c_model.weight_decay)

        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.c_model.optimizer)

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = try_gpu(v)
