import os
import pickle
import subprocess
import logging
from torch.utils.data import sampler
from tqdm import tqdm

from gypsum.data.dataset import CommentDataset, SortedBatchSampler
import gypsum.data.batch_handler as vector
from gypsum.model import Code2NL
from utils.bert_utils import *
from gypsum.data.utils import read_examples, convert_examples_to_features
from gypsum.data.dataloader import DataLoaderX, DataFetcher
from c2nl.inputters.timer import AverageMeter, Timer
from gypsum.data.c2s_dict import load_code2seq_dict
from gypsum.utils.utility import get_default_config, set_logging, load_bert, try_gpu

# 设置logger
logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Train loop
# ------------------------------------------------------------------------------
def train(m_config, tokenizer, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss = AverageMeter()

    perplexity = AverageMeter()
    epoch_time = Timer()
    c_model = m_config.model
    current_epoch = global_stats['epoch']
    progress_bar = tqdm(range(m_config.train_len // m_config.batch_size))

    progress_bar.set_description("%s" % 'Epoch = %d [perplexity = 0.00, ml_loss = 0.00]' %
                                 current_epoch)
    pre_fetcher = DataFetcher(data_loader)
    # 开始训练
    for _ in progress_bar:
        ex = pre_fetcher.next()
        bsz = ex['batch_size']
        # 设置learning rate
        if c_model.optimizer in ['sgd', 'adam'] and current_epoch <= c_model.warmup_epochs:
            cur_lr = global_stats['warmup_factor'] * (model.updates + 1)
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = cur_lr

        net_loss = model.update(ex, tokenizer)

        # TODO 判断是否使用cons
        ml_loss.update(net_loss['ml_loss'], bsz)

        perplexity.update(net_loss['perplexity'], bsz)
        log_info = 'Epoch = %d [perplexity = %.2f, ml_loss = %.2f]' % \
                   (current_epoch, perplexity.avg, ml_loss.avg)

        progress_bar.set_description("%s" % log_info)

    logger.info('train: Epoch %d | perplexity = %.2f | ml_loss = %.2f | '
                'Time for epoch = %.2f (s)' %
                (current_epoch, perplexity.avg, ml_loss.avg, epoch_time.time()))

    # Checkpoint
    if m_config.checkpoint:
        model.checkpoint(m_config.model_file +
                         '.checkpoint', current_epoch + 1)
    return ml_loss.avg

# ------------------------------------------------------------------------------
# Validation loops
# ------------------------------------------------------------------------------


def validate_official(m_config, data_loader, model, tokenizer, global_stats, mode='dev'):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    # Run through examples
    examples = 0
    sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()
    with torch.no_grad():
        progress_bar = tqdm(range(m_config.dev_len // m_config.batch_size))
        pre_fetcher = DataFetcher(data_loader)
        for idx in progress_bar:
            ex = pre_fetcher.next()
            batch_size = ex['batch_size']
            ex_ids = list(
                range(idx * batch_size, (idx * batch_size) + batch_size))
            predictions, targets, copy_info, _ = model.predict(
                ex, tokenizer, replace_unk=True)

            src_sequences = [code for code in ex['code_texts']]
            examples += batch_size

            for key, src, pred, tgt in zip(ex_ids, src_sequences, predictions, targets):
                hypotheses[key] = [pred]
                references[key] = tgt if isinstance(tgt, list) else [tgt]
                # print(pred)
                # print(tgt)
                # print('='*60)
                sources[key] = src

            if copy_info is not None:
                flag = isinstance(copy_info, list)
                if not flag:
                    copy_info = copy_info.cpu().numpy().astype(int).tolist()
                for key, cp in zip(ex_ids, copy_info):
                    if not flag:
                        copy_dict[key] = cp
                    else:
                        copy_dict[key] = cp.cpu().numpy().astype(int)

            progress_bar.set_description(
                "%s" % 'Epoch = %d [validating ... ]' % global_stats['epoch'])

    copy_dict = None if len(copy_dict) == 0 else copy_dict

    bleu, rouge_l, meteor, precision, recall, f1 = eval_accuracies(m_config,
                                                                   hypotheses,
                                                                   references,
                                                                   copy_dict,
                                                                   sources=sources,
                                                                   filename=m_config.pred_file,
                                                                   print_copy_info=m_config.print_copy_info,
                                                                   mode=mode)
    result = dict()
    result['bleu'] = bleu
    result['rouge_l'] = rouge_l
    result['meteor'] = meteor
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1

    if mode == 'test':
        logger.info('test valid official: '
                    'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                    (bleu, rouge_l, meteor) +
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | '
                    'examples = %d | ' %
                    (precision, recall, f1, examples) +
                    'test time = %.2f (s)' % eval_time.time())

    else:
        logger.info('dev valid official: Epoch = %d | ' %
                    (global_stats['epoch']) +
                    'bleu = %.2f | rouge_l = %.2f | '
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | examples = %d | ' %
                    (bleu, rouge_l, precision, recall, f1, examples) +
                    'valid time = %.2f (s)' % eval_time.time())

    return result


def load_model(config, code_bert, tokenizer):
    if config.only_test:
        if config.pretrained:
            model = Code2NL.load(
                config.pretrained, code_bert, tokenizer, config)
        else:
            if not os.path.isfile(config.model_file):
                raise IOError('No such file: %s' % config.model_file)
            model = Code2NL.load(
                config.model_file, code_bert, tokenizer, config)
    else:
        if config.checkpoint and os.path.isfile(config.model_file + '.checkpoint'):
            logger.info('Found a checkpoint...')
            checkpoint_file = config.model_file + '.checkpoint'
            model, start_epoch = Code2NL.load_checkpoint(checkpoint_file, config, code_bert, tokenizer,
                                                         config.cuda)
        else:
            if config.pretrained:
                logger.info('Using pretrained model...')
                model = Code2NL.load(
                    config.pretrained, code_bert, tokenizer, config)
            else:
                logger.info('Training model from scratch...')
                model = Code2NL(config, code_bert, tokenizer)

            # Set up optimizer
            model.init_optimizer()

    return model

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main(m_config):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load pretrained model')
    # 加载模型

    code_bert, tokenizer = load_bert(m_config)

    c_model = m_config.model

    train_exs = []

    '''
    Load Training Dataset
    '''
    if not m_config.only_test:
        src_examples = read_examples(m_config, 'train')
        train_exs = convert_examples_to_features(
            src_examples, tokenizer, m_config, stage='train', logger=logger)

        logger.info('Num train examples = %d' % len(train_exs))
        m_config.num_train_examples = len(train_exs)

    '''
    Load Validate Dataset
    '''
    dev_examples = read_examples(m_config, 'test')
    dev_exs = convert_examples_to_features(
        dev_examples, tokenizer, m_config, stage='dev', logger=logger)
    logger.info('Num dev examples = %d' % len(dev_exs))

    '''
    Model Initialization Or Loading
    '''
    logger.info('-' * 100)
    start_epoch = 1

    model = load_model(m_config, code_bert, tokenizer)

    '''
    Use GPU
    '''
    if m_config.cuda:
        model.cuda()
        if m_config.parallel:
            model.parallelize()
    else:
        model.cpu()

    '''
    Load Train And Dav Dataset
    '''
    logger.info('-' * 100)
    logger.info('Make data loaders')

    train_loader = None
    if not m_config.only_test:
        train_dataset = CommentDataset(train_exs, tokenizer, m_config)

        if m_config.sort_by_len:
            train_sampler = SortedBatchSampler(train_dataset.lengths(),
                                               m_config.batch_size,
                                               shuffle=True)
        else:
            train_sampler = sampler.RandomSampler(train_dataset)

        train_loader = DataLoaderX(
            train_dataset,
            batch_size=m_config.batch_size,
            sampler=train_sampler,
            num_workers=m_config.data_workers,
            collate_fn=vector.batch_handle,
            pin_memory=m_config.cuda,
            drop_last=False
        )
        m_config.train_len = len(train_dataset)

    dev_dataset = CommentDataset(dev_exs, tokenizer, m_config)

    dev_sampler = sampler.SequentialSampler(dev_dataset)

    dev_loader = DataLoaderX(
        dev_dataset,
        batch_size=m_config.test_batch_size,
        sampler=dev_sampler,
        num_workers=m_config.data_workers,
        collate_fn=vector.batch_handle,
        pin_memory=m_config.cuda,
        drop_last=False        # drop_last=m_config.parallel
    )
    m_config.dev_len = len(dev_dataset)

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(m_config.to_vanilla_(), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST
    if m_config.only_test:
        stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0,
                 'no_improvement': 0}
        validate_official(m_config, dev_loader, model,
                          tokenizer, stats, mode='test')

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    else:
        logger.info('-' * 100)
        logger.info('Starting training...')
        stats = {'timer': Timer(), 'epoch': start_epoch,
                 'best_valid': 0, 'no_improvement': 0}

        if c_model.optimizer in ['sgd', 'adam'] and c_model.warmup_epochs >= start_epoch:
            logger.info("Use warmup lrate for the %d epoch, from 0 up to %s." %
                        (c_model.warmup_epochs, c_model.learning_rate))
            num_batches = m_config.train_len // m_config.batch_size
            warmup_factor = (c_model.learning_rate + 0.) / \
                (num_batches * c_model.warmup_epochs)
            stats['warmup_factor'] = warmup_factor

        for epoch in range(start_epoch, m_config.num_epochs + 1):
            stats['epoch'] = epoch
            if c_model.optimizer in ['sgd', 'adam'] and epoch > c_model.warmup_epochs:
                model.optimizer.param_groups[0]['lr'] = model.optimizer.param_groups[0]['lr'] * c_model.lr_decay

            loss = train(m_config, tokenizer, train_loader, model, stats)

            if epoch % 10 == 0 or loss <= 1:
                result = validate_official(
                    m_config, dev_loader, model, tokenizer, stats)

                # Save best valid
                if result[m_config.valid_metric] > stats['best_valid']:
                    logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                                (m_config.valid_metric, result[m_config.valid_metric],
                                    stats['epoch'], model.updates))
                    model.save(m_config.model_file)
                    stats['best_valid'] = result[m_config.valid_metric]
                    stats['no_improvement'] = 0
                else:
                    stats['no_improvement'] += 1
                    if stats['no_improvement'] >= c_model.early_stop:
                        # break
                        pass


if __name__ == '__main__':

    # Parse cmdline args and setup environment
    train_config = get_default_config()
    set_logging(logger, train_config)

    # Run!
    main(train_config)
