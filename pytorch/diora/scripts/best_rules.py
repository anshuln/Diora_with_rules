''' 
Script to return the highest scoring rules for the model


python3 diora/scripts/best_rules.py     --batch_size 100    --data_type nli     --elmo_cache_dir ~/Content_alignment/diora_snli/data/elmo     --load_model_path ~/Content_alignment/fresh_diora/diora/log/451d0ffc/model_periodic.pt --validation_path ~/Content_alignment/diora_snli/data/snli_1.0/snli_1.0_dev.jsonl --train_path ~/Content_alignment/diora_snli/data/snli_1.0/snli_1.0_dev.jsonl --model_flags ~/Content_alignment/fresh_diora/diora/log/451d0ffc/flags.json --validation_filter_length 40 --data_type nli --rule_based 

'''
import argparse
import datetime
import gc
import math
import os
import random
import sys
import uuid

import torch
import torch.nn as nn
import numpy as np

from diora.data.dataset import ConsolidateDatasets, ReconstructDataset, make_batch_iterator

from diora.utils.path import package_path
from diora.logging.configuration import configure_experiment, get_logger
from diora.utils.flags import stringify_flags, init_with_flags_file, save_flags
from diora.utils.checkpoint import save_experiment

from diora.net.experiment_logger import ExperimentLogger

data_types_choices = ('nli', 'conll_jsonl', 'txt', 'txt_id', 'synthetic', 'jsonl')

def build_net(options, embeddings, batch_iterator=None):
    from diora.net.trainer import build_net

    trainer = build_net(options, embeddings, batch_iterator, random_seed=options.seed)

    # logger = get_logger()
    # logger.info('# of params = {}'.format(count_params(trainer.net)))

    return trainer
def get_train_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.train_path,
        embeddings_path=options.embeddings_path, filter_length=options.train_filter_length,
        data_type=options.train_data_type)


def get_train_iterator(options, dataset):
    print("getting new iter")
    return make_batch_iterator(options, dataset, shuffle=True,
            include_partial=False, filter_length=options.train_filter_length,
            batch_size=options.batch_size, length_to_size=options.length_to_size)


def get_validation_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.validation_path,
            embeddings_path=options.embeddings_path, filter_length=options.validation_filter_length,
            data_type=options.validation_data_type)


def get_validation_iterator(options, dataset):
    return make_batch_iterator(options, dataset, shuffle=False,
            include_partial=True, filter_length=options.validation_filter_length,
            batch_size=options.validation_batch_size, length_to_size=options.length_to_size)


def get_train_and_validation(options):
    train_dataset = get_train_dataset(options)
    validation_dataset = get_validation_dataset(options)

    # Modifies datasets. Unifying word mappings, embeddings, etc.
    ConsolidateDatasets([train_dataset, validation_dataset]).run()

    return train_dataset, validation_dataset


def run(options):

    train_dataset, validation_dataset = get_train_and_validation(options)
    train_iterator = get_train_iterator(options, train_dataset)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    embeddings = train_dataset['embeddings']

    trainer = build_net(options, embeddings, validation_iterator)

    rules = trainer.get_top_rules(options.load_model_path)
    np.savetxt("top_rules_{}.txt".format(options.load_model_path.split("/")[-2]),rules)


def argument_parser():
    parser = argparse.ArgumentParser()

    # Debug.
    parser.add_argument('--seed', default=15, type=int)
    parser.add_argument('--git_sha', default=None, type=str)
    parser.add_argument('--git_branch_name', default=None, type=str)
    parser.add_argument('--git_dirty', default=None, type=str)
    parser.add_argument('--uuid', default=None, type=str)
    parser.add_argument('--model_flags', default=None, type=str,
                        help='Load model settings from a flags file.')
    parser.add_argument('--flags', default=None, type=str,
                        help='Load any settings from a flags file.')

    parser.add_argument('--master_addr', default='127.0.0.1', type=str)
    parser.add_argument('--master_port', default='29500', type=str)
    parser.add_argument('--world_size', default=None, type=int)

    # Pytorch
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--multigpu', action='store_true')
    parser.add_argument("--local_rank", default=None, type=int) # for distributed-data-parallel

    # Logging.
    parser.add_argument('--default_experiment_directory', default=os.path.join(package_path(), '..', 'log'), type=str)
    parser.add_argument('--experiment_name', default=None, type=str)
    parser.add_argument('--experiment_path', default=None, type=str)
    parser.add_argument('--log_every_batch', default=10, type=int)
    parser.add_argument('--save_latest', default=1000, type=int)
    parser.add_argument('--save_distinct', default=50000, type=int)
    parser.add_argument('--save_after', default=1000, type=int)
    parser.add_argument('--save_init', action='store_true')

    # Loading.
    parser.add_argument('--load_model_path', default=None, type=str)

    # Data.
    parser.add_argument('--data_type', default='nli', choices=data_types_choices)
    parser.add_argument('--train_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--validation_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--train_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_train.jsonl'), type=str)
    parser.add_argument('--validation_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_dev.jsonl'), type=str)
    parser.add_argument('--embeddings_path', default=os.path.expanduser('~/data/glove/glove.6B.300d.txt'), type=str)

    # Data (synthetic).
    parser.add_argument('--synthetic-nexamples', default=1000, type=int)
    parser.add_argument('--synthetic-vocabsize', default=1000, type=int)
    parser.add_argument('--synthetic-embeddingsize', default=1024, type=int)
    parser.add_argument('--synthetic-minlen', default=20, type=int)
    parser.add_argument('--synthetic-maxlen', default=21, type=int)
    parser.add_argument('--synthetic-seed', default=11, type=int)
    parser.add_argument('--synthetic-length', default=None, type=int)
    parser.add_argument('--use-synthetic-embeddings', action='store_true')

    # Data (preprocessing).
    parser.add_argument('--uppercase', action='store_true')
    parser.add_argument('--train_filter_length', default=50, type=int)
    parser.add_argument('--validation_filter_length', default=0, type=int)

    # Model.
    parser.add_argument('--arch', default='treelstm', choices=('treelstm', 'mlp', 'mlp-shared'))
    parser.add_argument('--hidden_dim', default=10, type=int)
    parser.add_argument('--normalize', default='unit', choices=('none', 'unit'))
    parser.add_argument('--compress', action='store_true',
                        help='If true, then copy root from inside chart for outside. ' + \
                             'Otherwise, learn outside root as bias.')

    # Model (Objective).
    parser.add_argument('--reconstruct_mode', default='margin', choices=('margin', 'softmax'))
    parser.add_argument('--rule_based', action='store_true',help='Whether to use rule based loss or not')

    # Model (Embeddings).
    parser.add_argument('--emb', default='w2v', choices=('w2v', 'elmo', 'both'))

    # Model (Negative Sampler).
    parser.add_argument('--margin', default=1, type=float)
    parser.add_argument('--k_neg', default=3, type=int)
    parser.add_argument('--freq_dist_power', default=0.75, type=float)

    # ELMo
    parser.add_argument('--elmo_options_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', type=str)
    parser.add_argument('--elmo_weights_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', type=str)
    parser.add_argument('--elmo_cache_dir', default=None, type=str,
                        help='If set, then context-insensitive word embeddings will be cached ' + \
                             '(identified by a hash of the vocabulary).')

    # Training.
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--length_to_size', default=None, type=str,
                        help='Easily specify a mapping of length to batch_size.' + \
                             'For instance, 10:32,20:16 means that all batches' + \
                             'of length 10-19 will have batch size 32, 20 or greater' + \
                             'will have batch size 16, and less than 10 will have batch size' + \
                             'equal to the batch_size arg. Only applies to training.')
    parser.add_argument('--train_dataset_size', default=None, type=int)
    parser.add_argument('--validation_dataset_size', default=None, type=int)
    parser.add_argument('--validation_batch_size', default=None, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--max_step', default=None, type=int)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_after', default=0, type=int)

    # Parsing.
    parser.add_argument('--postprocess', action='store_true')

    # Optimization.
    parser.add_argument('--lr', default=4e-3, type=float)

    # Curriculum Learning
    parser.add_argument('--curriculum', action='store_true')

    return parser


def parse_args(parser):
    options, other_args = parser.parse_known_args()

    # Set default flag values (data).
    options.train_data_type = options.data_type if options.train_data_type is None else options.train_data_type
    options.validation_data_type = options.data_type if options.validation_data_type is None else options.validation_data_type
    options.validation_batch_size = options.batch_size if options.validation_batch_size is None else options.validation_batch_size

    # Set default flag values (config).
    if not options.git_branch_name:
        options.git_branch_name = os.popen(
            'git rev-parse --abbrev-ref HEAD').read().strip()

    if not options.git_sha:
        options.git_sha = os.popen('git rev-parse HEAD').read().strip()

    if not options.git_dirty:
        options.git_dirty = os.popen("git diff --quiet && echo 'clean' || echo 'dirty'").read().strip()

    if not options.uuid:
        options.uuid = str(uuid.uuid4())

    if not options.experiment_name:
        options.experiment_name = '{}'.format(options.uuid[:8])

    if not options.experiment_path:
        options.experiment_path = os.path.join(options.default_experiment_directory, options.experiment_name)

    if options.length_to_size is not None:
        parts = [x.split(':') for x in options.length_to_size.split(',')]
        options.length_to_size = {int(x[0]): int(x[1]) for x in parts}

    options.lowercase = not options.uppercase

    for k, v in options.__dict__.items():
        if type(v) == str and v.startswith('~'):
            options.__dict__[k] = os.path.expanduser(v)

    # Load model settings from a flags file.
    if options.model_flags is not None:
        flags_to_use = []
        flags_to_use += ['arch']
        flags_to_use += ['compress']
        flags_to_use += ['emb']
        flags_to_use += ['hidden_dim']
        flags_to_use += ['normalize']
        flags_to_use += ['reconstruct_mode']
        flags_to_use += ['curriculum']
        flags_to_use += ['rule_based']
        options = init_with_flags_file(options, options.model_flags, flags_to_use)

    # Load any setting from a flags file.
    if options.flags is not None:
        options = init_with_flags_file(options, options.flags)

    return options


def configure(options):
    # Configure output paths for this experiment.
    configure_experiment(options.experiment_path, rank=options.local_rank)

    # Get logger.
    logger = get_logger()

    # Print flags.
    logger.info(stringify_flags(options))
    save_flags(options, options.experiment_path)


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
