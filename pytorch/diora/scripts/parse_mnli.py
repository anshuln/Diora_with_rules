'''
python3 diora/scripts/parse_mnli.py     --batch_size 10     --data_type txt_id     --elmo_cache_dir ~/Content_alignment/diora_snli/data/elmo     --load_model_path ~/Content_alignment/fresh_diora/diora/log/9e5b36ef/model_periodic.pt --validation_path ~/Content_alignment/diora_snli/multinli_1.0_dev_matched.jsonl --model_flags ~/Content_alignment/fresh_diora/diora/Downloads/diora-checkpoints/mlp-softmax-shared/flags.json    --validation_filter_length 50

'''



import os
import collections
import json
import types

import torch

from tqdm import tqdm
from diora.data.dataset_mnli import ConsolidateDatasets, ReconstructDataset, make_batch_iterator
from train import argument_parser, parse_args, configure
#from train import get_validation_dataset, get_validation_iterator
# from train_snli import build_net_snli
from train import build_net

from diora.logging.configuration import get_logger

from diora.analysis.cky import ParsePredictor as CKY


def get_train_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.train_path,
        embeddings_path=options.embeddings_path, filter_length=options.train_filter_length,
        data_type='nli')


def get_train_iterator(options, dataset):
    return make_batch_iterator(options, dataset, shuffle=True,
            include_partial=False, filter_length=options.train_filter_length,
            batch_size=options.batch_size, length_to_size=options.length_to_size)


def get_validation_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.validation_path,
            embeddings_path=options.embeddings_path, filter_length=options.validation_filter_length,
            data_type='nli')


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


punctuation_words = set([x.lower() for x in ['.', ',', ':', '-LRB-', '-RRB-', '\'\'',
    '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']])


def remove_using_flat_mask(tr, mask):
    kept, removed = [], []
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            if mask[pos] == False:
                removed.append(tr)
                return None, 1
            kept.append(tr)
            return tr, 1

        size = 0
        node = []

        for subtree in tr:
            x, xsize = func(subtree, pos=pos + size)
            if x is not None:
                node.append(x)
            size += xsize

        if len(node) == 1:
            node = node[0]
        elif len(node) == 0:
            return None, size
        return node, size
    new_tree, _ = func(tr)
    return new_tree, kept, removed


def flatten_tree(tr):
    def func(tr):
        if not isinstance(tr, (list, tuple)):
            return [tr]
        result = []
        for x in tr:
            result += func(x)
        return result
    return func(tr)


def postprocess(tr, tokens=None):
    if tokens is None:
        tokens = flatten_tree(tr)

    # Don't remove the last token. It's not punctuation.
    if tokens[-1].lower() not in punctuation_words:
        return tr

    mask = [True] * (len(tokens) - 1) + [False]
    tr, kept, removed = remove_using_flat_mask(tr, mask)
    assert len(kept) == len(tokens) - 1, 'Incorrect tokens left. Original = {}, Output = {}, Kept = {}'.format(
        binary_tree, tr, kept)
    assert len(kept) > 0, 'No tokens left. Original = {}'.format(tokens)
    assert len(removed) == 1
    tr = (tr, tokens[-1])

    return tr


def override_init_with_batch(var):
    init_with_batch = var.init_with_batch

    def func(self, *args, **kwargs):
        init_with_batch(*args, **kwargs)
        self.saved_scalars = {i: {} for i in range(self.length)}
        self.saved_scalars_out = {i: {} for i in range(self.length)}

    var.init_with_batch = types.MethodType(func, var)


def override_inside_hook(var):
    def func(self, level, h, c, s):
        length = self.length
        B = self.batch_size
        L = length - level

        assert s.shape[0] == B
        assert s.shape[1] == L
        # assert s.shape[2] == N
        assert s.shape[3] == 1
        assert len(s.shape) == 4
        smax = s.max(2, keepdim=True)[0]
        s = s - smax

        for pos in range(L):
            self.saved_scalars[level][pos] = s[:, pos, :]

    var.inside_hook = types.MethodType(func, var)


def replace_leaves(tree, leaves):
    #0print(leav0es)
    def func(tr, pos=0):
        #print(tr,pos,len(leaves))
        if not isinstance(tr, (list, tuple)):
            return 1, leaves[pos]

        newtree = []
        sofar = 0
        for node in tr:
            size, newnode = func(node, pos+sofar)
            sofar += size
            newtree.append(str(newnode))
        newtree = ' '.join(newtree)
        #print(newtree)
        return sofar, str('( '+str(newtree)+' )')

    _, newtree = func(tree)

    return newtree


def run(options):
    logger = get_logger()

    validation_dataset = get_validation_dataset(options)
    #print(validation_dataset['sentence1'][0],validation_dataset['example_ids'][0])
    validation_iterator = get_validation_iterator(options, validation_dataset)
    word2idx = validation_dataset['word2idx']
    embeddings = validation_dataset['embeddings']

    idx2word = {v: k for k, v in word2idx.items()}

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings, validation_iterator)

    # Parse

    diora = trainer.net.encoder

    ## Monkey patch parsing specific methods.
    override_init_with_batch(diora)
    override_inside_hook(diora)

    ## Turn off outside pass.
    #trainer.net.encoder.outside = False

    ## Eval mode.
    trainer.net.eval()

    ## Parse predictor.
    parse_predictor = CKY(net=diora, word2idx=word2idx)

    batches = validation_iterator.get_iterator(random_seed=options.seed)

    output_path1 = os.path.abspath(os.path.join(options.experiment_path, 'parse_mnli1.jsonl'))
    output_path2 = os.path.abspath(os.path.join(options.experiment_path, 'parse_mnli2.jsonl'))

    logger.info('Beginning.')
    logger.info('Writing output to = {}'.format(output_path1))
    logger.info('Writing output to = {}'.format(output_path2))

    f = open(output_path1, 'w')

    with torch.no_grad():
        for i, batch_map in tqdm(enumerate(batches)):
            #print(batch_map.keys())
            sentences1 = batch_map['sentences_1']
            sentences2 = batch_map['sentences_2']
            #print(sentences.shape)
            batch_size = sentences1.shape[0]
            length = sentences1.shape[1]

            # Skip very short sentences.
            if length <= 2:
                continue

            _ = trainer.step(batch_map, train=False, compute_loss=False)

            trees1 = parse_predictor.parse_batch(sentences1)
            trees2 = parse_predictor.parse_batch(sentences2)
            #print(list(zip(trees1,trees2)))
            for ii,tree in enumerate(list(zip(trees1,trees2))):
                tr1,tr2 = tree[0],tree[1]
                example_id = batch_map['example_ids'][ii]
                #print(batch_map['example_ids'])
                s1 = [idx2word[idx] for idx in sentences1[ii].tolist()]
                s2 = [idx2word[idx] for idx in sentences2[ii].tolist()]
                tr1 = replace_leaves(tr1, s1)
                tr2 = replace_leaves(tr2, s2)
                if options.postprocess:
                    tr = postprocess(tr, s1)
                o = collections.OrderedDict(example_id=example_id, sentence1=tr1,sentence2=tr2)
                #print(o)
                #exit()

                f.write(json.dumps(o) + '\n')
  
    f.close()
'''
    f = open(output_path2, 'w')
    with torch.no_grad():
        for i, batch_map in tqdm(enumerate(batches)):
            sentences = batch_map['sentences_2']
            batch_size = sentences.shape[0]
            length = sentences.shape[1]

            # Skip very short sentences.
            if length <= 2:
                continue

            _ = trainer.step(batch_map, train=False, compute_loss=False)

            trees = parse_predictor.parse_batch(sentences)

            for ii, tr in enumerate(trees):
                example_id = batch_map['example_ids'][ii]
                s = [idx2word[idx] for idx in sentences[ii].tolist()]
                tr = replace_leaves(tr, s)
                if options.postprocess:
                    tr = postprocess(tr, s)
                o = collections.OrderedDict(example_id=example_id, tree=tr)

                f.write(json.dumps(o) + '\n')

    f.close()
'''    


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
