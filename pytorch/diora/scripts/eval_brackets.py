'''Evaluates parsing performance

'''

import json

import argparse
import re

# import matplotlib.pyplot as plt
import nltk
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import hashlib
from collections import Counter


punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-',"''", '`']
PUNCT = set(punctuation_words + [x.lower() for x in punctuation_words]) 

def get_brackets(tree, idx=0):
    brackets = set()
    brac_dict = []
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, node_brac_dict, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                for i in node_brac_dict.items():
                    brac_dict.append(i)
                if isinstance(node, nltk.Tree):
                    brac_dict.append(((idx, next_idx),node.label()))
                else:
                    brac_dict.append(((idx, next_idx),"leaf"))
                brackets.update(node_brac)

            idx = next_idx
        return brackets, dict(brac_dict), idx
    else:
        return brackets, dict(brac_dict), idx + 1



def remove_labels(tree):
    if len(tree) == 1:
        tree.set_label('')
        return [tree[0]]
    children = []
    for i in tree:
        children.append(remove_labels(i))
    tree.set_label('')
    return tree

def tokenize_parse(parse):
    return [token for token in parse.split() if token not in ['(', ')']]


def to_string(parse):
    if type(parse) is not list:
        return parse
    if len(parse) == 1:
        return parse[0]
    else:
        return '( ' + to_string(parse[0]) + ' ' + to_string(parse[1]) + ' )'


def tokens_to_rb(tree):
    if type(tree) is not list:
        return tree
    if len(tree) == 1:
        return tree[0]
    else:
        return [tree[0], tokens_to_rb(tree[1:])]


def to_rb(gt_table):
    new_data = {}
    for key in gt_table:
        parse = gt_table[key]
        tokens = tokenize_parse(parse)
        new_data[key] = to_string(tokens_to_rb(tokens))
    return new_data


def tokens_to_lb(tree):
    if type(tree) is not list:
        return tree
    if len(tree) == 1:
        return tree[0]
    else:
        return [tokens_to_lb(tree[:-1]), tree[-1]]


def to_lb(gt_table):
    new_data = {}
    for key in gt_table:
        parse = gt_table[key]
        tokens = tokenize_parse(parse)
        new_data[key] = to_string(tokens_to_lb(tokens))
    return new_data


def average_depth(parse):
    depths = []
    current_depth = 0
    for token in parse.split():
        if token == '(':
            current_depth += 1
        elif token == ')':
            current_depth -= 1
        else:
            depths.append(current_depth)
    return float(sum(depths)) / len(depths)


def corpus_average_depth(corpus):
    local_averages = []
    for key in corpus:
        local_averages.append(average_depth(corpus[key]))
    return float(sum(local_averages)) / len(local_averages)


def average_length(parse):
    return len(parse.split())


def corpus_average_length(corpus):
    local_averages = []
    for key in corpus:
        local_averages.append(average_length(corpus[key]))
    return float(sum(local_averages)) / len(local_averages)


def corpus_stats(corpus_1, corpus_2, first_two=False, neg_pair=False):
    """ 
    Note: If a few examples in one dataset are missing from the other (i.e., some examples from the source corpus were not included 
      in a model corpus), the shorter dataset must be supplied as corpus_1.
    """

    f1_accum = 0.0
    count = 0.0
    first_two_count = 0.0
    last_two_count = 0.0
    three_count = 0.0
    neg_pair_count = 0.0
    neg_count = 0.0
    for key in corpus_1.keys():    
        c1,w1 = to_indexed_contituents(corpus_1[key])
        # corp = ' '.join(str(remove_labels(nltk.Tree.fromstring(corpus_2[key]))).split()) 
        c2,w2 = to_indexed_contituents(corpus_2[key])
        if w1!=w2:
            # print(key,w1,w2)
            # print(corpus_1[key], )
            continue
        # if len(c1) < 15:
        #   print(key) 
        #   print(corpus_1[key])
        #   print(c1)
        #   print(corpus_2[key])
        #   print(c2)
        # print(corpus_1[key],corp,c1,c2)
        f1_accum += example_f1(c1, c2)
        count += 1

        if first_two and len(c1) > 1:
            if (0, 2) in c1:
                first_two_count += 1
            num_words = len(c1) + 1
            if (num_words - 2, num_words) in c1:
                last_two_count += 1
            three_count += 1 
        if neg_pair:
            word_index = 0
            tokens = corpus_1[key].split()
            for token_index, token in enumerate(tokens):
                if token in ['(', ')']:
                    continue
                if token in ["n't", "not", "never", "no", "none", "Not", "Never", "No", "None"]:
                    if tokens[token_index + 1] not in ['(', ')']:
                        neg_pair_count += 1
                    neg_count += 1
                word_index += 1
    stats = f1_accum / (count + 1e-10)
    if first_two:
        stats = str(stats) + '\t' + str(first_two_count / three_count) + '\t' + str(last_two_count / three_count)
    if neg_pair:
        stats = str(stats) + '\t' + str(neg_pair_count / neg_count)
    return stats


def to_indexed_contituents(parse):
    sp = parse.split()
    if len(sp) == 1:
        return set([(0, 1)])

    backpointers = []
    indexed_constituents = set()
    word_index = 0
    for index, token in enumerate(sp):
        if token == '(':
            backpointers.append(word_index)
        elif token == ')':
            start = backpointers.pop()
            end = word_index
            constituent = (start, end)
            indexed_constituents.add(constituent)
        else:
            word_index += 1
    return indexed_constituents, word_index

def to_indexed_contituents_no_punct(parse):
    sp = parse.split()
    if len(sp) == 1:
        return set([(0, 1)])

    backpointers = []
    indexed_constituents = set()
    word_index = 0
    quote_countr = 0
    for index, token in enumerate(sp):
        if token == "`":
            # print("Yo")
            quote_countr = 1

        if token == '(':
            backpointers.append(word_index)
        elif token == ')':
            start = backpointers.pop()
            end = word_index
            constituent = (start, end)
            indexed_constituents.add(constituent)
        elif token in PUNCT:
            continue
        elif quote_countr == 1 and token == "'":
            quote_countr = 0
            continue
        # elif :
        else:
            word_index += 1
    return indexed_constituents, word_index

def corpus_stats_labeled(corpus_unlabeled, corpus_labeled):
    """
    Note: If a few examples in one dataset are missing from the other (i.e., some examples from the source corpus were not included
      in a model corpus), the shorter dataset must be supplied as corpus_1.
    """

    correct = Counter()
    total = Counter()

    for key in corpus_unlabeled:
        c1,wi = to_indexed_contituents_no_punct(corpus_unlabeled[key])
        c2,wil = to_indexed_contituents_labeled(corpus_labeled[key])
        if wi != wil:
            # print(key,wi,wil)
            # print(corpus_unlabeled[key])
            # print(corpus_labeled[key])
            continue
        # if len(c2) == 0:
        #     continue

        ex_correct, ex_total = example_labeled_acc(c1, c2)
        correct.update(ex_correct)
        total.update(ex_total)
    return correct, total

def example_labeled_acc(c1, c2):
    '''Compute the number of non-unary constituents of each type in the labeled (non-binirized) parse appear in the model output.'''
    correct = Counter()
    total = Counter()
    for constituent in c2:
        if (constituent[0], constituent[1]) in c1:
            correct[constituent[2]] += 1
        total[constituent[2]] += 1
    return correct, total

def to_indexed_contituents_labeled(parse):
    # sp = re.findall(r'\([^ ]+| [^\(\) ]+|\)', parse)
    sp = parse.split()
    if len(sp) == 1:
        return set([(0, 1)])

    backpointers = []
    indexed_constituents = set()
    word_index = 0
    for index, token in enumerate(sp):
        if token[0] == '(':
            backpointers.append((word_index, token[1:]))
        elif token == ')':
            start, typ = backpointers.pop()
            end = word_index
            constituent = (start, end, typ)
            if end - start > 1:
                indexed_constituents.add(constituent)
        else:
            word_index += 1
    # print(parse,indexed_constituents)
    return indexed_constituents, word_index


def unpad(parse):
    tokens = parse.split()
    to_drop = 0
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == "_PAD":
            to_drop += 1
        elif tokens[i] == ")":
            continue
        else:
            break
    if to_drop == 0:
        return parse
    else:
        return " ".join(tokens[to_drop:-2 * to_drop])



def eval(gt_path,test_path, print_top=False):
    # print("VG")
    gt_trees = {}
    file = open(gt_path,"r")
    for f in file.readlines():
        data = json.loads(f)
        gt_trees["{}_1".format(data["pairID"])] = data['sentence1_binary_parse']
        gt_trees["{}_2".format(data["pairID"])] = data['sentence2_binary_parse']

    lb = to_lb(gt_trees)
    rb = to_rb(gt_trees)


    file.close()

    reports = []
    depths = []
    l = []
    r = []
    c = []
    file = open(test_path,"r")
    for f in file.readlines():
        data = json.loads(f)
        # std_out = gt_trees[data["example_id"]]
        model_out = data["tree"]
        # print(' '.join(str(nltk.Tree("",data["tree"])).slit
        # print(data["example_id"],unpad(to_string(model_out)))
        # if len(model_out) 
        # if len(tree_to_spans(model_out)) < 10:
        #   print(model_out,tree_to_spans(model_out)) #unpad(to_string(model_out)))
        reports.append({data["example_id"]:unpad(to_string(model_out))})

    max_scores = {}
    for i,report in enumerate(reports):
        # print(str(corpus_stats(report, lb)) + '\t' + str(corpus_stats(report, rb)) + '\t' + str(corpus_stats(report, gt_trees, first_two=False, neg_pair=False)) + '\t' + str(corpus_average_depth(report)))
        depths.append(corpus_average_depth(report))
        # l.append(corpus_stats(report, lb))
        # r.append(corpus_stats(report,rb))
        f1 = corpus_stats(report,gt_trees)
        c.append(f1)
    print(np.mean(depths))
    # print(np.mean(l))
    # print(np.mean(r))
    # if print_top:
    #   indices = np.argsort(c)[::-1]
    #   for i in indices[:30]:
    #       print(list(reports[i].keys())[0], c[i])
    #       print(list(reports[i].values()))
    # print(np.mean(c))


def example_f1(c1, c2):
    prec = float(len(c1.intersection(c2))) / (len(c2) + 1e-10)  # TODO: More efficient.
    rec  = float(len(c1.intersection(c2))) / (len(c1) + 1e-10)
    if (len(c1.intersection(c2))) == 0:
        return 0

    return 2*prec*rec/(prec+rec)  # For strictly binary trees, P = R = F1



def compute_seg_rec(gt_path,test_path):

    gt_trees = {}
    file = open(gt_path,"r")
    for f in file.readlines():
        data = json.loads(f)
        gt_trees["{}_1".format(data["pairID"])] = data['sentence1_parse']
        gt_trees["{}_2".format(data["pairID"])] = data['sentence2_parse']

    lb = to_lb(gt_trees)
    rb = to_rb(gt_trees)


    file.close()

    reports = []
    depths = []
    l = []
    r = []
    c = []
    file = open(test_path,"r")
    for f in file.readlines():
        data = json.loads(f)
        # std_out = gt_trees[data["example_id"]]
        model_out = data["tree"]
        # print(' '.join(str(nltk.Tree("",data["tree"])).slit
        # print(data["example_id"],unpad(to_string(model_out)))
        # if len(model_out) 
        # if len(tree_to_spans(model_out)) < 10:
        #   print(model_out,tree_to_spans(model_out)) #unpad(to_string(model_out)))
        reports.append({data["example_id"]:unpad(to_string(model_out))})

    max_scores = {}
    correct, total = Counter(), Counter()
    for i,report in enumerate(reports):
        # print(str(corpus_stats(report, lb)) + '\t' + str(corpus_stats(report, rb)) + '\t' + str(corpus_stats(report, gt_trees, first_two=False, neg_pair=False)) + '\t' + str(corpus_average_depth(report)))
        # depths.append(corpus_average_depth(report))
        # l.append(corpus_stats(report, lb))
        # r.append(corpus_stats(report,rb))
        # f1 = corpus_stats(report,gt_trees)


        correct_, total_ = corpus_stats_labeled(report,gt_trees)
        correct += correct_
        total += total_
        # print(i,correct)
        # print(i,total)
        # print('ADJP:', correct['ADJP'], total['ADJP'])
        # print('NP:', correct['NP'], total['NP'])
        # print('PP:', correct['PP'], total['PP'])
        # print('INTJ:', correct['INTJ'], total['INTJ'])
    # print(i,correct,total)
    for k in correct:
        print(k,total[k]/2,correct[k]/total[k])

def compute_f_score(gt_path,test_path):
    # In an ideal world we could have just iterated over either and got the corresponding stuff. However, we need to load the entire test/dev trees since the order is messed up 
    gt_trees = {}
    gt_dict  = {}
    file = open(gt_path,"r")
    for f in file.readlines():
        data = json.loads(f)
        b1,b1_dict,_ = get_brackets(nltk.Tree.fromstring(data['sentence1_parse']))
        b2,b2_dict,_ = get_brackets(nltk.Tree.fromstring(data['sentence2_parse']))

        print(b1_dict.values(),b2_dict.values())
        gt_trees["{}_1".format(data["pairID"])] = b1 
        gt_trees["{}_2".format(data["pairID"])] = b2

        gt_dict["{}_1".format(data["pairID"])] = b1_dict
        gt_dict["{}_2".format(data["pairID"])] = b2_dict
        # print(b1_dict)

    file.close()

    file = open(test_path,"r")

    prec_list = []

    reca_list = []

    f1_list = []
    label_corr = {}
    label_all  = {}
    label_wrong = {}
    for f in file.readlines():
        data = json.loads(f)
        std_out = gt_trees[data["example_id"]]
        std_dict = gt_dict[data["example_id"]]

        model_out = get_brackets(data["tree"])[0]
        overlap = model_out.intersection(std_out)

        for m in std_out:
            if std_dict[m] not in label_all:
                label_all[std_dict[m]] = 1
            else:
                label_all[std_dict[m]] += 1
            if m in overlap:
                if std_dict[m] not in label_corr:
                    label_corr[std_dict[m]] = 1
                else:
                    label_corr[std_dict[m]] += 1
            else:
                if std_dict[m] not in label_wrong:
                    label_wrong[std_dict[m]] = 1
                else:
                    label_wrong[std_dict[m]] += 1


        prec = float(len(overlap)) / (len(model_out) + 1e-8)
        reca = float(len(overlap)) / (len(std_out) + 1e-8)
        if len(std_out) == 0:
            reca = 1.
            if len(model_out) == 0:
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        prec_list.append(prec)
        reca_list.append(reca)
        f1_list.append(f1)





    # step = 0
    # dataset = zip(corpus.test_sens, corpus.test_trees, corpus.test_nltktrees)
    # prec_list = []

    # reca_list = []

    # f1_list = []

    # print(len(brackets))

    # # print(len(corpus))
    # for sen, sen_tree, sen_nltktree in dataset:
    #   # print("HERE")
    #   model_out = set([tuple(x) for x in brackets[step]])
    #   std_out, _ = get_brackets(sen_tree)
    #   overlap = model_out.intersection(std_out)


    #   prec = float(len(overlap)) / (len(model_out) + 1e-8)
    #   reca = float(len(overlap)) / (len(std_out) + 1e-8)
    #   if len(std_out) == 0:
    #       reca = 1.
    #       if len(model_out) == 0:
    #           prec = 1.
    #   f1 = 2 * prec * reca / (prec + reca + 1e-8)
    #   prec_list.append(prec)
    #   reca_list.append(reca)
    #   f1_list.append(f1)
    #   print(std_out,model_out,f1)
    #   step += 1
    # print(f1_list)
    for k in label_corr.keys():
        print(k,label_corr[k],label_all[k])
        # print(label_wrong)
        # print(label_all)  
    # print(np.mean(np.array(f1_list)))
if __name__ == '__main__':
    numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)

    parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

    # Model parameters.
    parser.add_argument('--test_file', type=str, default='',
                        help='location of the data corpus')
    parser.add_argument('--gt_file', type=str, default='',
                        help='location of the brackets json file')

    args = parser.parse_args()

    eval(args.gt_file,args.test_file, True)

    # if 'wsj' in args.gt_file.split('/')[-1]: 
    compute_seg_rec(args.gt_file,args.test_file)
