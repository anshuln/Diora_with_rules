from nltk import Tree
from tqdm import tqdm
import json 

import data_ptb

CORPUS_FILE = '/home/ritesh/Content_alignment/diora_snli/data/snli_1.0/snli_1.0_train.jsonl'
RULE_FILE = '/home/ritesh/Content_alignment/diora_snli/Basic-CYK-Parser/grammar_wsj_no_cnf.txt'

if __name__ == '__main__':
    rules = dict()
    # data = open(CORPUS_FILE,"r")
    corpus = data_ptb.Corpus("/home/ritesh/Content_alignment/Tree-Transformer/data/")
    # corpus.dictionary = dictionary
    dataset = zip(corpus.test_sens, corpus.test_trees, corpus.test_nltktrees)
    output = open("wsj_bracket.jsonl","w")
    idx = 0
    for sen, sen_tree, sen_nltktree in tqdm(dataset):
      tree = sen_nltktree
      for sub in tree.subtrees():
          for n, child in enumerate(sub):
              if isinstance(child, str):
                  continue
                

              if len(list(child.subtrees(filter=lambda x:x.label()=='-NONE-')))==len(child.leaves()):
                  del sub[n]

              child.set_label(child.label().split("-")[0])

      # tree.chomsky_normal_form()
      for p in tree.productions():
          if str(p) in rules.keys():
              rules[str(p)] += 1
          else:
              rules[str(p)] = 1

    # for line in tqdm(data.readlines()):
    #   dict = json.loads(line)
    #   S = dict['sentence1_parse']
    #   tree = Tree.fromstring(S)
    #   # tree.pretty_print()
    #   tree.chomsky_normal_form()
    #   for p in tree.productions():
    #       if str(p) in rules.keys():
    #           rules[str(p)] += 1
    #       else:
    #           rules[str(p)] = 1

    #   S = dict['sentence2_parse']
    #   tree = Tree.fromstring(S)
    #   tree.chomsky_normal_form()
    #   for p in tree.productions():
    #       if str(p) in rules.keys():
    #           rules[str(p)] += 1
    #       else:
    #           rules[str(p)] = 1

    rules = {k: v for k, v in sorted(rules.items(), key=lambda item: item[1], reverse=True)}
    out = open(RULE_FILE,"w")
    for r in rules:
        out.write("{} \n".format(str(r)))
    print(len(rules))