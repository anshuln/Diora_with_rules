#usage: pyhton3 example.py
# 427949, 237742, 46310
# 52904,  245447, 488522
from CYK_Paser import Grammar
import json 
from tqdm import tqdm
import numpy as np 
import argparse

def get_tokens(parse):
    transitions = []
    tokens = []
    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                tokens.append(word)
    return tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ Arguments: arg """
    parser.add_argument('--part',type=int,default=0)
    
    args = parser.parse_args()
    

    g = Grammar('grammar_wsj_cnf_no_hyp.txt')
    SNLI_FILE = '/home/ritesh/Content_alignment/Gumble/tree_data/nli_train.jsonl'
    # SNLI_FILE = '/home/ritesh/Content_alignment/Gumble/tree_data/nli_train_wsjrules_nlivocab.jsonl'
    OUT_FILE  ='/home/ritesh/Content_alignment/Gumble/tree_data/nli_train_wsjrules_nohyp_part_{}.jsonl'.format(args.part)
    # OUT_FILE = '/home/ritesh/Content_alignment/Gumble/tree_data/nli_train_wsjrules_nlivocab_out.jsonl'
    data = open(SNLI_FILE,"r")
    out  = open(OUT_FILE,"w")
    failed = 0

    idx = 0
    lens = []
    for line in tqdm(data.readlines()):
        # print(idx)
        if idx < (args.part)*50000 or idx > (args.part+1)*50000:
            idx += 1
            continue
        try:
            dict = json.loads(line)
            tok = get_tokens(dict['sentence1_binary_parse'])
            if len(tok) == 0:
                continue
            # lens.append(len(tok))

            if len(tok) < 20:
              a = 1
              g.parse(" ".join(tok))
              dict['sentence1_rule'] = g.parse_table_diora()
            else:
              failed += 1
              continue

            if len(tok) != np.array([len(x[2]) for x in dict['sentence1_rule']])[-1]+1:
                failed += 1
                continue

            tok = get_tokens(dict['sentence2_binary_parse'])
            if len(tok) == 0:
                continue


            lens.append(len(tok))
            if len(tok) < 20:
              a = 1
              g.parse(" ".join(tok))
              dict['sentence2_rule'] = g.parse_table_diora()
            else:
              failed += 1
              continue
            # out.write(json.dumps(dict))
            # out.write("\n")
    
            if len(tok) != np.array([len(x[2]) for x in dict['sentence2_rule']])[-1]+1:
                failed += 1
                continue

            out.write(json.dumps(dict))
            out.write("\n")
        except Exception as e:
            print(e)
            failed += 1
            print("Failed {}".format(failed))
        idx += 1

    print("Failed with {}".format(failed))
    # print(np.mean(np.array(lens)))

    out.close()
    # print('')
    # print('')
    # print('')

    # g = Grammar('example_grammar2.txt')
    # g.parse('she eats a fish with a fork')
    # g.print_parse_table()

    # print('')
    # print('')
    # print('')

    # g = Grammar('example_grammar2.txt')
    # g.parse('eats she fork a fish')
    # g.print_parse_table()
    # g.get_trees()


