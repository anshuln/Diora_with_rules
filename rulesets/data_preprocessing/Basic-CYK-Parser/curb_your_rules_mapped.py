#usage: pyhton3 example.py
# 427949, 237742, 46310
# 52904,  245447, 488522
# from CYK_Paser import Grammar
import json 
from tqdm import tqdm
import numpy as np
SNLI_FILE = '/home/ritesh/Content_alignment/Gumble/tree_data/nli_train_wsjrules.jsonl'
OUT_FILE  ='/home/ritesh/Content_alignment/Gumble/tree_data/nli_train_wsjrules_topk.jsonl'
INDEX_FILE_MAP = 'rules_non_terminal_mapping_wsj_2500.json'
# NUM_RULES = 1000
data = open(SNLI_FILE,"r")
out  = open(OUT_FILE,"w")
permitted_indices_map = json.load(open(INDEX_FILE_MAP,'r'))

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


for line in tqdm(data.readlines()):
    try:
        dict = json.loads(line)
        r1 = dict['sentence1_rule']
        new_rules = [[x[0],x[1],[[permitted_indices_map[z] for z in y if z in permitted_indices_map] for y in x[2]]] for x in r1]
        dict['sentence1_rule'] = r1
        tok = get_tokens(dict['sentence1_binary_parse'])
        
        if len(tok) != np.array([len(x[2]) for x in dict['sentence1_rule']])[-1]+1:
            continue
        # print(r1)
        # print("---------")
        # print(new_rules)
        r2 = dict['sentence2_rule']
        new_rules = [[x[0],x[1],[[permitted_indices_map[z] for z in y if z in permitted_indices_map] for y in x[2]]] for x in r2]
        dict['sentence2_rule'] = r2
        tok = get_tokens(dict['sentence2_binary_parse'])
        if len(tok) != np.array([len(x[2]) for x in dict['sentence2_rule']])[-1]+1:
            continue

        out.write(json.dumps(dict))
        out.write("\n")
    except Exception as e:
        print(e)
    # break
