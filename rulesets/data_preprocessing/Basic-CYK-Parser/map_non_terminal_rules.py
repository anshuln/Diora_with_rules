rule_file = open("grammar_wsj_cnf_top_2500.txt")
rule_file_nt = open("grammar_wsj_cnf.txt")
rules_orig = []
for r in rule_file.readlines():
     rules_orig.append(r)
 
rules_nt = []
for r in rule_file_nt.readlines():
     rules_nt.append(r)
 
rule_map = {}
for idx,r in enumerate(rules_orig):
     if r in rules_nt:
             rule_map[rules_nt.index(r)] = int(idx) 
 

import json
json.dump(rule_map,open("rules_non_terminal_mapping_wsj_2500.json","w"))
import time
print(rule_map)
time.sleep(5)
print("")