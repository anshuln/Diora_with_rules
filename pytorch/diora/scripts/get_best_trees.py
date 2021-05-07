import json

import numpy as np

from eval_brackets import *

def best_trees(gt_path,test_path, print_top=False):
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
	file = open(test_path,"r")
	for f in file.readlines():
		data = json.loads(f)
		model_out = data["tree"]
		reports.append({data["example_id"]:unpad(to_string(model_out))})

	max_scores = {}
	c = []
	for i,report in enumerate(reports):
		depths.append(corpus_average_depth(report))
		f1 = corpus_stats(report,gt_trees)
		c.append(f1)
	indices = np.argsort(c)[::-1]
	idscores = [(list(reports[i].keys())[0],(c[i],list(reports[i].values()))) for i in indices]
	# print(np.mean(c))
	return idscores, gt_trees




def rel_scores(idscore_1,idscore_2, gt):
	d1 = dict(idscore_1)
	d2 = dict(idscore_2)
	all_id_score = [(k,(d1[k][0]**2)/(d2[k][0]+1e-10)) for k in d1]
	all_id_score.sort(reverse=True,key=lambda x:x[1])
	keys = set()
	for x in all_id_score[:50]:
		print(x[0])
		if x[0].split("_")[0][:-1] in keys:
			continue
		keys.add(x[0].split("_")[0][:-1])
		print("Id: {}, Our Score : {}, Our Tree : {}, Diora Score : {}, Diora Tree: {}, Ground Truth {}".format(x[0],d1[x[0]][0],d1[x[0]][1],d2[x[0]][0],d2[x[0]][1], gt[x[0]].lower()))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

	# Model parameters.
	parser.add_argument('--test_file_1', type=str, default='/home/ritesh/Content_alignment/fresh_diora/diora/log/5b2a1da8/parse.jsonl',
						help='location of the data corpus')
	parser.add_argument('--test_file_2', type=str, default='/home/ritesh/Content_alignment/fresh_diora/diora/log/5b2a1da8/parse.jsonl')
	parser.add_argument('--gt_file', type=str, default='/home/ritesh/Content_alignment/diora_snli/data/snli_1.0/snli_1.0_dev.jsonl',
						help='location of the brackets json file')

	args = parser.parse_args()

	id1,gt = best_trees(args.gt_file,args.test_file_1)
	print("one done")
	id2,_ = best_trees(args.gt_file,args.test_file_2)
	print("both done")
	rel_scores(id1,id2,gt)