from diora.data.dataloader import FixedLengthBatchSampler, SimpleDataset
from diora.blocks.negative_sampler import choose_negative_samples

from allennlp.modules.elmo import Elmo, batch_to_ids

import torch
import numpy as np

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

device = "cuda:0"

def get_config(config, **kwargs):
	for k, v in kwargs.items():
		if k in config:
			config[k] = v
	return config


def get_default_config():

	default_config = dict(
		batch_size=16,
		forever=False,
		drop_last=False,
		sort_by_length=True,
		shuffle=True,
		random_seed=None,
		filter_length=None,
		workers=0,
		pin_memory=False,
		include_partial=False,
		cuda=False,
		ngpus=1,
		k_neg=3,
		negative_sampler=None,
		options_path=None,
		weights_path=None,
		vocab=None,
		length_to_size=None,
		rank=None,
	)

	return default_config




def pad_batch_rules(rules):
	'''
	Returns rule lens for a batch
	
	TODO optimize!
	Remember that rules for a single element is a list of list of list, 
	[l1,l2,...,l_(n**2)], where l1 -> [[r11,r12,...,r1m],...,[rn1,rn2,...,rnj]]
	
	'''
	# print(rules)
	batch_lens = [[[len(x) for x in y] for y in z] for z in rules]  # Number of rules for a cell in each batch
	sent_lens = [len(x) for x in rules[0]]  # Number of cells at each level
	# print(sent_lens)
	# print([np.max(np.array([len(x) for x in rules[i]])) for i in range(len(rules))])
	max_batch_lens = [[max(max(map(lambda x:x[idx][i],batch_lens)),1) for i in range(l)] for idx,l in enumerate(sent_lens)] # This is a list of list of lists having maximum no of rules for each constituent
	# print(batch_lens,sent_lens)
	pad_lens = [x for x in map(lambda z:[[[max_batch_lens[i][j]-z[i][j] for j in range(sent_lens[i])] for i in range(len(sent_lens))]],batch_lens)]
	padded_batch = [[[rules[i][j][k]+[2500]*pad_lens[i][0][j][k] for k in range(sent_lens[j])] for j in range(len(sent_lens))] for i in range(len(rules))]
	padded_batch = [torch.cat([torch.stack([torch.tensor(padded_batch[i][j][k]) for i in range(len(rules))],dim=0).long() for k in range(sent_lens[j])],dim=-1) for j in range(len(sent_lens))]
	# print("MAX",max_batch_lens)
	rule_indices = [torch.cat([torch.tensor(([idx]*k)).long() for idx,k in enumerate(max_batch_lens[j])],dim=-1).repeat(len(rules)).view(len(rules),-1) for j in range(len(sent_lens))]
	# print(rule_indices)
	mask = [1.0*(x<2500) for x in padded_batch]
	# for i in range(len(rules)):
	#   for j in range(len(rules[i])):
	#       # for k in range(len(rules[i][j])):
	#       del rules[i][j][:]
			# del batch_lens[i][j]
	for i in range(len(max_batch_lens)):
		# for j in range(len(max_batch_lens[i])):
		del max_batch_lens[i][:]

	for i in range(len(pad_lens)):
		# for j in range(len(pad_lens[i])):
		del pad_lens[i][:]
	return padded_batch,mask, rule_indices
class BatchIterator(object):

	def __init__(self, sentences, rules=None,extra={}, **kwargs):
		self.sentences = sentences
		self.rules     = rules
		self.config = config = get_config(get_default_config(), **kwargs)
		self.extra = extra
		self.loader = None

	def chunk(self, tensor, chunks, dim=0, i=0):
		if isinstance(tensor, torch.Tensor):
			return torch.chunk(tensor, chunks, dim=dim)[i]
		index = torch.chunk(torch.arange(len(tensor)), chunks, dim=dim)[i]
		return [tensor[ii] for ii in index]

	def partition(self, tensor, rank, device_ids):
		if tensor is None:
			return None
		if isinstance(tensor, dict):
			for k, v in tensor.items():
				tensor[k] = self.partition(v, rank, device_ids)
			return tensor
		return self.chunk(tensor, len(device_ids), 0, rank)

	def get_dataset_size(self):
		return len(self.sentences)

	def get_dataset_minlen(self):
		return min(map(len, self.sentences))

	def get_dataset_maxlen(self):
		return max(map(len, self.sentences))

	def get_dataset_stats(self):
		return 'size={} minlen={} maxlen={}'.format(
			self.get_dataset_size(), self.get_dataset_minlen(), self.get_dataset_maxlen()
		)

	def choose_negative_samples(self, negative_sampler, k_neg):
		return choose_negative_samples(negative_sampler, k_neg)

	def get_iterator(self, **kwargs):
		config = get_config(self.config.copy(), **kwargs)

		random_seed = config.get('random_seed')
		batch_size = config.get('batch_size')
		filter_length = config.get('filter_length')
		pin_memory = config.get('pin_memory')
		include_partial = config.get('include_partial')
		cuda = config.get('cuda')
		ngpus = config.get('ngpus')
		rank = config.get('rank')
		k_neg = config.get('k_neg')
		negative_sampler = config.get('negative_sampler', None)
		workers = config.get('workers')
		length_to_size = config.get('length_to_size', None)

		def collate_fn(batch):
			index, sents,rules = zip(*batch)
			# sents,rules = zip(*sents)
			# print(sents)
			# assert False
			# sents = sents[0]
			# rules = sents[1]
			# print("----------------Sents len")
			# print([len(x) for x in sents])
			# print([len(x) for x in rules])
			# print(rules[1])
			sents = torch.from_numpy(np.array(sents)).long()

			batch_map = {}
			batch_map['index'] = index
			batch_map['sents'] = sents
			batch_map['rules'],batch_map['rules_mask'], batch_map['rule_indices'] = pad_batch_rules(rules)

			for k, v in self.extra.items():
				batch_map[k] = [v[idx] for idx in index]

			if ngpus > 1:
				for k in batch_map.keys():
					batch_map[k] = self.partition(batch_map[k], rank, range(ngpus))

			return batch_map

		def collate_fn_simple(batch):
			index, sents = zip(*batch)
			sents = torch.from_numpy(np.array(sents)).long()

			batch_map = {}
			batch_map['index'] = index
			batch_map['sents'] = sents
			batch_map['rules'],batch_map['rules_mask'], batch_map['rule_indices'] = None, None, None
			for k, v in self.extra.items():
				batch_map[k] = [v[idx] for idx in index]

			if ngpus > 1:
				for k in batch_map.keys():
					batch_map[k] = self.partition(batch_map[k], rank, range(ngpus))

			return batch_map

		if self.loader is None:
			rng = np.random.RandomState(seed=random_seed)
			if self.rules is not None:
				# print("---------------------------------------\nRules")
				dataset = SimpleDataset([x for x in zip(self.sentences,self.rules)])
				cf = collate_fn
			else:
				dataset = SimpleDataset(self.sentences,item_size=1)
				cf = collate_fn_simple
			sampler = FixedLengthBatchSampler(dataset, batch_size=batch_size, rng=rng,
				maxlen=filter_length, include_partial=include_partial, length_to_size=length_to_size)
			loader = torch.utils.data.DataLoader(dataset, shuffle=(sampler is None), num_workers=workers, pin_memory=pin_memory,batch_sampler=sampler, collate_fn=cf)
			self.loader = loader

		def myiterator():
			# print("Getting new iterator")
			for batch in self.loader:
				index = batch['index']
				sentences = batch['sents']
				rules  = batch['rules']
				mask   = batch['rules_mask']
				indices = batch['rule_indices']

				batch_size, length = sentences.shape

				neg_samples = None
				if negative_sampler is not None:
					neg_samples = self.choose_negative_samples(negative_sampler, k_neg)

				if cuda:
					sentences = sentences.cuda()
					rules = [x.cuda() for x in rules]
					mask  = [x.cuda() for x in mask]
					indices = [x.cuda() for x in indices]

				if cuda and neg_samples is not None:
					neg_samples = neg_samples.cuda()

				batch_map = {}
				batch_map['sentences'] = sentences
				batch_map['rules'] = rules
				batch_map['rules_mask'] = mask
				batch_map['rule_indices'] = indices
				batch_map['neg_samples'] = neg_samples
				batch_map['batch_size'] = batch_size
				batch_map['length'] = length

				for k, v in self.extra.items():
					batch_map[k] = batch[k]

				yield batch_map

		return myiterator()

