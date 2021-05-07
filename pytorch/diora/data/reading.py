"""
Each reader should return:

	- sentences - This is the primary input (raw text) to the model. Not tokenized.
	- extra - Additional model input such as entity or sentence labels.
	- metadata - Info about the data that is not specific to examples / batches.

"""

import os
import json

from tqdm import tqdm


def pick(lst, k):
	return [d[k] for d in lst]


def flatten_tree(tr):
	def func(tr):
		if not isinstance(tr, (list, tuple)):
			return [tr]
		result = []
		for x in tr:
			result += func(x)
		return result
	return func(tr)


def convert_binary_bracketing(parse, lowercase=True):
	transitions = []
	tokens = []

	for word in parse.split(' '):
		if len(word) and word[0] != "(":
			if word == ")":
				transitions.append(1)
			else:
				# Downcase all words to match GloVe.
				if lowercase:
					tokens.append(word.lower())
				else:
					tokens.append(word)
				transitions.append(0)
	return tokens, transitions


def build_tree(tokens, transitions):
	stack = []
	buf = tokens[::-1]

	for t in transitions:
		if t == 0:
			stack.append(buf.pop())
		elif t == 1:
			right = stack.pop()
			left = stack.pop()
			stack.append((left, right))

	assert len(stack) == 1

	return stack[0]


def get_spans_and_siblings(tree):
	def helper(tr, idx=0, name='root'):
		if isinstance(tr, (str, int)):
			return 1, [(idx, idx+1)], []

		l_size, l_spans, l_sibs = helper(tr[0], name='l', idx=idx)
		r_size, r_spans, r_sibs = helper(tr[1], name='r', idx=idx+l_size)

		size = l_size + r_size

		# Siblings.
		spans = [(idx, idx+size)] + l_spans + r_spans
		siblings = [(l_spans[0], r_spans[0], name)] + l_sibs + r_sibs

		return size, spans, siblings

	_, spans, siblings = helper(tree)

	return spans, siblings


def get_spans(tree):
	def helper(tr, idx=0):
		if isinstance(tr, (str, int)):
			return 1, []

		spans = []
		sofar = idx

		for subtree in tr:
			size, subspans = helper(subtree, idx=sofar)
			spans += subspans
			sofar += size

		size = sofar - idx
		spans += [(idx, sofar)]

		return size, spans

	_, spans = helper(tree)

	return spans


class BaseTextReader(object):
	def __init__(self, lowercase=True, filter_length=0, include_id=False):
		self.lowercase = lowercase
		self.filter_length = filter_length if filter_length is not None else 0
		self.include_id = include_id

	def read(self, filename):
		return self.read_sentences(filename)

	def read_sentences(self, filename):
		sentences = []
		extra = dict()

		example_ids = []

		with open(filename) as f:
			for line in tqdm(f, desc='read'):
				for s in self.read_line(line):
					if self.filter_length > 0 and len(s) > self.filter_length:
						continue
					if self.include_id:
						example_id = s[0]
						s = s[1:]
					else:
						example_id = len(sentences)
					if self.lowercase:
						s = [w.lower() for w in s]
					example_ids.append(example_id)
					sentences.append(s)

		extra['example_ids'] = example_ids

		return {
			"sentences": sentences,
			"extra": extra
			}

	def read_line(self, line):
		raise NotImplementedError


class PlainTextReader(BaseTextReader):
	def __init__(self, lowercase=True, filter_length=0, delim=' ', include_id=False):
		super(PlainTextReader, self).__init__(lowercase=lowercase, filter_length=filter_length, include_id=include_id)
		self.delim = delim

	def read_line(self, line):
		s = line.strip().split(self.delim)
		if self.lowercase:
			s = [w.lower() for w in s]
		yield s


class JSONLReader(object):
	def __init__(self, lowercase=True, filter_length=0, delim=' ', include_id=False):
		self.lowercase = lowercase
		self.filter_length = filter_length if filter_length is not None else 0

	def read(self, filename):
		sentences = []

		# extra
		extra = dict()
		example_ids = []
		trees = []

		# read
		with open(filename) as f:
			for line in tqdm(f, desc='read'):
				ex = json.loads(line)
				example_id = ex['example_id']
				tr = ex['tree']
				if not 'sentence' in ex:
					ex['sentence'] = flatten_tree(tr)
				s = ex['sentence']

				if self.filter_length > 0 and len(s) > self.filter_length:
					continue
				if self.lowercase:
					s = [w.lower() for w in s]

				example_ids.append(example_id)
				sentences.append(s)
				trees.append(tr)

		extra['example_ids'] = example_ids
		extra['trees'] = trees

		return {
			"sentences": sentences,
			"extra": extra
			}


class NLIReader(object):

	LABEL_MAP = {
		"entailment": 0,
		"neutral": 1,
		"contradiction": 2,
		"yes" : 3
	}

	def __init__(self, lowercase=True, filter_length=0):
		self.lowercase = lowercase
		self.filter_length = filter_length if filter_length is not None else 0

	@staticmethod
	def build(lowercase=True, filter_length=0):
		return NLISentenceReader(lowercase=True, filter_length=0)

	def read(self, filename):
		return self.read_sentences(filename)

	def read_sentences(self, filename):
		raise NotImplementedError

	def read_line(self, line):
		example = json.loads(line)
		try:
			label = self.read_label(example['gold_label'])
		except:
			return None

		s1, t1 = convert_binary_bracketing(example['sentence1_binary_parse'], lowercase=self.lowercase)
		s2, t2 = convert_binary_bracketing(example['sentence2_binary_parse'], lowercase=self.lowercase)
		# print(example)
		try:
			r1, r2 = example['sentence1_rule'], example['sentence2_rule'] 
		except:
			r1 = []
			r2 = []
		example_id = example['pairID']

		return dict(s1=s1, label=label, s2=s2, r1=[x[2] for x in r1],r2=[x[2] for x in r2],t1=t1, t2=t2, example_id=example_id)

	def read_label(self, label):
		return self.LABEL_MAP[label]


class NLISentenceReader(NLIReader):
	def read_sentences(self, filename):
		sentences = []
		rules = []
		extra = {}
		example_ids = []
		# print("READING SENTENCES from {}".format(filename))
		with open(filename) as f:
			for line in tqdm(f, desc='read'):
				# print(line)
				try:
					smap = self.read_line(line)
				except Exception as e:
					# print(e)
					continue
				if smap is None:
					continue

				s1, s2, r1,r2, label = smap['s1'], smap['s2'], smap['r1'],smap['r2'],smap['label']
				example_id = smap['example_id']
				skip_s1 = (self.filter_length > 0 and len(s1) > self.filter_length) or (not len(s1))
				skip_s2 = (self.filter_length > 0 and len(s2) > self.filter_length) or (not len(s2))
				# print(s1)
				if not skip_s1:
					# print("App")
					example_ids.append(example_id + '_1')
					sentences.append(s1)
					rules.append(r1)
				if not skip_s2:
					example_ids.append(example_id + '_2')
					sentences.append(s2)
					rules.append(r2)
		# print(sentences)
		extra['example_ids'] = example_ids
		# print("READING SENTENCES {}".format(len(sentences)))
		return {
			"sentences": sentences,
			"rules": rules,
			"extra": extra,
			}


class ConllReader(object):
	def __init__(self, lowercase=True, filter_length=0):
		self.lowercase = lowercase
		self.filter_length = filter_length if filter_length is not None else 0

	def read(self, filename):
		sentences = []
		extra = {}
		example_ids = []
		entity_labels = []

		with open(filename) as f:
			for line in tqdm(f, desc='read'):
				data = json.loads(line)
				s = data['sentence']

				# skip long sentences
				if self.filter_length > 0 and len(s) > self.filter_length:
					continue

				sentences.append(s)
				example_ids.append(data['example_id'])
				entity_labels.append(data['entities'])

		extra['example_ids'] = example_ids
		extra['entity_labels'] = entity_labels

		return {
			"sentences": sentences,
			"extra": extra,
			}


class SyntheticReader(object):
	def __init__(self, nexamples=100, embedding_size=10, vocab_size=14, seed=11, minlen=10,
				 maxlen=20, length=None):
		super(SyntheticReader, self).__init__()
		self.nexamples = nexamples
		self.embedding_size = embedding_size
		self.vocab_size = vocab_size
		self.seed = seed
		self.minlen = minlen
		self.maxlen = maxlen
		self.length = length

	def read(self, filename=None):
		min_length = self.minlen
		max_length = self.maxlen

		if self.length is not None:
			min_length = self.length
			max_length = min_length + 1

		sentences = synthesize_training_data(self.nexamples, self.vocab_size,
			min_length=min_length, max_length=max_length, seed=self.seed)

		metadata = {}
		metadata['embeddings'] = np.random.randn(self.vocab_size, self.embedding_size).astype(np.float32)

		return {
			"sentences": sentences,
			"extra": extra,
			"metadata": metadata
			}

