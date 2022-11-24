from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import os
import torch

sent_map = {}
sent_map['POS'] = 'positive'
sent_map['NEU'] = 'neutral'
sent_map['NEG'] = 'negative'


def read_data(path, k_shot, absa_task = 'aste'):

	sents = open( path + '.sent', 'r')
	sentences = sents.readlines()
	if absa_task == 'aste':
		tups = open(path +  '.tup', 'r')
	else:
		tups = open(f'{path}.{absa_task}', 'r')
	tuples = tups.readlines()

	if k_shot > 0:
		sentences = sentences[:k_shot]  ## Temporally taking just the first K examples
		#print(sentences)
		tuples = tuples[:k_shot]  ## Temporally taking just the first K examples

	return sentences, tuples


def generate_target(d, absa_task = 'aste'):
	"""
	takes a aspect triple dictionary and linearizes it
	"""
	if absa_task == 'ae':
		summary = ""
		for item in d:
			summary += '<aspect> '
			summary += item + ' '
		
		return summary.strip()
				
	elif absa_task == 'oe':
		summary = ""
		for item in d:
			summary += '<opinion> '
			summary += item + ' '
		
		return summary.strip()

	elif absa_task == 'pair':
		summary = ""
		if len(d.items()) == 0:
			return summary
		for items in d.items():
			summary += '<aspect> '
			summary += items[0] + ' '
			for opinion in items[1]:
				summary += '<opinion> '
				summary += opinion + ' '

		return summary.strip()
				
	elif absa_task == 'aesc':
		summary = ""
		if len(d.items()) == 0:
			return summary
		for items in d.items():
			summary += '<aspect> '
			summary += items[0] + ' '
			for sentiment in items[1]:
				summary += '<sentiment> '
				summary += sentiment + ' '
		return summary.strip()
		
	elif absa_task == 'aste':
		summary = ""
		if len(d.items()) == 0:
			return summary
		for items in d.items():
			summary += '<triplet> '
			summary += items[0] + ' '
			for opinion in items[1]:
				summary += '<opinion> '
				summary += opinion[0] + ' '
				summary += '<sentiment> '
				summary += sent_map[opinion[1]] + ' '

		return summary.strip()
	else:
		raise Exception("Unsupported ABSA task")
	

def generate_aspect_list(tuples, sentence):
	aspects = tuples.split('|')
	aspects = [asp.strip() for asp in aspects]
	aspects = list(set(aspects))
	ordered_aspects = []
	for aspect in aspects:
		ordered_aspects.append( ( sentence.find(aspect.strip())  , aspect) )

	ordered_aspects = sorted(ordered_aspects)

	aspects = [asp[1] for asp in ordered_aspects]

	return aspects

def generate_opinion_list(tuples, sentence):
	opinions = tuples.split('|')
	opinions = [op.strip() for op in opinions]
	opinions = list(set(opinions))

	ordered_opinions = []
	for opinion in opinions:
		ordered_opinions.append( ( sentence.find(opinion.strip())  , opinion) )

	ordered_opinions = sorted(ordered_opinions)

	opinions = [op[1] for op in ordered_opinions]
	return opinions

def generate_pair_dict(tuples, sentence):
	"""
	takes a set of tuples and generates aspect opinion pair dictionary
	"""
	pairs = tuples.split('|')
	d = OrderedDict()
	ordered_pairs = []
	for pair in pairs:
		a, o = pair.split(';')
		ordered_pairs.append( ( sentence.find(a.strip()), sentence.find(o.strip())  , pair) )
	

	ordered_pairs = sorted(ordered_pairs)
	
	for pair in ordered_pairs:
		a, o = pair[2].split(';')
		if(a.strip() in d.keys()):
			d[a.strip()].append(o.strip())
		else:
			d[a.strip()] = []
			d[a.strip()].append(o.strip())
	
	return d 

def generate_aesc_dict(tuples, sentence):
	"""
	takes a set of tuples and generates aspect sentiment pair dictionary
	"""
	aescs = tuples.split('|')
	d = OrderedDict()
	ordered_pairs = []
	for aesc in aescs:
		a, s = aesc.split(';')
		ordered_pairs.append( ( sentence.find(a.strip())  , aesc) )

	ordered_pairs = sorted(ordered_pairs)
	
	for pair in ordered_pairs:
		a, s = pair[1].split(';')
		if(a.strip() in d.keys()):
			d[a.strip()].append(s.strip())
		else:
			d[a.strip()] = []
			d[a.strip()].append(s.strip())
	
	return d 


def generate_triplet_dict(tuples, sentence):
	"""
	takes a set of tuples and generates triplet dictionary
	"""
	triplets = tuples.split('|')
	d = OrderedDict()
	ordered_triplets = []
	for triplet in triplets:
		#print(triplet)
		a, o, _ = triplet.split(';')
		ordered_triplets.append( ( sentence.find(a.strip()), sentence.find(o.strip())  , triplet) )
	
	#print(ordered_triplets)
	ordered_triplets = sorted(ordered_triplets)
	#print(ordered_triplets)
	
	for triplet in ordered_triplets:
		a, o, s = triplet[2].split(';')
		if(a.strip() in d.keys()):
			d[a.strip()].append((o.strip(), s.strip()))
		else:
			d[a.strip()] = []
			d[a.strip()].append((o.strip(), s.strip()))
	
	return d 


def get_transformed_data(sentences_list, tuples_list, absa_task):
	"""
	Preprocess the raw data into Generative Targets
	"""
	inputs = []
	targets = []    
	
	for i in range(len(sentences_list)):
		
		sent = sentences_list[i].strip()
		tup = tuples_list[i]
		if absa_task == 'ae':
			tup_dict = generate_aspect_list(tup, sent)

		elif absa_task == 'oe':
			tup_dict = generate_opinion_list(tup, sent)

		elif absa_task == 'pair':
			tup_dict = generate_pair_dict(tup, sent)

		elif absa_task == 'aesc':
			tup_dict = generate_aesc_dict(tup, sent)
		
		if absa_task == 'aste':
			tup_dict = generate_triplet_dict(tup, sent)

		target = generate_target(tup_dict, absa_task)
		inputs.append(sent)
		targets.append(target)

	return inputs, targets


class ABSA_Dataset(Dataset):

	def __init__(self, tokenizer, data_path , task, absa_task = 'aste', k_shot =-1, max_len=128):
		# 'data/aste/rest16/train.txt'
		self.data_path = data_path
		self.task = task
		self.max_len = max_len
		self.tokenizer = tokenizer
		self.k_shot = k_shot
		self.absa_task = absa_task
		
		self.inputs = []
		self.targets = []
		self.input_tags = []
		self.trip_counts = []

		self._build_examples()

	
	def __len__(self):
		return len(self.inputs)    

	
	def __getitem__(self, index):
		source_ids = self.inputs[index]["input_ids"].squeeze()
		target_ids = self.targets[index]["input_ids"].squeeze()

		src_mask = self.inputs[index]["attention_mask"].squeeze()      # might need to squeeze
		target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
		op_tags = self.input_tags[index].squeeze()
		triplet_count = self.trip_counts[index]

		return {"source_ids": source_ids, "source_mask": src_mask, 
		"target_ids": target_ids, "target_mask": target_mask, 
		"op_tags": op_tags, "triplet_count": triplet_count
		}


	def get_tags(self, text, tuples):
		
		tuples.split('|')
		triplets = tuples.split('|')
		target_tokens = []
		for triplet in triplets:
			a, o, _ = triplet.split(';')
			target_tokens.append(o.strip())

		tokens = self.tokenizer.tokenize(text.strip())
		target = [0 for i in range(len(tokens))]

		for target_token in target_tokens:

			sub_tok = self.tokenizer.tokenize(target_token)

			if(len(sub_tok) == 0):
				continue

			for idx in range(len(tokens) + 1 - len(sub_tok)):
				start_token = tokens[idx]
				match = True

				if sub_tok[0] == start_token:
					for j in range(idx, idx + len(sub_tok)):
						if sub_tok[j - idx] != tokens[j]:
							match = False
					if match:
						target[idx] = 1 ########## 1 == 'B'
						for k in range(idx + 1, idx + len(sub_tok)):
							target[k] = 2 ###########  2 == 'I'

		return target

	
	def get_all_tags(self, sentences_list, tuples_list):
		"""
		Preprocess the raw data into tags for opinion
		"""
		tags = []
		
		for i in range(len(sentences_list)):
			
			sent = sentences_list[i].strip()
			tup = tuples_list[i]
			t = self.get_tags(sent, tup)
			tags.append(t)

		return tags
		
	
	def count_triplets(self, tuples_list):

		trip_count = []

		for i in range(len(tuples_list)):
			trip_count.append( len( tuples_list[i].split('|') ) )

		return trip_count


	def _build_examples(self):

		sentences, tuples = read_data(self.data_path, self.k_shot, self.absa_task)
		inputs, targets = get_transformed_data(sentences, tuples, self.absa_task)
		# input_tags = self.get_all_tags(sentences, tuples)  ### pad this and letzgoooo
		# trip_counts = self.count_triplets(tuples)

		for i in range(len(inputs)):

			input = inputs[i]
			target = targets[i]
			#input_tag = input_tags[i]
			#trip_count = trip_counts[i]

			tokenized_input = self.tokenizer(
			  [input], max_length=self.max_len, pad_to_max_length=True, truncation=True,
			  return_tensors="pt",
			)
			
			with self.tokenizer.as_target_tokenizer():
				tokenized_target = self.tokenizer(
				[target], max_length=self.max_len, pad_to_max_length=True, truncation=True,
				return_tensors="pt"
			)

			# input_tag = input_tag + [0] * (self.max_len - len(input_tag))
			# input_tag = torch.tensor(input_tag)
			# trip_count = torch.tensor(trip_count)

			#self.input_tags.append(input_tag)
			self.inputs.append(tokenized_input)
			self.targets.append(tokenized_target)
			#self.trip_counts.append(trip_count)