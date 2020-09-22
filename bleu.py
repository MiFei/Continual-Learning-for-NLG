"""
Implementation of calculation of bleu value
"""

import os
import sys
import json
import argparse
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import time

def score_woz3(res_file, ignore=False):
	"""
	Compute score
	"""
	feat2content = {}
	with open(res_file) as f:
		for line in f:
			if 'Feat' in line:
				feat = line.strip().split(':')[1][1:]
	
				if feat not in feat2content:
					feat2content[feat] = [[], [], []] # [ [refs], [bases], [gens] ]
				continue

			if 'Target' in line:
				target = line.strip().split(':')[1][1:]
				if feat in feat2content:
					feat2content[feat][0].append(target)
	
			if 'Base' in line:
				base = line.strip().split(':')[1][1:]
				if base[-1] == ' ':
					base = base[:-1]
				if feat in feat2content:
					feat2content[feat][1].append(base)
	
			if 'Gen' in line:
				gen = line.strip().split(':')[1][1:]
				if feat in feat2content:
					feat2content[feat][2].append(gen)

	return feat2content

def get_bleu(feat2content, template=False, ignore=False):
	"""
	Get bleu value
	"""

	test_type = 'base' if template else 'gen'
	print('Start', test_type, file=sys.stderr)

	gen_count = 0
	list_of_references, hypotheses = {'gen': [], 'base': []}, {'gen': [], 'base': []}
	for feat in feat2content:
		refs, bases, gens = feat2content[feat]
		gen_count += len(gens)
		refs = [s.split() for s in refs]
	
		for gen in gens:
			gen = gen.split()
			list_of_references['gen'].append(refs)
			hypotheses['gen'].append(gen)
	
		for base in bases:
			base = base.split()
			list_of_references['base'].append(refs)
			hypotheses['base'].append(base)
	
	
	print('TEST TYPE:', test_type)
	print('Ignore General Acts:', ignore)
	smooth = SmoothingFunction()
	print('Calculating BLEU...', file=sys.stderr)
	print( 'Avg # feat:', len(feat2content) )
	print( 'Avg # gen: {:.2f}'.format(gen_count / len(feat2content)) )
	BLEU = []
	weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.333, 0.333, 0.333, 0), (0.25, 0.25, 0.25, 0.25)]
	for i in range(4):
		if i == 0 or i == 1 or i == 2:
			continue
		t = time.time()
		bleu = corpus_bleu(list_of_references[test_type], hypotheses[test_type], weights=weights[i], smoothing_function=smooth.method1)
		BLEU.append(bleu)
		print('Done BLEU-{}, time:{:.1f}'.format(i+1, time.time()-t))
	print('BLEU 1-4:', BLEU)
	print('BLEU 1-4:', BLEU, file=sys.stderr)
	print('Done', test_type, file=sys.stderr)
	print('-----------------------------------')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train dialogue generator')
	parser.add_argument('--res_file', type=str, help='result file')
	parser.add_argument('--dataset', type=str, default='woz3', help='result file')
	parser.add_argument('--template', type=bool, default=False, help='test on template-based words')
	parser.add_argument('--ignore', type=bool, default=False, help='whether to ignore general acts, e.g. bye')
	args = parser.parse_args()
	assert args.dataset == 'woz3' or args.dataset == 'domain4'
	if args.dataset == 'woz3':
		assert args.template is False
		feat2content = score_woz3(args.res_file, ignore=args.ignore)
	else: # domain4
		assert args.ignore is False
		feat2content = score_domain4(args.res_file)
	get_bleu(feat2content, template=args.template, ignore=args.ignore)
