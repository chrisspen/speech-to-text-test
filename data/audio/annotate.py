#!/usr/bin/env python
"""
Builds annoations.yaml.
"""
from __future__ import print_function
import os
from collections import OrderedDict

import yaml

RATE16K_MONO = 'rate16k-mono'

# Force dictionaries to be serialized in multi-line format.
def _represent_dictorder(self, data):
	return self.represent_mapping(u'tag:yaml.org,2002:map', sorted(data.items()))

def _represent_tuple(self, data):
	return self.represent_sequence(u'tag:yaml.org,2002:seq', data)

def _construct_tuple(self, node):
	return tuple(self.construct_sequence(node))

yaml.add_representer(dict, _represent_dictorder)
yaml.add_representer(OrderedDict, _represent_dictorder)
yaml.add_constructor(u'tag:yaml.org,2002:python/tuple', _construct_tuple)

def convert_to_rate16mono(base_name):
	print('Converting %s...' % base_name)
	os.system('sox {name}.wav {name}.rate16k-mono.wav channels 1 rate 16k'.format(name=base_name))

annotations_fn = 'annotations.yaml'
os.system('touch %s' % annotations_fn)
data = yaml.load(open(annotations_fn)) or {}

all_names = set()
converted_rate16k_mono = set()

for fn in os.listdir('.'):
	_, ext = os.path.splitext(fn)
	ext = ext.lower()
	if ext in ('.mp3', '.wav'):
		data.setdefault(fn, '')
		parts = fn.split('.')
		base_name = parts[0]
		all_names.add(base_name)
		if len(parts) > 2:
			conversion_type = parts[1]
			if conversion_type == RATE16K_MONO:
				converted_rate16k_mono.add(base_name)

unconverted = all_names.difference(converted_rate16k_mono)
for _name in unconverted:
	convert_to_rate16mono(_name)

for _name in data:
	if not data[_name] and RATE16K_MONO in _name:
		_other_name = _name.replace('.'+RATE16K_MONO, '')
		data[_name] = data[_other_name]

print('Found %i audio files.' % len(data))
print('Found %i unconverted to rate 16k with mono.' % len(unconverted))
#print(data)
with open(annotations_fn, 'w') as fout:
	yaml.dump(data, fout, default_flow_style=False)

print('Done.')
