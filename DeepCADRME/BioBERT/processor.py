import sys
import numpy as np
from pathlib import Path
import pickle as pkl
import codecs
	
class InputExample(object):
    def __init__(self, guid, text_a, label, pos):
        self.guid = guid
        self.text_a = text_a
        self.label = label
        self.pos = pos


class DataProcessor(object):
    def __init__(self, train_data, test_data, dev_data=None):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

    def get_train_examples(self):
        return self._create_example(self.read_data(self.train_data), "train")

    def get_test_examples(self):
        return self._create_example(self.read_data(self.test_data), "test")
    
    def get_valid_examples(self):
        return self._create_example(self.read_data(self.dev_data), "valid")

    def get_labels(self):
        l1 = set([a for l in self.train_data.t_segment_mention for i in l for a in i])
        l2 = []
        if self.dev_data is not None:
            l2 = set([a for l in self.dev_data.t_segment_mention for i in l for a in i])
        l3 = set([a for l in self.test_data.t_segment_mention for i in l for a in i])         
        return set(list(l1)+list(l2)+list(l3) + ['X', '[CLS]', '[SEP]'])

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines): 
            guid  = "%s-%s" % (set_type, i)
            sentence = line[0]
            label = line[1] 
            pos = line[2] 
            examples.append(InputExample(guid=guid, text_a=sentence, label=label, pos=pos))
        return examples

    def read_data(self, tac_set):
        lines  = []
        for i in range(len(tac_set.t_toks_mention)):
            sentence = tac_set.t_toks_mention[i]
            label = tac_set.t_segment_mention[i]
            pos = tac_set.t_pos[i]
            lines.append([sentence, label, pos])
        return lines

