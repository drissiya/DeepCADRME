import sys
import numpy as np
from pathlib import Path
import pickle as pkl
import codecs
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
import torch

def convert_examples_to_features(examples, max_seq_length, tokenizer, label_list, print_examples=False):
    label_path = 'pretrained_models/label2id_bert.pkl'        
    if not Path(label_path).is_file():
        label_map = {}
        for (i, label) in enumerate(label_list, 1): 
            label_map[label] = i
        with codecs.open(label_path, 'wb') as w:
            pkl.dump(label_map, w)
    else:
        with open(label_path, 'rb') as f: 
            label_map = pkl.load(f)
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = example.text_a
        lev1 = example.label[0]
        lev2 = example.label[1]
        lev3 = example.label[2]
        bert_tokens = []
        level1 = []
        level2 = []
        level3 = []
        orig_to_tok_map = []
        tok_len_map = []
        bert_tokens.append("[CLS]")
        level1.append(label_map['[CLS]'])
        level2.append(label_map['[CLS]'])
        level3.append(label_map['[CLS]'])
        for token, l1, l2, l3 in zip(tokens_a, lev1, lev2, lev3):   
            orig_to_tok_map.append(len(bert_tokens))
            t = tokenizer.tokenize(token)
            bert_tokens.extend(t)
            tok_len_map.append(len(bert_tokens)-orig_to_tok_map[-1])
            for m in range(len(t)):
                if m==0:
                    level1.append(label_map[l1])
                    level2.append(label_map[l2])
                    level3.append(label_map[l3])
                else:
                    level1.append(label_map['X'])
                    level2.append(label_map['X'])
                    level3.append(label_map['X'])
                  

        if len(bert_tokens) > max_seq_length - 2:
            #print (len(bert_tokens))
            bert_tokens = bert_tokens[:(max_seq_length - 2)]
            level1 = level1[:(max_seq_length - 2)]
            level2 = level2[:(max_seq_length - 2)]
            level3 = level3[:(max_seq_length - 2)]
            
        tokens = bert_tokens + ["[SEP]"]

        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        level1 = level1 + [label_map['[SEP]']]
        level2 = level2 + [label_map['[SEP]']]
        level3 = level3 + [label_map['[SEP]']]      

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        level1 += padding
        level2 += padding
        level3 += padding
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(level1) == max_seq_length
        assert len(level2) == max_seq_length
        assert len(level3) == max_seq_length
        

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("level 1: %s" % " ".join([str(x) for x in level1]))
            print("level 2: %s" % " ".join([str(x) for x in level2]))
            print("level 3: %s" % " ".join([str(x) for x in level3]))
        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      level1=level1,
                                      level2=level2,
                                      level3=level3))
    return features

	
class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, level1, level2, level3):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.level1 = level1
        self.level2 = level2
        self.level3 = level3
		
class TACLoader(object):
    def __init__(self, args, processor, tokenizer):
        self.args = args
        self.processor = processor
        self.tokenizer = tokenizer
        self.train_examples = self.processor.get_train_examples()
        self.test_examples = self.processor.get_test_examples()
        self.valid_examples = self.processor.get_valid_examples()

    def get_data_loader(self, data_examples, print_examples=False):
        data_features = convert_examples_to_features(
                data_examples, self.args.max_seq_length, self.tokenizer, self.processor.get_labels(), print_examples)

        input_ids = [f.input_ids for f in data_features]
        input_mask = [f.input_mask for f in data_features]
        segment_ids = [f.segment_ids for f in data_features]
        level1 = [f.level1 for f in data_features]
        level2 = [f.level2 for f in data_features]
        level3 = [f.level3 for f in data_features]


        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        level1 = torch.tensor(level1, dtype=torch.long)
        level2 = torch.tensor(level2, dtype=torch.long)
        level3 = torch.tensor(level3, dtype=torch.long)
        
        data_tensor = TensorDataset(input_ids, input_mask, segment_ids, level1, level2, level3)
        data_sampler = SequentialSampler(data_tensor)
        dataloader = DataLoader(data_tensor, sampler=data_sampler, batch_size=self.args.batch_size)  
        return dataloader

    def get_all_data_loader(self, print_examples=False):
        train_loader = self.get_data_loader(self.train_examples, print_examples)
        test_loader = self.get_data_loader(self.test_examples, print_examples)
        valid_loader = self.get_data_loader(self.valid_examples, print_examples)
        return train_loader, valid_loader, test_loader