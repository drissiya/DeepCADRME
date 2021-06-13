from DeepCADRME.BiLSTM.embeddings import POSEmbedding, WordEmbedding, CharEmbedding
from data_utils.tac_corpus import convert_data
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import torchtext
import os


def data_to_csv(max_levels, data_examples, data_dir, guid):
    tag = convert_data(data_examples.t_segment_mention, max_levels)
    text = [" ".join(j) for j in data_examples.t_toks_mention]
    pos = [" ".join(j) for j in data_examples.t_pos]
    level1 = [" ".join(j) for j in tag[0]]
    level2 = [" ".join(j) for j in tag[1]]
    level3 = [" ".join(j) for j in tag[2]]
    data = pd.DataFrame({'text':text, 'pos':pos, 'char':text, 'level1': level1, 'level2': level2, 'level3': level3})
    data.to_csv(os.path.join(data_dir, guid + '.csv'), index = False, header=True) 
	
class TACIterator(object):
    def __init__(self, args, train_examples, test_examples, valid_examples, device):
        self.args = args
        self.train_examples = train_examples
        self.test_examples = test_examples
        self.valid_examples = valid_examples
        self.device = device
        self.TEXT = None
        self.POS = None
        self.CHAR = None
        self.TAG = None

    def get_data_iterator(self):
        data_to_csv(self.args.max_levels, self.train_examples, self.args.data_dir, guid='train')
        data_to_csv(self.args.max_levels, self.test_examples, self.args.data_dir, guid='test')
        data_to_csv(self.args.max_levels, self.valid_examples, self.args.data_dir, guid='valid')

        word_feature = WordEmbedding(self.args, self.train_examples, self.test_examples, self.valid_examples)
        if not Path(self.args.embedding_word_path).is_file():
            word_feature.load_pretrained_word_emb()
        else:
            word_feature.load()
    
        pos_feature = POSEmbedding(self.args, self.train_examples, self.test_examples, self.valid_examples)
        if not Path(self.args.embedding_pos_path).is_file():
            pos_feature.train()
        else:
            pos_feature.load()

        char_feature = CharEmbedding(self.args, self.train_examples, self.test_examples, self.valid_examples)
        if not Path(self.args.embedding_char_path).is_file():
            char_feature.train()
        else:
            char_feature.load()

        self.TEXT = word_feature.word_field
        self.POS = pos_feature.pos_field
        self.CHAR = char_feature.char_field
        self.TAG = torchtext.legacy.data.Field(unk_token = None) 

        train_data, valid_data, test_data = torchtext.legacy.data.TabularDataset.splits(
                path= self.args.data_dir ,
                train="train.csv",
                validation="valid.csv",
                test="test.csv", format='csv', skip_header=True,
                fields=(("text", self.TEXT), ("pos", self.POS), ("char", self.CHAR), ("level1", self.TAG), ("level2", self.TAG), ("level3", self.TAG))
            )
        

        word_vectors = word_feature.embedding_word
        pos_vectors = pos_feature.embedding_pos
        char_vectors = char_feature.embedding_char

        self.TEXT.vocab.set_vectors(
                stoi=self.TEXT.vocab.stoi,
                vectors=word_vectors,
                dim=self.args.word_dim
            )

        self.POS.vocab.set_vectors(
                stoi=self.POS.vocab.stoi,
                vectors=pos_vectors,
                dim=self.args.pos_dim
            )
        
        self.CHAR.vocab.set_vectors(
                stoi=self.CHAR.vocab.stoi,
                vectors=char_vectors,
                dim=self.args.char_dim
            )

        self.TAG.build_vocab(test_data)

        train_iterator, valid_iterator, test_iterator = torchtext.legacy.data.BucketIterator.splits(
            (train_data, valid_data, test_data), 
            batch_size = self.args.batch_size,
            device = self.device, sort=False)
        return train_iterator, valid_iterator, test_iterator