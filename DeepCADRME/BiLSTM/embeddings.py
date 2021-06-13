from gensim.models import Word2Vec
from torchtext.vocab import Vocab
from collections import Counter
from DeepCADRME.BiLSTM.modified_orelm.OR_ELM import ORELM
import pickle as pkl
import numpy as np
import codecs
import torchtext
import gensim
import torch


class WordEmbedding(object):
    def __init__(self, args, train_data, test_data, dev_data=None):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.args = args
        self.word_field = None
        self.embedding_word = None
        
    def get_word_vocab(self):
        data = []
        vocab = []
        for sentence in self.train_data.t_toks_mention:
            data.append(sentence)
            vocab.extend(sentence)
        if self.dev_data is not None:
            for sentence in self.dev_data.t_toks_mention:
                data.append(sentence)
                vocab.extend(sentence)
        for sentence in self.test_data.t_toks_mention:
            data.append(sentence)
            vocab.extend(sentence)
        return data, list(set(vocab))
    
    def generate_embedding(self, wv_model, word_field):
        vectors = []
        for word, idx in word_field.vocab.stoi.items():
            if word in wv_model.vocab.keys():
                vectors.append(torch.as_tensor(wv_model[word].tolist()))
            else:
                vectors.append(torch.zeros(self.args.word_dim))
        return vectors
    
    def load_pretrained_word_emb(self):
        word_tac, vocab_word = self.get_word_vocab()
        word2idx = {w: i + 1 for i, w in enumerate(vocab_word)}
        word_counter = Counter(word2idx)
        word_field = torchtext.legacy.data.Field(lower=True)
        word_field.vocab = Vocab(word_counter, min_freq=0)

        self.word_field = word_field
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.args.path_pretrained_word_emb, binary=True)
        embedding_word = self.generate_embedding(word2vec, word_field)
        self.embedding_word = embedding_word

        with codecs.open(self.args.word2idx_path, 'wb') as w:
            pkl.dump(word_field, w)

        with codecs.open(self.args.embedding_word_path, 'wb') as w:
            pkl.dump(embedding_word, w)

    def load(self):
        with codecs.open(self.args.embedding_word_path, 'rb') as f: 
            embedding_word = pkl.load(f)

        with codecs.open(self.args.word2idx_path, 'rb') as f: 
            word_field = pkl.load(f)

        self.embedding_word = embedding_word
        self.word_field = word_field

class POSEmbedding(object):
    def __init__(self, args, train_data, test_data, dev_data=None):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.args = args
        self.pos_field = None
        self.embedding_pos = None
        
    def get_pos_vocab(self):
        data = []
        vocab = []
        for sentence in self.train_data.t_pos:
            data.append(sentence)
            vocab.extend(sentence)
        if self.dev_data is not None:
            for sentence in self.dev_data.t_pos:
                data.append(sentence)
                vocab.extend(sentence)
        for sentence in self.test_data.t_pos:
            data.append(sentence)
            vocab.extend(sentence)
        return data, list(set(vocab))
    
    def generate_embedding(self, wv_model, word_field):
        vectors = []
        for word, idx in word_field.vocab.stoi.items():
            if word in wv_model.wv.vocab.keys():
                vectors.append(torch.as_tensor(wv_model.wv[word].tolist()))
            else:
                vectors.append(torch.zeros(self.args.pos_dim))
        return vectors
    
    def train(self):
        pos_tac, vocab_pos = self.get_pos_vocab()
        pos2idx = {w: i + 1 for i, w in enumerate(vocab_pos)}

        pos_counter = Counter(pos2idx)
        pos_field = torchtext.legacy.data.Field(lower=False)
        pos_field.vocab = Vocab(pos_counter, min_freq=0)

        self.pos_field = pos_field

        pos2vec = Word2Vec(sentences=pos_tac, size=self.args.pos_dim, window=5, min_count=0, max_vocab_size=len(pos_tac))
        pos2vec.train(pos_tac, total_words=len(pos_tac), epochs=3)

        embedding_pos = self.generate_embedding(pos2vec, pos_field)
        self.embedding_pos = embedding_pos

        with codecs.open(self.args.pos2idx_path, 'wb') as w:
            pkl.dump(pos_field, w)

        with codecs.open(self.args.embedding_pos_path, 'wb') as w:
            pkl.dump(embedding_pos, w)

    def load(self):
        with codecs.open(self.args.embedding_pos_path, 'rb') as f: 
            embedding_pos = pkl.load(f)

        with codecs.open(self.args.pos2idx_path, 'rb') as f: 
            pos_field = pkl.load(f)

        self.embedding_pos = embedding_pos
        self.pos_field = pos_field


class CharEmbedding(object):
    """
    character-level embedding with the modified OR-ELM

    """
    def __init__(self, args, train_data, test_data, dev_data=None):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.args = args
        self.vocab_chars = set()
        self.word2vec_char = dict()
        self.embedding_char = None
        self.char_field = None
        
    def get_word_vocab(self):
        vocab = []
        for sentence in self.train_data.t_toks_mention:
            vocab.extend(sentence)
        if self.dev_data is not None:
            for sentence in self.dev_data.t_toks_mention:
                vocab.extend(sentence)
        for sentence in self.test_data.t_toks_mention:
            vocab.extend(sentence)
        return list(set(vocab))

    def map_character(self, char_map, tokset):
        char_word = dict()
        for word in tokset:
            temp = []
            for tup in word:
                temp.append(char_map[tup])
            char_word[word] = temp
        return char_word
  
    def train_modified_orelm(self):
        tokset = self.get_word_vocab()
        vocab = set([w_i for w in tokset for w_i in w])
        self.vocab_chars = vocab
        n_chars = len(self.vocab_chars) + 2
        self.n_chars = n_chars
        char_map = {ind: np.random.uniform(-0.25,0.25, self.args.char_dim).reshape(1,self.args.char_dim) for ind in self.vocab_chars}
        char_word = self.map_character(char_map, tokset)
        net = ORELM(inputs=self.args.char_dim, numHiddenNeurons=self.args.orelm_hidden_dim, activationFunction='sig')
        net.initializePhase(lamb = self.args.regularization)
        for word, vec in zip(char_word.keys(), char_word.values()):
            vec = np.asarray(vec)
            for i in range(len(vec)):
                H = net.predict_H(vec[i]) 
            self.word2vec_char[word] = H[0] * 2 - 1

    def generate_embedding(self, model_char, word_field):
        vectors = []
        for word, idx in word_field.vocab.stoi.items():
            if word in list(model_char.keys()):
                vectors.append(torch.as_tensor(model_char[word].tolist())) 
            else:
                vectors.append(torch.zeros(self.args.orelm_hidden_dim))
        return vectors
    
    def train(self):
        self.train_modified_orelm()
        with codecs.open(self.args.word2idx_path, 'rb') as f: 
            char_field = pkl.load(f)    
        self.char_field = char_field     
        embedding_char = self.generate_embedding(self.word2vec_char, self.char_field)
        self.embedding_char = embedding_char
        with codecs.open(self.args.embedding_char_path, 'wb') as w:
            pkl.dump(embedding_char, w)
            
    def load(self):
        with codecs.open(self.args.embedding_char_path, 'rb') as f: 
            embedding_char = pkl.load(f)
        self.embedding_char = embedding_char
        with codecs.open(self.args.word2idx_path, 'rb') as f: 
            char_field = pkl.load(f)    
        self.char_field = char_field