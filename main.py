from data_utils.tac_corpus import TAC, split_data
from argparse import ArgumentParser

from DeepCADRME.BiLSTM.iterator import TACIterator
from DeepCADRME.BiLSTM.model import BiLSTMModel
from DeepCADRME.BiLSTM.learner import BiLSTMLearner
from DeepCADRME.BiLSTM.inference import BiLSTMInference

from DeepCADRME.BioBERT.loader import TACLoader
from DeepCADRME.BioBERT.processor import DataProcessor
from DeepCADRME.BioBERT.model import BioBERTModel
from DeepCADRME.BioBERT.learner import BioBERTLearner
from DeepCADRME.BioBERT.inference import BioBERTInference

from DeepCADRME.utils import init_weights, get_optimizer, write_guess_xml_files
from DeepCADRME.extract import extract_guess_ADE_mention

from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import codecs
import random
import os
from collections import OrderedDict


def get_args():
    parser = ArgumentParser(description="DeepCADRME: A deep neural model for complex adverse drug reaction mentions extraction")

    parser.add_argument('--data-dir', type=str, default='TAC')
    parser.add_argument('--train-xml-dir', type=str, default='train_xml')
    parser.add_argument('--gold-xml-dir', type=str, default='gold_xml')
    parser.add_argument('--train-dir-sentences', type=str, default='TR')
    parser.add_argument('--gold-dir-sentences', type=str, default='TE')
    parser.add_argument('--guess-xml-dir', type=str, default='guess_xml')
    parser.add_argument('--output_dir', type=str, default='checkpoint')

    parser.add_argument('--pretrained-models-dir', type=str, default='pretrained_models')
    parser.add_argument('--pretrained-word-emb', type=str, default='PMC-w2v.bin', 
                        help='wikipedia-pubmed-and-PMC-w2v.bin/PMC-w2v.bin/PubMed-and-PMC-w2v.bin/PubMed-w2v.bin')
    parser.add_argument('--emb-type', type=str, default='word', help='word/word+pos/word+char/word+pos+char')
    parser.add_argument('--bert-type', type=str, default='biobert_trained_on_pmc',
                        help='biobert_trained_on_pmc/biobert_trained_on_pubmed/biobert_trained_on_pubmed_pmc/bert')
    parser.add_argument('--model-type', type=str, default='biobert', 
                        help='biobert/bilstm')
    parser.add_argument('--step', type=str, default='train',
                        help='train/test')
    
    parser.add_argument('--vocab-file', type=str, default='vocab.txt')
    parser.add_argument('--config-file', type=str, default='config.json')
    parser.add_argument('--bert_model_file', type=str, default='pytorch_model.bin')
     
    
    parser.add_argument('--pos-dim', type=int, default=50)
    parser.add_argument('--word-dim', type=int, default=200)
    parser.add_argument('--char-dim', type=int, default=20)
    parser.add_argument('--orelm-hidden-dim', type=int, default=20)
    parser.add_argument('--regularization', type=float, default=0.001)
    parser.add_argument('--hidden-lstm', type=int, default=100)
    
    
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-seq-length', type=int, default=140)
    parser.add_argument('--epochs', type=int, default=9)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--lr', type=float, default=5e-5)# 5e-5
    parser.add_argument('--dropout', type=int, default=0.5)
       

    args = parser.parse_args()
    return args
	
def main():
    args = get_args()
    args.max_levels = 3

    train_xml_dir_path = os.path.join(args.data_dir, args.train_xml_dir)
    gold_xml_dir_path = os.path.join(args.data_dir, args.gold_xml_dir)
    guess_xml_dir_path = os.path.join(args.data_dir, args.guess_xml_dir)
    train_dir_sentences_path = os.path.join(args.data_dir, args.train_dir_sentences)
    gold_dir_sentences_path = os.path.join(args.data_dir, args.gold_dir_sentences)

    TAC_train = TAC(max_level=args.max_levels, 
                    sentence_dir=train_dir_sentences_path, 
                    label_dir=train_xml_dir_path)
    TAC_train.load_corpus()

    TAC_train, TAC_valid = split_data(TAC_train)

    TAC_test = TAC(max_level=args.max_levels, 
                   sentence_dir=gold_dir_sentences_path, 
                   label_dir=gold_xml_dir_path)
    TAC_test.load_corpus()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\tModel initialized with {args.model_type}')

    if args.model_type == 'biobert':
        processor = DataProcessor(TAC_train, 
                                  TAC_test, 
                                  TAC_valid)

        tokenizer = BertTokenizer(vocab_file=os.path.join(args.pretrained_models_dir, args.bert_type, args.vocab_file), 
                                  do_lower_case=False)
        tac_loader = TACLoader(args, 
                              processor, 
                              tokenizer)
        train_loader, valid_loader, test_loader  = tac_loader.get_all_data_loader(print_examples=False)

        args.num_labels = len(processor.get_labels()) + 1
        
        with open(os.path.join(args.pretrained_models_dir, 'label2id_bert.pkl'), 'rb') as f: 
            label_map = pkl.load(f)
        label_map = {i: w for w, i in label_map.items()}
        
        config = BertConfig.from_json_file(os.path.join(args.pretrained_models_dir, args.bert_type, args.config_file))
        tmp_d = torch.load(os.path.join(args.pretrained_models_dir, args.bert_type, args.bert_model_file), map_location=device)
        state_dict = OrderedDict()
        for i in list(tmp_d.keys())[:199]:
            x = i
            if i.find('bert') > -1:
                x = '.'.join(i.split('.')[1:])
            state_dict[x] = tmp_d[i]
            
        model = BioBERTModel(args, 
                            config, 
                            state_dict).to(device)
        total_steps = len(train_loader) * args.epochs
        optimizer, scheduler = get_optimizer(model, 
                                            args.model_type, 
                                            lr=args.lr, 
                                            eps=1e-8, 
                                            total_steps=total_steps)

        loss_fn = nn.CrossEntropyLoss().to(device)

        learner = BioBERTLearner(args, 
                                model, 
                                optimizer, 
                                train_loader, 
                                valid_loader, 
                                loss_fn, 
                                device, 
                                scheduler)

        inference = BioBERTInference(args, 
                                    model, 
                                    test_loader, 
                                    label_map, 
                                    device) 

    elif args.model_type == 'bilstm':
        args.embedding_pos_path =  os.path.join(args.pretrained_models_dir, 'embedding_pos.pkl')
        args.embedding_char_path =  os.path.join(args.pretrained_models_dir, 'embedding_char.pkl')
        args.embedding_word_path =  os.path.join(args.pretrained_models_dir, 'embedding_word.pkl')
        args.pos2idx_path =  os.path.join(args.pretrained_models_dir, 'pos2idx.pkl')
        args.word2idx_path =  os.path.join(args.pretrained_models_dir, 'word2idx.pkl')
        args.path_pretrained_word_emb = os.path.join(args.pretrained_models_dir, 'word_embedding', args.pretrained_word_emb)

        tac_iterator = TACIterator(args, 
                                  TAC_train, 
                                  TAC_test, 
                                  TAC_valid, 
                                  device)
        train_iterator, valid_iterator, test_iterator = tac_iterator.get_data_iterator()
        TEXT = tac_iterator.TEXT
        POS = tac_iterator.POS
        CHAR = tac_iterator.CHAR
        TAG = tac_iterator.TAG
        args.word_matrix = TEXT.vocab.vectors
        args.pos_matrix = POS.vocab.vectors
        args.char_matrix = CHAR.vocab.vectors

        args.size_vocab_word = len(TEXT.vocab)
        args.size_vocab_pos = len(POS.vocab)

        args.num_labels = len(TAG.vocab)
        args.pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
        args.tag_pad_idx = TAG.vocab.stoi[TAG.pad_token]

        model = BiLSTMModel(args)
        model = model.apply(init_weights)
        model = model.to(device)

        optimizer, scheduler = get_optimizer(model, 
                                            args.model_type, 
                                            lr=args.lr, 
                                            eps=1e-6)
        
        loss_fn = nn.CrossEntropyLoss().to(device)
        learner = BiLSTMLearner(args, 
                                model, 
                                optimizer, 
                                train_iterator, 
                                valid_iterator, 
                                loss_fn, 
                                device, 
                                scheduler)
        inference = BiLSTMInference(args, 
                                    model, 
                                    test_iterator, 
                                    TAC_test, 
                                    TAG)


    if args.step == "train":
        os.makedirs(args.output_dir, exist_ok=True)
        for l in range(args.max_levels): 
            print(f'==================================================')
            print(f'Level: {l+1}')
            print(f'==================================================')
            learner.train(currentlevel=l+1)

    if args.step == "test":
        predicted_labels = []
        for l in range(args.max_levels): 
            print(f'Predicting level: {l+1} ...')
            model_save = args.model_type + '_model_level_' + str(l+1)+ '.bin'
            path = F"{args.output_dir}/{model_save}" 
            model.load_state_dict(torch.load(path, map_location=device))
            predicted_label = inference.predict(currentlevel=l+1)
            predicted_labels.append(predicted_label)
        print(f'Extract guess ADE mentions ...')
        dict_ade = extract_guess_ADE_mention(TAC_test, 
                                            predicted_labels, 
                                            gold_dir=gold_xml_dir_path)
        os.makedirs(guess_xml_dir_path, exist_ok=True)
        write_guess_xml_files(gold_xml_dir_path, 
                              guess_xml_dir_path, 
                              dict_ade)
        print(f'Done')
		
if __name__ == '__main__':
    main()