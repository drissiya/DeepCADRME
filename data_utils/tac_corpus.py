import os
from nltk.tokenize import word_tokenize
from sys import path
path.append(os.getcwd())
from data_utils.drug_label import xml_files, read
from data_utils.preprocessing import process, replace_ponctuation_with_space, spans, tokenize_sentence
from data_utils.complex_mention import get_section, get_mentions, get_mentions_from_sentence
from data_utils.tagging import N_level_tags
from nltk import pos_tag
from xml.etree import ElementTree
from sklearn.model_selection import ShuffleSplit

def load_dir(dir_name_labels, dir_name_sentences):
    """
    Args:
        dir_name_labels: directory that contains all xml files with Section, Mention and Relation elements.
        dir_name_sentences: directory that contains all xml files with Section and Sentence elements.
    """
    ADE_mention_data = []
    files = xml_files(dir_name_labels)
    for key, value in zip(files.keys(), files.values()):
        label = read(value)
        sections = label.sections
        mentions = label.mentions

        root = ElementTree.parse(os.path.join(dir_name_sentences, key + '.xml')).getroot() 
        assert root.tag == 'Label', 'Root is not Label: ' + root.tag
        assert root[0].tag == 'Section', 'Expected \'Text\': ' + root[0].tag

        for sec in root: 
            #Return id_section and section
            id_section, section = get_section(sections, sec)
            
            #Return mentions for a given id_section            
            unique_section_mentions, section_mentions = get_mentions(mentions, id_section)
            
            
            #Return sentence from the offset.
            for sent in sec:
                start = int(sent.attrib['start'])
                end = int(sent.attrib['len'])
                
                #Ignore empty sentence
                sentence = section[start:end]
                if len(process(replace_ponctuation_with_space(sentence)).strip()) == 0:
                    continue
                
                #Tokenize sentence
                sentence = process(replace_ponctuation_with_space(sentence))
                tok_text = tokenize_sentence(sentence)
                
                
                set_ADE_mention = get_mentions_from_sentence(unique_section_mentions, start, end)
                #Map each token to its corresponding section, start and len
                entity_section = [id_section]*len(tok_text)
                entity_drug = [key]*len(tok_text)
                entity_start, entity_end = spans(sentence, tok_text, start)

                if len(set_ADE_mention)>0:
                    labels_set = N_level_tags(set_ADE_mention, tok_text, entity_start, start, end, sentence, section)
                    ADE_mention_data.append((tok_text,labels_set,entity_start,entity_section,entity_end, entity_drug, sentence))
    return ADE_mention_data

def split_data(sentences_train):
    TAC_dev = TAC()
    TAC_train_temp = TAC()
    
    rs = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_index, test_index in rs.split(sentences_train.t_toks_mention):
        TAC_dev.t_toks_mention = [sentences_train.t_toks_mention[i] for i in test_index]
        TAC_dev.t_segment_mention = [sentences_train.t_segment_mention[i] for i in test_index]
        TAC_dev.t_start_mention = [sentences_train.t_start_mention[i] for i in test_index]
        TAC_dev.t_section_mention = [sentences_train.t_section_mention[i] for i in test_index]
        TAC_dev.t_len_mention = [sentences_train.t_len_mention[i] for i in test_index]
        TAC_dev.t_drug_mention = [sentences_train.t_drug_mention[i] for i in test_index]
        TAC_dev.t_sentence_input_mention = [sentences_train.t_sentence_input_mention[i] for i in test_index]
        TAC_dev.t_pos = [sentences_train.t_pos[i] for i in test_index]
        
        TAC_train_temp.t_toks_mention = [sentences_train.t_toks_mention[i] for i in train_index]
        TAC_train_temp.t_segment_mention = [sentences_train.t_segment_mention[i] for i in train_index]
        TAC_train_temp.t_start_mention = [sentences_train.t_start_mention[i] for i in train_index]
        TAC_train_temp.t_section_mention = [sentences_train.t_section_mention[i] for i in train_index]
        TAC_train_temp.t_len_mention = [sentences_train.t_len_mention[i] for i in train_index]
        TAC_train_temp.t_drug_mention = [sentences_train.t_drug_mention[i] for i in train_index]
        TAC_train_temp.t_sentence_input_mention = [sentences_train.t_sentence_input_mention[i] for i in train_index]
        TAC_train_temp.t_pos = [sentences_train.t_pos[i] for i in train_index]
        
    return TAC_train_temp, TAC_dev 	
	
def append_data(data_y, max_level):
    data_y_final = []
    for labels in data_y:
        y= []
        y.extend(labels)
        if len(y)!=max_level:
            for i in range(len(y),max_level):
                y.append(labels[0])
        data_y_final.append(y)
    return data_y_final


def convert_data(data_y, max_level):
    data_y_final = []
    for i in range(max_level):
        y = []
        for labels in data_y:
            y.append(labels[i])
        data_y_final.append(y)  
    return data_y_final


class TAC:
    def __init__(self, max_level=3, sentence_dir=None, label_dir=None):
        self.sentence_dir = sentence_dir
        self.label_dir = label_dir
        self.max_level = max_level        
        self.ade_mention_data = []
        self.t_drug_mention = []
        self.t_toks_mention = []
        self.t_segment_mention = []
        self.t_section_mention = []
        self.t_start_mention = []
        self.t_len_mention = []
        self.t_sentence_input_mention = []
        self.t_pos = []
             

    def load_corpus(self):
        ADE_mention_data = load_dir(self.label_dir, self.sentence_dir)
        self.ade_mention_data = ADE_mention_data
        #self.preproc()
            
        for seq in self.ade_mention_data:
            self.t_toks_mention.append(seq[0])
            self.t_segment_mention.append(seq[1])
            self.t_start_mention.append(seq[2])
            self.t_section_mention.append(seq[3])
            self.t_len_mention.append(seq[4])
            self.t_drug_mention.append(seq[5])
            self.t_sentence_input_mention.append(seq[6])
            pos_temp = pos_tag(seq[0])
            pos = []
            for p in pos_temp:
                pos.append(p[1])
            self.t_pos.append(pos)
			
        self.t_segment_mention = append_data(self.t_segment_mention, self.max_level)			
            	
