import os
from sys import path
path.append(os.getcwd())
from data_utils.complex_mention import filter_mention
from data_utils.preprocessing import replace_ponctuation_with_space, process, split_on_uppercase
from nltk.tokenize import word_tokenize

LABELSET = {'AdverseReaction': {'b': 'B-AdverseReaction', 'i': 'I-AdverseReaction', 'db': 'DB-AdverseReaction', 'di': 'DI-AdverseReaction'},
             'Severity': {'b': 'B-Severity', 'i': 'I-Severity', 'db': 'DB-Severity', 'di': 'DI-Severity'},
            'Negation': {'b': 'B-Negation', 'i': 'I-Negation', 'db': 'DB-Negation', 'di': 'DI-Negation'},
            'Factor': {'b': 'B-Factor', 'i': 'I-Factor', 'db': 'DB-Factor', 'di': 'DI-Factor'},
            'Animal': {'b': 'B-Animal', 'i': 'I-Animal', 'db': 'DB-Animal', 'di': 'DI-Animal'},
            'DrugClass': {'b': 'B-DrugClass', 'i': 'I-DrugClass', 'db': 'DB-DrugClass', 'di': 'DI-DrugClass'}}
			
def N_level_tags(set_ADE_mention, tok_text, entity_start, start, end, sentence, section):
    labels_set = []
    first_sequence, dis_con_overl_ment, con_overl_ment, dis_overl_ment = filter_mention(set_ADE_mention, start, end)
    if len(con_overl_ment)==0 and len(dis_overl_ment)==0 and len(dis_con_overl_ment)==0:
        sequence_1 = ['O'] * len(tok_text) 
        sequence_1 = one_level_tagging(first_sequence, sequence_1, tok_text, entity_start, start, end, sentence, section)
        labels_set.append(sequence_1)                    
    elif len(con_overl_ment)>=1 and len(dis_overl_ment)==0 and len(dis_con_overl_ment)==0:
        labels_set = N_level_tagging(first_sequence, con_overl_ment, tok_text, entity_start, start, end, sentence, section)
    elif len(con_overl_ment)==0 and len(dis_overl_ment)>=1 and len(dis_con_overl_ment)==0:
        labels_set = N_level_tagging(first_sequence, dis_overl_ment, tok_text, entity_start, start, end, sentence, section)
    elif len(con_overl_ment)==0 and len(dis_overl_ment)==0 and len(dis_con_overl_ment)>=1:
        labels_set = N_level_tagging(first_sequence, dis_con_overl_ment, tok_text, entity_start, start, end, sentence, section)
    elif len(con_overl_ment)>=1 and len(dis_overl_ment)==0 and len(dis_con_overl_ment)>=1:
        a = con_overl_ment+ dis_con_overl_ment
        labels_set = N_level_tagging(first_sequence, a, tok_text, entity_start, start, end, sentence, section)
    elif len(con_overl_ment)==0 and len(dis_overl_ment)>=1 and len(dis_con_overl_ment)>=1:
        a = dis_overl_ment+ dis_con_overl_ment
        labels_set = N_level_tagging(first_sequence, a, tok_text, entity_start, start, end, sentence, section)
    return labels_set

def one_level_tagging(set_ADE_mention, sequence_1, tok_text, entity_start, start, end, sentence, section):
    for m in set_ADE_mention:
        start_mention = m[1].split(',')
        len_mention = m[0].split(',')
        mstart = int(start_mention[0])                   
        mend = mstart + int(len_mention[0])  
        m_str = m[3]         
        if len(start_mention)==1: 
            sequence_1 = tagging_sequence(m_str, LABELSET[m[2]], tok_text, sequence_1, mstart, entity_start, 'simple', start, end, len_mention, sentence)                                
        else:  
            sequence_1 = tagging_sequence(section[mstart:mend], LABELSET[m[2]], tok_text, sequence_1, mstart, entity_start, 'complex', start, end, len_mention, sentence)
            for s, l in zip(start_mention[1:],len_mention[1:]):
                sequence_1 = tagging_sequence(section[int(s):int(s)+int(l)], LABELSET[m[2]], tok_text, sequence_1, int(s), entity_start, 'complex_follow', start, end, len_mention, sentence)
    return sequence_1
	
def tagging_sequence(string, labelset, tok_text, labels, mstart, entity_start, status, start, end, len_mention, ss):
    string = process(replace_ponctuation_with_space(string))
    tok_mention = []
    tok = split_on_uppercase(string, True)
    for t in tok:
        tok_mention.extend(word_tokenize(t))
                        
    mention_str_len = len(string.lstrip())
    index = int(len_mention[0]) - mention_str_len

    if mstart in entity_start:                            
        labels = tag_mention(tok_text, tok_mention, mstart, labelset, labels, entity_start, status)
    elif (mstart + index) in entity_start:
        labels = tag_mention(tok_text, tok_mention, mstart + index, labelset, labels, entity_start, status)
    elif (mstart + index) in range(start, end):
        labels = tag_mention(tok_text, tok_mention, ss.find(tok_mention[0]) + start, labelset, labels, entity_start, status)
    return labels

def N_level_tagging(first_sequence, con_overl_ment, tok_text, entity_start, start, end, sentence, section):
    labels_set = []
    max_level = max([len(x) for x in con_overl_ment])
    for row in con_overl_ment:
        if len(row)<max_level:
            row.extend([row[0]]*(max_level-len(row)))
    for ment in list(zip(*con_overl_ment)):
        sequence_1 = ['O'] * len(tok_text) 
        sequence_1 = one_level_tagging(first_sequence, sequence_1, tok_text, entity_start, start, end, sentence, section)
        sequence_1 = one_level_tagging(ment, sequence_1, tok_text, entity_start, start, end, sentence, section)
        labels_set.append(sequence_1)
    return labels_set

	
def tag_mention(tok_text, tok_entity, begin, labelset, labels, entity_start, status):
    for i, (word, start) in enumerate(zip(tok_text, entity_start)):
        if start == begin:
            if status == 'simple': 
                if len(tok_entity) == 1:
                    labels[i] = labelset['b'] 
                else:
                    labels[i] = labelset['b'] 
                    labels[i+1:i+len(tok_entity)-1] = [labelset['i']] * (len(tok_entity)-2)
                    labels[i+len(tok_entity)-1] = labelset['i'] 
            elif status == 'complex': 
                if len(tok_entity) == 1:
                    labels[i] = labelset['db']
                else:
                    labels[i] = labelset['db'] 
                    labels[i+1:i+len(tok_entity)-1] = [labelset['di']] * (len(tok_entity)-2)
                    labels[i+len(tok_entity)-1] = labelset['di'] 
            elif status == 'complex_follow': 
                if len(tok_entity) == 1:
                    labels[i] = labelset['di'] 
                else:
                    labels[i] = labelset['di'] 
                    labels[i+1:i+len(tok_entity)-1] = [labelset['di']] * (len(tok_entity)-2)
                    labels[i+len(tok_entity)-1] = labelset['di']                   
    return labels
