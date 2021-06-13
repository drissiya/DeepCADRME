import os
from data_utils.drug_label import xml_files
#from utils import *

def extract_mention_from_sentences(drug, set_toks, ys_bio, section, start, leng):
    data_drug = []
    data_toks = []
    data_ys = []
    data_sec = []
    data_start = []
    data_len = []
    for d, tok, ys, sec, st, le in zip(drug, set_toks, ys_bio, section, start, leng):
        temp_toks = []
        temp_ys = []
        temp_sec = []
        temp_start = []
        temp_len = []
        temp_drug = []
        for i, (t, yb, s, a, b, e) in enumerate(zip(tok, ys, sec, st, le, d)):          
            if yb.startswith('B-'):
                tok_txt = t
                ys_txt = yb[2:]
                sec_txt = s
                start_txt = a
                len_text = b
                drug_text = e
                if (i+1) == len(ys):
                    temp_toks.append(t)
                    temp_ys.append(yb[2:])
                    temp_sec.append(s)
                    temp_start.append(a)
                    temp_len.append(b)
                    temp_drug.append(e)
                    break
                elif ys[i+1].startswith('O') and ys[i-1].startswith('O'):
                    temp_toks.append(t)
                    temp_ys.append(yb[2:])
                    temp_sec.append(s)
                    temp_start.append(a)
                    temp_len.append(b)
                    temp_drug.append(e)
                elif ys[i+1].startswith('B-') and ys[i-1].startswith('B-'):
                    temp_toks.append(t)
                    temp_ys.append(yb[2:])
                    temp_sec.append(s)
                    temp_start.append(a)
                    temp_len.append(b)
                    temp_drug.append(e)
                elif ys[i+1].startswith('DB-') and ys[i-1].startswith('DB-'):
                    temp_toks.append(t)
                    temp_ys.append(yb[2:])
                    temp_sec.append(s)
                    temp_start.append(a)
                    temp_len.append(b)
                    temp_drug.append(e)
                elif ys[i+1].startswith('DI-') and ys[i-1].startswith('DI-'):
                    temp_toks.append(t)
                    temp_ys.append(yb[2:])
                    temp_sec.append(s)
                    temp_start.append(a)
                    temp_len.append(b)
                    temp_drug.append(e)
                elif ys[i+1].startswith('DB-') and ys[i-1].startswith('DI-'):
                    temp_toks.append(t)
                    temp_ys.append(yb[2:])
                    temp_sec.append(s)
                    temp_start.append(a)
                    temp_len.append(b)
                    temp_drug.append(e)
                elif ys[i+1].startswith('DI-') and ys[i-1].startswith('DB-'):
                    temp_toks.append(t)
                    temp_ys.append(yb[2:])
                    temp_sec.append(s)
                    temp_start.append(a)
                    temp_len.append(b)
                    temp_drug.append(e)
                else: 
                    start_n = a
                    for k,j in enumerate(ys[i+1:]):
                        if j.startswith('I-'):
                            tok_txt += ' ' + tok[i+k+1]
                            len_text = le[i+k+1] 
                            start_n = st[i+k+1]
                            
                        else:
                            break
                    len_t = (start_n-start_txt)+len_text
                    temp_toks.append(tok_txt)
                    temp_ys.append(ys_txt)
                    temp_sec.append(sec_txt)
                    temp_start.append(start_txt)
                    temp_len.append(len_t)
                    temp_drug.append(drug_text)
            elif yb.startswith('DB-'):
                tok_txt = t
                ys_txt = yb[3:]
                sec_txt = s
                start_txt = a
                start_t = str(a)
                len_text = b
                drug_text = e
                l = i
                len_text_list = []
                for k,j in enumerate(ys[i+1:]):
                    if j.startswith('DI-'):
                        len_text += le[i+k+1] + 1
                        if (i+k+2) == len(ys):
                            len_text_list.append(len_text)
                    elif j.startswith('DB-'):
                        len_text_list.append(len_text)
                        break
                    else:
                        len_text_list.append(len_text)
                        len_text = 0
                if len(len_text_list)!=0:
                    len_t = str(len_text_list[0])
                    for m in len_text_list[1:]:
                        if m!=0:
                            len_t += ',' + str(m-1)

                    for k,j in enumerate(ys[i+1:]):
                        if j.startswith('DI-'):
                            tok_txt += ' ' + tok[i+k+1]                      
                            if ys[l].startswith('B-') or ys[l].startswith('I-') or ys[l].startswith('O'):
                                start_t += ',' + str(st[i+k+1])                       
                       
                        elif j.startswith('DB-'):
                            break
                        l = l + 1
                    temp_toks.append(tok_txt)
                    temp_ys.append(ys_txt)
                    temp_sec.append(sec_txt)
                    temp_start.append(start_t)
                    temp_len.append(len_t)
                    temp_drug.append(drug_text)
        data_toks.append(temp_toks)
        data_ys.append(temp_ys)
        data_sec.append(temp_sec)
        data_start.append(temp_start)
        data_len.append(temp_len)
        data_drug.append(temp_drug)
    return data_drug, data_toks, data_ys, data_sec, data_start, data_len

def extract_guess_ADE_mention(TAC_guess, labels, gold_dir):   
    drug_mention = []
    toks_mention = []
    type_mention = []
    sec_mention = []
    start_mention = []
    len_mention = []
    dict_ade = {}
    for l in labels:
        drug, toks, type_, sec, start, len_ = extract_mention_from_sentences(TAC_guess.t_drug_mention, TAC_guess.t_toks_mention, l, TAC_guess.t_section_mention, TAC_guess.t_start_mention, TAC_guess.t_len_mention)
        drug_mention.extend(drug)
        toks_mention.extend(toks)
        type_mention.extend(type_)
        sec_mention.extend(sec)
        start_mention.extend(start)
        len_mention.extend(len_)
    
    guess_files = xml_files(gold_dir)
    for key, value in zip(guess_files.keys(), guess_files.values()):
        ADE_mention = []
        duplicate_ade = set()
        id_mention = 1
        for drug_m, toks_m, type_m, sec_m, start_m, len_m in zip(drug_mention, toks_mention, type_mention, sec_mention, start_mention, len_mention):
            if len(set(drug_m))==0:
                continue
            if list(set(drug_m))[0]==key:
                for tok, typ, sec, st, le in zip(toks_m, type_m, sec_m, start_m, len_m):
                    if (str(le), str(st), typ , tok, sec) not in duplicate_ade:
                        ade = (str(le), str(st), typ , tok, sec, "M"+str(id_mention))
                        ADE_mention.append(ade)
                        id_mention+=1 
                        duplicate_ade.add((str(le), str(st), typ , tok, sec))
                        
        #print (duplicate_ade)
        dict_ade[key] = ADE_mention
    return dict_ade
