
def get_section(section_label, section_sentence):
    """
    Arguments:
        section_label: all sections for a given label.
        section_sentence: all sections with boundaries sentences
        
    Outputs:
        id_section: id of section
        section: text of section
    """
    id_section = None
    section = None
    for s in section_label:
        if s.name == section_sentence.attrib['name']:
            id_section = s.id
            section = s.text
            break
    return id_section, section
	
def get_mentions(mentions, id_section):
    """
    Arguments:
        mentions: all mentions for given label
        id_section: id of section
        
    Output:
        unique_section_mentions: set of mentions for given id_section
        section_mentions: list of mentions for given section
    """
    section_mentions = []
    unique_section_mentions = set()
    for m in mentions:
        if m.section==id_section:
            section_mentions.append(m) 
                    
    #Return unique mentions
    for m in section_mentions:
        mention = (m.len, m.start, m.type, m.str, m.section)
        unique_section_mentions.add(mention)
    return unique_section_mentions, section_mentions

def get_mentions_from_sentence(section_mentions, start, end):
    set_ADE_mention = []
    for m in section_mentions:
        start_mention = m[1].split(',')         
        mstart = int(start_mention[0])                   
        if mstart in range(start, end):
            set_ADE_mention.append((int(m[0].split(',')[0]), (m[0], m[1], m[2], m[3], m[4])))
    return set_ADE_mention
	
def filter_mention(set_ADE_mention, start, end):
    first_sequence = []
    dis_con_overl_ment = []
    con_overl_ment = []
    dis_overl_ment = []
    set_ADE_mention = sorted(set_ADE_mention, key=lambda tup: tup[0], reverse=True)
    for z, m in set_ADE_mention:
        start_mention = m[1].split(',')
        len_mention = m[0].split(',')
        mstart = int(start_mention[0])                   
        mend = mstart + int(len_mention[0])  
                        
        if len(start_mention)==1: 
            contin_m, discontin_m = return_overlapping_mentions_per_mention1(mstart, mend, set_ADE_mention)
            set_ADE_mention = [(y, x) for y, x in set_ADE_mention if x not in set(discontin_m)]
            set_ADE_mention = [(y, x) for y, x in set_ADE_mention if x not in set(contin_m)]
            
            if len(set(discontin_m))>=1 and len(set(contin_m))>=1:
                dis_con_overl_ment.append(list(set(discontin_m).union(set(contin_m))))                       
            elif len(set(discontin_m))==0 and len(set(contin_m))>=2: 
                con_overl_ment.append(list(set(contin_m))) 
            elif len(set(discontin_m))==0 and len(set(contin_m))<=1:
                first_sequence.extend(contin_m)
                
        else:  
            contin_ms = [] 
            discontin_ms = []
            for s, l in zip(start_mention,len_mention):                          
                contin_m, discontin_m = return_overlapping_mentions_per_mention1(int(s),int(s)+int(l), set_ADE_mention)
                contin_ms.extend(contin_m)
                discontin_ms.extend(discontin_m)
                            
            set_ADE_mention = [(y, x) for y, x in set_ADE_mention if x not in set(discontin_ms)]
            set_ADE_mention = [(y, x) for y, x in set_ADE_mention if x not in set(contin_ms)]
            
            if len(set(discontin_ms))>=1 and len(set(contin_ms))>=1:
                dis_con_overl_ment.append(list(set(discontin_ms).union(set(contin_ms))))        
            elif len(set(discontin_ms))==0 and len(set(contin_ms))>=2:
                con_overl_ment.append(list(set(contin_ms)))
            elif len(set(discontin_ms))>=2 and len(set(contin_ms))==0:
                dis_overl_ment.append(list(set(discontin_ms)))
            elif len(set(discontin_ms))<=1 and len(set(contin_ms))==0:
                first_sequence.extend(discontin_ms)
                
                
    return first_sequence, dis_con_overl_ment, con_overl_ment, dis_overl_ment

def return_overlapping_mentions_per_mention1(start_m, end_m, modifier_set):
    contin_m = []
    discontin_m = []
    for z, m in modifier_set:
        start_mention = m[1].split(',')
        len_mention = m[0].split(',')       
        mstart = int(start_mention[0])                   
        mend = mstart + int(len_mention[0])
        
        if len(start_mention)==1:
            if mstart in range(start_m,end_m) or start_m in range(mstart,mend):
                contin_m.append(m)
        else:
            for b, e in zip(start_mention, len_mention):
                if int(b) in range(start_m,end_m) or start_m in range(int(b), int(b) + int(e)):
                    discontin_m.append(m)
    return contin_m, discontin_m