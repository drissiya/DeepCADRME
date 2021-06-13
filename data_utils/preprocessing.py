import re
from nltk.tokenize import word_tokenize

def replace_ponctuation_with_space(s):
    #s = re.sub(r"[^\w(),|!?\'\`\:\-\.;\$%#]", " ", s)
    s = re.sub(r"-", " ", s)
    s = re.sub(r"/", " ", s)
    s = re.sub(r"\*", " ", s)
    s = re.sub(r"'", " ", s)
    return s

def process(s):
    s = re.sub(r"[^\w(),|!?\'\`\:\-\.;\$%#]", " ", s)
    #s = re.sub(r"-", " ", s)
    return s

def split_on_uppercase(s, keep_contiguous=False):
    string_length = len(s)
    is_lower_around = (lambda: s[i-1].islower() or 
                       string_length > (i + 1) and s[i + 1].islower())

    start = 0
    parts = []
    for i in range(1, string_length):
        if s[i].isupper() and (not keep_contiguous or is_lower_around()):
            parts.append(s[start: i])
            start = i
    parts.append(s[start:])
    return parts
	
def spans(txt, tokens, start):
    entity_start = []
    entity_end = []
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset) 
        entity_start.append(offset + start)
        entity_end.append(len(token))
        offset += len(token)
    return entity_start, entity_end

def tokenize_sentence(sentence):
    """
    Arguments:
        sentence: string of texts
    Outputs:
        tok_text: list of tokens
    """
    tok_text = []
    tok = split_on_uppercase(sentence, True)
    for t in tok:
        tok_text.extend(word_tokenize(t))
    return tok_text