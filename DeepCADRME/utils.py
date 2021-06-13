from transformers import AdamW, get_linear_schedule_with_warmup
from data_utils.drug_label import xml_files
from xml.etree import ElementTree
import torch.optim as optim
import torch.nn as nn
import os

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)
        

def get_optimizer(model, model_type, lr, eps, total_steps=0):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}]
    if model_type=='bilstm':
        optimizer = optim.Adam(
            optimizer_grouped_parameters,
            lr=lr,
            eps=eps
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    elif model_type=='biobert':
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            eps=eps
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
    return optimizer, scheduler
	
def write_guess_xml_files(gold_xml_dir, guess_xml_dir, dict_ade):
    guess_files = xml_files(gold_xml_dir)
    for key, value in zip(guess_files.keys(), guess_files.values()):
        root = ElementTree.parse(value).getroot()
        root.remove(root[1])
        root.remove(root[2])
        root.remove(root[-1])
        Mentions = ElementTree.SubElement(root, "Mentions")
        for m in dict_ade[key]:
            ElementTree.SubElement(Mentions, "Mention", id=m[5], section=m[4], type=m[2], start=m[1], len=m[0], str=m[3])
            
        Relations = ElementTree.SubElement(root, "Relations")    
        Reactions = ElementTree.SubElement(root, "Reactions")

        tree = ElementTree.ElementTree(root)
        tree.write(os.path.join(guess_xml_dir, key + '.xml'), encoding="utf-8") 

