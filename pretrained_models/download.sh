#!/usr/bin/env bash
############################################################## 
# This script is used to download resources for DeepCADRME experiments
############################################################## 

cd "pretrained_models"
mkdir "biobert_trained_on_pubmed"
wget https://huggingface.co/monologg/biobert_v1.1_pubmed/resolve/main/config.json -O "biobert_trained_on_pubmed/config.json"
wget https://huggingface.co/monologg/biobert_v1.1_pubmed/resolve/main/pytorch_model.bin -O "biobert_trained_on_pubmed/pytorch_model.bin"
wget https://huggingface.co/monologg/biobert_v1.1_pubmed/resolve/main/vocab.txt -O "biobert_trained_on_pubmed/vocab.txt"

cd "pretrained_models"
mkdir "biobert_trained_on_pubmed_pmc"
wget https://huggingface.co/monologg/biobert_v1.0_pubmed_pmc/resolve/main/pytorch_model.bin -O "biobert_trained_on_pubmed_pmc/pytorch_model.bin"
wget https://huggingface.co/monologg/biobert_v1.0_pubmed_pmc/resolve/main/config.json -O "biobert_trained_on_pubmed_pmc/config.json"
wget https://huggingface.co/monologg/biobert_v1.0_pubmed_pmc/resolve/main/vocab.txt -O "biobert_trained_on_pubmed_pmc/vocab.txt"

cd "pretrained_models"
mkdir "bert"
wget https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin -O "bert/pytorch_model.bin"
wget https://huggingface.co/bert-base-cased/resolve/main/config.json -O "bert/config.json"
wget https://huggingface.co/bert-base-cased/resolve/main/vocab.txt -O "bert/vocab.txt"

cd "pretrained_models"
mkdir "biobert_trained_on_pmc"
wget https://huggingface.co/xinzhi/biobert_v1.0_pmc/resolve/main/biobert_v1.0_pmc.pytorch.bin -O "biobert_trained_on_pmc/pytorch_model.bin"
wget https://huggingface.co/xinzhi/biobert_v1.0_pmc/resolve/main/config.json -O "biobert_trained_on_pmc/config.json"
wget https://huggingface.co/xinzhi/biobert_v1.0_pmc/resolve/main/vocab.txt -O "biobert_trained_on_pmc/vocab.txt"

cd "pretrained_models"
mkdir "word_embedding"
wget http://evexdb.org/pmresources/vec-space-models/PMC-w2v.bin -O "word_embedding/PMC-w2v.bin"
wget http://evexdb.org/pmresources/vec-space-models/PubMed-and-PMC-w2v.bin -O "word_embedding/PubMed-and-PMC-w2v.bin"
wget http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin -O "word_embedding/PubMed-w2v.bin"
wget http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin -O "word_embedding/wikipedia-pubmed-and-PMC-w2v.bin"