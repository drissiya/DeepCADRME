# DeepCADRME: A deep neural model for complex adverse drug reaction mentions extraction
DeepCADRME is a deep neural model for extracting complex adverse drug reaction (ADR) mentions (simple, nested, discontinuous and overlapping). It first transforms the ADR mentions extraction problem as an N-level tagging sequence. Then, it feeds the sequences to an N-level model based on contextual embeddings where the output of the pre-trained model of the current level is used to build a new deep contextualized representation for the next level. This allows the DeepCADRME system to transfer knowledge between levels.

## Requirements

1. Download the pre-trained model files used for DeepCADRME experiments. They include BERT (trained on general domain text), BioBERT (trained on PubMed), BioBERT (trained on PMC), BioBERT (trained on PubMed and PMC), pre-trained word embedding (trained on PubMed), pre-trained word embedding (trained on PMC), pre-trained word embedding (trained on PubMed and PMC) and pre-trained word embedding (trained on PubMed, PMC and Wikipedia). To do so, just run the following command:
```
$ bash pretrained_models/download.sh
```
2. Install all dependencies needed to run this code using the command:
```
$ pip install -r requirements.txt
```
3. Please use the NLTK Downloader to obtain the punkt and averaged_perceptron_tagger resources used for tokenization and part-of-speech tagging, respectively:
```
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Quick start

1. To train the DeepCADRME system, you can run the main.py file:
```
$ python main.py --model-type "biobert" --step "train"
```
- model-type: specifies the model type used for initialization phase. It accepts two values: biobert (for BERT-based models) and bilstm (for baseline models).
- step: accepts two values: train (for training step) and test (for test step).

After the program is finished, the model weights for each level will be saved in:
```
checkpoint/{model-type}_model_level_1.bin
checkpoint/{model-type}_model_level_2.bin
checkpoint/{model-type}_model_level_3.bin
```

2. To test the DeepCADRME system, just run the following command:
```
$ python main.py --model-type "biobert" --step "test"
```
After the program is finished, the guess xml files will be generated in the TAC/guess_xml folder.

3. To evaluate DeepCADRME, run evaluate.py file which includes the official script for TAC 2017 ADR evaluation:
```
$ python evaluate.py "TAC/gold_xml" "TAC/guess_xml"
```

## Citation 

```
@article{El_allaly_2021,
	doi = {10.1016/j.patrec.2020.12.013},
	year = 2021,
	month = {mar},
	publisher = {Elsevier {BV}},
	volume = {143},
	pages = {27--35},
	author = {Ed-drissiya El-allaly and Mourad Sarrouti and Noureddine En-Nahnahi and Said Ouatik El Alaoui},
	title = {{DeepCADRME}: A deep neural model for complex adverse drug reaction mentions extraction},
	journal = {Pattern Recognition Letters}
}
```

## Acknowledgements

Thanks to the TAC 2017 ADR challenge organizers who provided us the gold set used to evaluate this work.

