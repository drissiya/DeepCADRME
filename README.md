# DeepCADRME: A deep neural model for complex adverse drug reaction mentions extraction
DeepCADRME is a deep neural model for extracting complex adverse drug reaction (ADR) mentions (simple, nested, discontinuous and overlapping). It first transforms the ADR mentions extraction problem as an N-level tagging sequence. Then, it feeds the sequences to an N-level model based on contextual embeddings where the output of the pre-trained model of the current level is used to build a new deep contextualized representation for the next level. This allows the DeepCADRME system to transfer knowledge between levels.

## Requirements

1. Download the pre-trained model files used for DeepCADRME experiments. They include BERT (trained on general domain text), BioBERT (trained on PubMed), BioBERT (trained on PMC), BioBERT (trained on PubMed and PMC), pre-trained word embedding (trained on PubMed), pre-trained word embedding (trained on PMC), pre-trained word embedding (trained on PubMed and PMC) and pre-trained word embedding (trained on PubMed, PMC and Wikipedia). To do so, just run the following command:
```
$ bash pretrained_models/download.sh
```
2. Install all dependencies automatically using the command:
```
$ pip install -r requirements.txt
```
3. Please use the NLTK Downloader to obtain the punkt and averaged_perceptron_tagger resources used for tokenization and part-of-speech tagging, respectively:
```
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```


