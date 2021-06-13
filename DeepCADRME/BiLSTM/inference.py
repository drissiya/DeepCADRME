from data_utils.tac_corpus import convert_data
import torch

def trim_seq_bilstm(predicted_labels, TAC_test, currentlevel, max_levels):
    predicted_labels_trimed = []
    tag = convert_data(TAC_test.t_segment_mention, max_levels)
    for i, j in zip(predicted_labels, tag[currentlevel-1]):
      pr = i[:len(j)]
      predicted_labels_trimed.append(pr)
    return predicted_labels_trimed

class BiLSTMInference(object):
    def __init__(self, args, model, test_iterator, TAC_test, label_map):
        self.args = args
        self.model = model
        self.test_iterator = test_iterator
        self.TAC_test = TAC_test
        self.label_map = label_map

    def predict(self, currentlevel):    
        self.model.eval()
        predicted_labels = []
        with torch.no_grad():       
            for batch in self.test_iterator:
                text = batch.text
                pos = batch.pos
                char = batch.char

                if currentlevel == 1:
                    tags = batch.level1
                if currentlevel == 2:
                    tags = batch.level2
                if currentlevel == 3:
                    tags = batch.level3
                
                predictions = self.model(text, pos, char)

                _,preds = torch.max(predictions,dim=2)
                preds = preds.transpose(1,0)
              
                for p in preds:
                    predicted_tags = [self.label_map.vocab.itos[t.item()] for t in p]
                    predicted_labels.append(predicted_tags)

        predicted_labels = trim_seq_bilstm(predicted_labels, self.TAC_test, currentlevel, self.args.max_levels)
        return predicted_labels