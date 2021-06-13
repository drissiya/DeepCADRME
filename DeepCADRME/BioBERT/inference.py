import torch

def trim_sequence(prediction, true_set, label_mappers):
    predict_lines = []
    for pred, true in zip(prediction, true_set):
        p_label = []
        for p, t in zip(pred, true):
            if p == 0: 
                l = 'O'
                p_label.append(l)
                continue
            l= label_mappers[p]
            tr= label_mappers[t]
            if tr == 'X': continue
            if l == '[CLS]': continue
            if l == '[SEP]': continue
            if l == 'X': l = 'O'
            p_label.append(l)
        predict_lines.append(p_label)
    return predict_lines
	
def trim(level, preds, valied_lenght):
    final_predict = []
    target = []
    l1 = level.tolist()
    p1 = preds.tolist()
    for idx, (p, t) in enumerate(zip(p1, l1)):
        final_predict.append(p[: valied_lenght[idx]])
        target.append(t[: valied_lenght[idx]])
    return final_predict, target
	
class BioBERTInference(object):
    def __init__(self, args, model, test_loader, label_map, device):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.label_map = label_map

    def predict(self, currentlevel):
        self.model = self.model.eval()
        
        predicted_labels = []
        target_labels = []

        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, level1, level2, level3 = batch
                output = self.model(input_ids, segment_ids, input_mask) 
                valied_lenght = input_mask.sum(1).tolist()   
              
                _,preds = torch.max(output,dim=2)               
                if currentlevel == 1:
                    level = level1
                if currentlevel == 2:
                    level = level2
                if currentlevel == 3:
                    level = level3

                final_predict, target = trim(level, preds, valied_lenght)

                predicted_labels.extend(final_predict)  
                target_labels.extend(target)   

            pred_labels = trim_sequence(predicted_labels, target_labels, self.label_map)
        return pred_labels