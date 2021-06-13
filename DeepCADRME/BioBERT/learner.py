import torch
#from collections import defaultdict, OrderedDict
import numpy as np

class BioBERTLearner(object):
    def __init__(self, args, model, optimizer, train_loader, valid_loader, loss_fn, device, scheduler):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, currentlevel):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0
        for batch in self.train_loader:
            batch = tuple(t.to(self.device) for t in batch)
            self.optimizer.zero_grad()

            input_ids, input_mask, segment_ids, level1, level2, level3 = batch
            output = self.model(input_ids, segment_ids, input_mask)     
                
            _,preds = torch.max(output,dim=2)
            output = output.view(-1,output.shape[-1])  
       
            if currentlevel == 1:
                level = level1
            if currentlevel == 2:
                level = level2
            if currentlevel == 3:
                level = level3               
            b_labels_shaped = level.view(-1)

            loss = self.loss_fn(output,b_labels_shaped)
            correct_predictions += torch.sum(preds == level)
            losses.append(loss.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        return correct_predictions.double()/len(self.train_loader), np.mean(losses)

    def train(self, currentlevel):
        t_loss = []
        t_acc = []
        v_loss = []
        v_acc = []
        best_valid_loss = float('inf')
        normalizer = self.args.batch_size*self.args.max_seq_length
        for epoch in range(self.args.epochs): 
            print(f'\tEpoch: {epoch+1:02}')          
            train_acc, train_loss = self.train_epoch(currentlevel)
            train_acc = train_acc/normalizer
            t_loss.append(train_loss)
            t_acc.append(train_acc) 
                        
            val_acc, val_loss = self.evaluate(currentlevel)
            val_acc = val_acc/normalizer
            v_loss.append(val_loss)
            v_acc.append(val_acc)

            self.scheduler.step()

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                model_save = self.args.model_type + '_model_level_' + str(currentlevel)+ '.bin'
                path = F"{self.args.output_dir}/{model_save}" 
                torch.save(self.model.state_dict(), path)

            print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc:.2f}')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. Accuracy: {val_acc:.2f}')

    def evaluate(self, currentlevel):
        self.model = self.model.eval()       
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for step, batch in enumerate(self.valid_loader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, level1, level2, level3 = batch
                output = self.model(input_ids, segment_ids, input_mask)   
              
                _,preds = torch.max(output,dim=2)
                output = output.view(-1,output.shape[-1])  

                if currentlevel == 1:
                    level = level1
                elif currentlevel == 2:
                    level = level2
                elif currentlevel == 3:
                    level = level3

                b_labels_shaped = level.view(-1)

                loss = self.loss_fn(output,b_labels_shaped)
                correct_predictions += torch.sum(preds == level)

                losses.append(loss.item())
                
        return correct_predictions.double()/len(self.valid_loader), np.mean(losses)