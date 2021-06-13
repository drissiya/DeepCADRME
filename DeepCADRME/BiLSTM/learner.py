import torch
import numpy as np

class BiLSTMLearner(object):
    def __init__(self, 
                 args, 
                 model, 
                 optimizer, 
                 train_iterator, 
                 valid_iterator, 
                 loss_fn, 
                 device, 
                 scheduler):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, currentlevel):        
        epoch_loss = 0
        correct_predictions = 0       
        self.model.train()  

        for batch in self.train_iterator:      
            text = batch.text
            pos = batch.pos
            char = batch.char

            if currentlevel == 1:
                tags = batch.level1
            if currentlevel == 2:
                tags = batch.level2
            if currentlevel == 3:
                tags = batch.level3

            self.optimizer.zero_grad()                 
            predictions = self.model(text, pos, char)
            _, preds = torch.max(predictions, 2)

            predictions = predictions.view(-1, predictions.shape[-1])
            
            targets = tags.view(-1)
            
            loss = self.loss_fn(predictions, targets)
            
            correct_predictions += torch.sum(preds == tags)
            
            loss.backward()          
            self.optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(self.train_iterator), correct_predictions.double() / len(self.train_iterator)

    def train(self, currentlevel):
        t_loss = []
        t_acc = []
        v_loss = []
        v_acc = []

        best_valid_loss = float('inf')
        normalizer = self.args.batch_size*self.args.max_seq_length

        for epoch in range(self.args.epochs):      
            print(f'\tEpoch: {epoch+1:02}')    
            train_loss, train_acc = self.train_epoch(currentlevel)
            train_acc = train_acc/normalizer
            t_loss.append(train_loss)
            t_acc.append(train_acc) 
            
            valid_loss, val_acc = self.evaluate(currentlevel)
            val_acc = val_acc/normalizer
            v_loss.append(valid_loss)
            v_acc.append(val_acc)
            
            self.scheduler.step()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_save = self.args.model_type + '_model_level_' + str(currentlevel)+ '.bin'
                path = F"{self.args.output_dir}/{model_save}" 
                torch.save(self.model.state_dict(), path)

            print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc:.2f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Accuracy: {val_acc:.2f}')

    def evaluate(self, currentlevel):      
        epoch_loss = 0
        correct_predictions = 0
        
        self.model.eval()
        with torch.no_grad():       
            for batch in self.valid_iterator:
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
                _, preds = torch.max(predictions, 2)

                predictions = predictions.view(-1, predictions.shape[-1])

                targets = tags.view(-1)
                
                loss = self.loss_fn(predictions, targets)
    
                correct_predictions += torch.sum(preds == tags)

                epoch_loss += loss.item()
            
        return epoch_loss/len(self.valid_iterator), correct_predictions.double()/len(self.valid_iterator)