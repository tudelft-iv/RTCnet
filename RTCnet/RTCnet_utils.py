

import sys 
import os 
import os.path as osp 

import numpy as np 
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn 
import tqdm



class Trainer(object):
    """  
    Helper class for model training
    """
    def __init__(self, 
                model, 
                loss_func,
                optimizer,
                lr_scheduler = None,
                eval_frequency = -1,
                use_gpu = False,
                lr_decay_step=0,
                lr_decay_f=0.5,
                lr_clip=1e-4,
                save_checkerpoint_to = '',
                append_str = ''):

        """  
        Constructor for trainer

        Parameters
        ----------
        model: RTCnet
               The RTC model
        loss_func: torch.nn.CrossEntropyLoss 
                   Loss function in torch.nn
        optimizer: torch.optim.Adam
                   Optimizer provided by Pytorch
        lr_scheduler: torch.optim.lr_scheduler.StepLR
                      Scheduler for training
        eval_frequency: int
                        How frequency the model is evaluated, -1 means evaluating the model after every epoch
        use_gpu: bool
                 Whether GPU is used
        lr_decay_step: int
                       The training step after which the learning rate is lowered
        lr_decay_f: float
                    The factor for the decay of learning rate
        lr_clip: float
                 The clip value after which the learning rate stops decaying
        save_checkerpoint_to: string
                              Path to save the trained model
        append_str: string
                    The string appeded for illustrating more details after the saved model
        """
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_frequency = eval_frequency
        self.use_gpu = use_gpu
        self.lr_decay_step=lr_decay_step
        self.lr_decay_f =lr_decay_f
        self.lr_clip = lr_clip
        self.save_checkerpoint_to = save_checkerpoint_to
        self.trace_loss = []
        self.trace_loss_train = []
        self.append_str = append_str
        if self.use_gpu:
            self.model.cuda()
    
    def _train_it(self, it, batch):
        """  
        Single iteration for training
        """
        self.model.train()
        self.optimizer.zero_grad()
        _, loss, res_dict = self._farward_pass(self.model, batch)
        loss.backward()
        self.optimizer.step()
        return res_dict

    def _farward_pass(self, model, batch):
        """  
        Forward pass for single iteration
        """
        inputs,labels = batch['data'], batch['label']
        if self.use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        preds = model(inputs)
        loss = self.loss_func(preds.view(labels.numel(),-1), labels.view(-1))
        res_dict = {
            "loss": loss.item()
        }
        return preds, loss, res_dict
    
    def _val_epoch(self, val_loader):
        """  
        One epoch for validation
        """
        self.model.eval()
        eval_dict = {}
        total_loss = 0.0
        cnt = 0
        for i, data in tqdm.tqdm(enumerate(val_loader, 0), total=len(val_loader), leave=False, desc='val'):
            with torch.no_grad():
                _, loss, res_dict = self._farward_pass(self.model, data)
                total_loss += loss.item()
            cnt += 1
        return total_loss/cnt

    def _train_epoch(self, train_loader):
        """  
        One epoch for training
        """
        self.model.eval()
        eval_dict = {}
        total_loss = 0.0
        cnt = 0
        for i, data in tqdm.tqdm(enumerate(train_loader, 0), total=len(train_loader), leave=False, desc='test_by_train'):
            with torch.no_grad():
                _, loss, res_dict = self._farward_pass(self.model, data)
                total_loss += loss.item()
            cnt += 1
        return total_loss/cnt

    def _save_checkerpoint(self, is_best, path=''):
        """  
        Save the trained model
        """
        if self.append_str=='':
            torch.save(self.model.state_dict(), osp.join(path,"checkerpoint.pth"))
        else:
            torch.save(self.model.state_dict(), osp.join(path,"checkerpoint_{}.pth".format(self.append_str)))
        if is_best:
            torch.save(self.model.state_dict(), osp.join(path,"best_checkerpoint_{}.pth".format(self.append_str) ))
        
    def train(self,
            n_epochs,
            train_loader,
            val_loader=None,
            best_loss=1e5,
            start_it=0):
        """  
        Wrapper for training
        
        Parameters
        ----------
        n_epochs: int
                  Number of epochs
        train_loader: dataloader
                      The dataloader for the training dataset
        val_loader: dataloader
                    The dataloader for the validation dataset
        best_loss: fload
                   The lowest loss until now
        start_it: int
                  The index of iteration from where the training continues. This is used for resuming training from a saved model.
        """
        eval_frequency = (
            self.eval_frequency if self.eval_frequency>0 else len(train_loader) 
            # Default choice: evaluate the model after every single epoch
        )
        it = start_it
        if val_loader is not None:
            val_init_loss = self._val_epoch(val_loader)
            self.trace_loss.append(val_init_loss)
            train_loss = self._train_epoch(train_loader)
            self.trace_loss_train.append(train_loss)
            tqdm.tqdm.write("initial_validation_loss:{}".format(val_init_loss))
        with tqdm.trange(0, n_epochs, desc='epochs') as tbar, \
            tqdm.tqdm(total=eval_frequency, leave=False, desc='train') as pbar:
            for epoch in tbar:
                for batch in train_loader:
                    res_dict = self._train_it(it, batch)
                    it+=1
                    pbar.update()
                    pbar.set_postfix(dict(total_it=it, loss="{:.2f}".format(res_dict['loss'])))
                    tbar.refresh()

                    if it%eval_frequency==0:
                        pbar.close()
                        if val_loader is not None:
                            val_loss = self._val_epoch(val_loader)
                            self.trace_loss.append(val_loss)
                            is_best = val_loss<best_loss 
                            best_loss = min(best_loss, val_loss)
                            self._save_checkerpoint(is_best, path = self.save_checkerpoint_to)
                            tqdm.tqdm.write("validation loss is:{}".format(val_loss))
                        else:
                            raise "No validation data loader"
                        pbar = tqdm.tqdm(
                            total=eval_frequency, leave=False, desc='train'
                        )
                        pbar.set_postfix(dict(total_it=it, loss=res_dict['loss']))
        return best_loss

class Tester(object):
    """  
    Helper class for model testing

    """
    def __init__(self, 
                model,
                loss_func,
                use_gpu = False):
        """  
        model: RTCnet
               The RTC model
        loss_func: torch.nn.CrossEntropyLoss 
                   Loss function in torch.nn
        use_gpu: bool
                 Whether using gpu 
        """
        self.model = model 
        self.model.double()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model.cuda()
        self.loss_func = loss_func
        
    def test(self, data_loader):
        """  
        Wrapper for testing
        
        Parameters
        ----------
        train_loader: dataloader
                      The dataloader for the training dataset
        """
        total_loss = 0
        cnter = 1
        pred_labels_all = np.array([])
        true_labels_all = np.array([])
        pred_scores_all = []
        with tqdm.tqdm(total=len(data_loader), leave=False, desc='test') as pbar:
            for batch in data_loader:
                pbar.update()
                pred_labels,labels,res_dict = self._test_it(batch)
                total_loss += res_dict['loss']
                cnter+=1
                pred_labels_np, labels_np = pred_labels.cpu().numpy(), labels.cpu().numpy()
                pred_labels_all = np.append(pred_labels_all, pred_labels_np)
                true_labels_all = np.append(true_labels_all, labels_np)
                pred_scores_all.append(self.score_batch.squeeze())
            pred_labels_all = pred_labels_all.flatten()
            true_labels_all = true_labels_all.flatten()
            pred_scores_all = np.concatenate(pred_scores_all, axis=0)
            loss = total_loss/cnter
            self.pred_scores_all = pred_scores_all

        return loss, pred_labels_all, true_labels_all

        

    def _farward_pass(self, batch):
        """  
        Forward pass for a single iteration
        """
        inputs,labels = batch['data'], batch['label']
        if self.use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        preds = self.model(inputs)
        loss = self.loss_func(preds.view(labels.numel(),-1), labels.view(-1))
        _, pred_labels = torch.max(preds, 1)
        acc = (pred_labels.view(-1) == labels.view(-1)).float().sum() / labels.numel()
        res_dict = {
            "loss": loss.item(),
            "acc": acc.item()
        }
        with torch.no_grad():
            scores = nn.functional.softmax(preds,dim = 1).cpu().numpy()
        self.score_batch = scores
        return pred_labels, labels, res_dict

    def _test_it(self, batch):
        """  
        Single iteration for testing
        """
        self.model.eval()
        self.model.zero_grad()
        classes, labels, res_dict = self._farward_pass(batch)
        return classes, labels, res_dict



class Tester_ensemble(object):
    """  
    Helper class for testing ensemble model: in order to get scores rather than directly getting labels
    """
    def __init__(self, 
                model,
                loss_func,
                use_gpu = False):
        """  
        model: RTCnet
               The RTC model
        loss_func: torch.nn.CrossEntropyLoss 
                   Loss function in torch.nn
        use_gpu: bool
                 Whether using gpu 
        """
        self.model = model 
        self.model.double()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model.cuda()
        self.loss_func = loss_func
        
    def test(self, data_loader):
        scores_all = []
        with tqdm.tqdm(total=len(data_loader), leave=False, desc='test') as pbar:
            for batch in data_loader:
                pbar.update()
                scores = self._test_it(batch).cpu().numpy()
                scores_all.append(scores)
            scores_all = np.concatenate(scores_all, axis=0)
        return scores_all

        

    def _farward_pass(self, batch):
        inputs,labels = batch['data'], batch['label']
        if self.use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        preds = self.model(inputs)
        with torch.no_grad():
            scores = nn.functional.softmax(preds,dim = 1)[:, 1]
        return scores

    def _test_it(self, batch):
        """  
        Single iteration for testing. Only scores not labels are returned
        """
        self.model.eval()
        self.model.zero_grad()
        scores = self._farward_pass(batch)
        return scores 
