# acil.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from utils.utils import EarlyStopping
from utils.AnalyticLinear import AnalyticLinear, RecursiveLinear,GeneralizedARM
from utils.Buffer import RandomBuffer
import os
from utils.optimizer import set_optimizer, adjust_learning_rate
import numpy as np

class TSACL(nn.Module):
    def __init__(self, backbone_output, backbone, buffer_size, gamma, device=None, dtype=torch.double, linear=RecursiveLinear,learned_classes=None):
        super(TSACL, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.buffer_size = buffer_size
        self.buffer = RandomBuffer(backbone_output, buffer_size, **factory_kwargs)
        self.analytic_linear = linear(buffer_size, gamma, **factory_kwargs)
       
        self.eval()
    
    @torch.no_grad()
    def feature_expansion(self, X):
        return self.buffer(self.backbone(X))
    
    @torch.no_grad()
    def forward(self, X):
        return self.analytic_linear(self.feature_expansion(X))
    
    @torch.no_grad()
    def fit(self, X, y, *args, **kwargs):

        X = self.feature_expansion(X)
        self.analytic_linear.fit(X, y)
    
    @torch.no_grad()
    def update(self):
        self.analytic_linear.update()


class TSACLLearner(BaseLearner):
    def __init__(self, model: nn.Module, args):
        super(TSACLLearner, self).__init__(model, args)
        self.learning_rate = args.lr
        self.buffer_size = args.buffer_size
        self.gamma = args.gamma
        self.epochs = args.epochs
        self.feature_dim = args.feature_dim
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.previous_model = None  
        self.optimizer=set_optimizer(self.model, args)
        self.input_norm = args.input_norm
    def make_model(self):
        # Extract backbone (input_norm and encoder) from previous model
        backbone = nn.Sequential(
            self.previous_model[0],
            self.previous_model[1]
        )
        self.acil_model = TSACL(
            backbone_output=self.feature_dim,
            backbone=backbone,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            device=self.device,
            dtype=torch.double,
            linear=RecursiveLinear,
           
        )
        self.acil_model.to(self.device)
        self.model = self.acil_model  # Overwrite self.model with ACIL model
        # print(self.model)

    def before_task(self, y_train):
        self.task_now += 1
        self.classes_in_task = list(set(y_train.tolist()))
        n_new_classes = len(self.classes_in_task)
        # assert n_new_classes > 1, "A task must contain more than 1 class"

       
            
    def after_task(self, x_train, y_train):
        self.learned_classes += self.classes_in_task




    def learn_task(self, task):
       
        (x_train, y_train), (x_val, y_val), _ = task
        

      
       
        self.before_task(y_train)
        print("self.task_now",self.task_now)
        if self.task_now == 0:
            # Base training with standard model
            train_dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
            val_dataloader = Dataloader_from_numpy(x_val, y_val, self.batch_size, shuffle=False)
            self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                        steps_per_epoch=len(train_dataloader),
                        epochs=self.epochs,
                        max_lr=self.args.lr)
            self.base_training(train_dataloader, val_dataloader)
            self.make_model()  # Create ACIL model, replacing the head

            data_loader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
            self.after_task(x_train, y_train)
            self.incremental_learning(data_loader)#Re-Align

           
        else:
           
            # Incremental learning with ACIL model
            data_loader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
            self.incremental_learning(data_loader)
            self.after_task(x_train, y_train)
           

    def base_training(self, train_loader, val_loader):
        # Initialize a temporary model for base training
        if self.input_norm=='none':
            self.model.input_norm = nn.Identity()
        temp_model = nn.Sequential(
            self.model.input_norm,
            self.model.encoder,
            nn.Linear(self.feature_dim, len(self.classes_in_task)),
        ).to(self.device)
  
  
        early_stopping = EarlyStopping(path=self.ckpt_path, patience=self.args.patience, mode='min', verbose=self.verbose)

        for epoch in range(self.epochs):
            temp_model.train()
            epoch_loss, epoch_acc = self.train_epoch(temp_model, train_loader)
            temp_model.eval()
            val_loss, val_acc = self.evaluate_epoch(temp_model, val_loader)
            adjust_learning_rate(self.optimizer, self.scheduler, epoch + 1, self.args)
            if self.args.lradj == 'TST':        
                self.scheduler.step()
            
            if self.verbose:
                print(f'Epoch {epoch+1}/{self.epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')
            
            early_stopping(val_loss, temp_model)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

        self.previous_model = temp_model               




    def train_epoch(self, model, dataloader):
        total = 0
        correct = 0
        epoch_loss = 0
        model.train()
        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)
            self.optimizer.zero_grad()
            outputs = model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            epoch_loss += loss.item()
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()
        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)
        return epoch_loss, epoch_acc
    

    @torch.no_grad()
    def evaluate_epoch(self, model, dataloader):
        total = 0
        correct = 0
        epoch_loss = 0
        model.eval()
        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)
            outputs = model(x)
            loss = self.criterion(outputs, y)
            epoch_loss += loss.item()
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()
        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)
        return epoch_loss, epoch_acc
    
    def incremental_learning(self, data_loader):
        self.model.eval()
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.fit(x, y)
        self.model.update()
        
        

    @torch.no_grad()
    def evaluate(self, task_stream, path=None):
        """
        Evaluate on the test sets of all the learned tasks (<= task_now).
        Save the test accuracies of the learned tasks in the Acc matrix.
        Visualize the feature space with TSNE, if self.tsne == True.

        Args:
            task_stream: Object of Task Stream, list of ((x_train, y_train), (x_val, y_val), (x_test, y_test)).
            path: path prefix to save the TSNE png files.

        """
        # Get num_tasks and create Accuracy Matrix for 'val set and 'test set'
        if self.task_now == 0:
            self.num_tasks = task_stream.n_tasks
            self.Acc_tasks = {'valid': np.zeros((self.num_tasks, self.num_tasks)),
                              'test': np.zeros((self.num_tasks, self.num_tasks))}

        # Reload the original optimal model to prevent the changes of statistics in BN layers.
        # self.model.load_state_dict(torch.load(self.ckpt_path))

        eval_modes = ['valid', 'test']  # 'valid' is for checking generalization.
        for mode in eval_modes:
            if self.verbose:
                print('\n ======== Evaluate on {} set ========'.format(mode))
            for i in range(self.task_now + 1):
                (x_eval, y_eval) = task_stream.tasks[i][1] if mode == 'valid' else task_stream.tasks[i][2]
                eval_dataloader_i = Dataloader_from_numpy(x_eval, y_eval, self.batch_size, shuffle=False)


                eval_loss_i, eval_acc_i = self.cross_entropy_epoch_run(eval_dataloader_i, mode='test')

                if self.verbose:
                    print('Task {}: Accuracy == {}, Test CE Loss == {} ;'.format(i, eval_acc_i, eval_loss_i))
                self.Acc_tasks[mode][self.task_now][i] = np.around(eval_acc_i, decimals=2)


            # Print accuracy matrix of the tasks on this run
            if self.task_now + 1 == self.num_tasks and self.verbose:
                with np.printoptions(suppress=True):  # Avoid Scientific Notation
                    print('Accuracy matrix of all tasks:')
                    print(self.Acc_tasks[mode])