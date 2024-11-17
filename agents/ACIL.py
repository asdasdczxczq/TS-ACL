# acil.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from utils.utils import EarlyStopping
from utils.AnalyticLinear import AnalyticLinear, RecursiveLinear
from utils.Buffer import RandomBuffer
import os


class ACIL(nn.Module):
    def __init__(self, backbone_output, backbone, buffer_size, gamma, device=None, dtype=torch.double, linear=RecursiveLinear,learned_classes=None):
        super(ACIL, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.buffer_size = buffer_size
        self.buffer = RandomBuffer(backbone_output, buffer_size, **factory_kwargs)
        self.analytic_linear = linear(buffer_size, gamma, **factory_kwargs)
        self.learned_classes = learned_classes
        self.eval()
    
    @torch.no_grad()
    def feature_expansion(self, X):
        return self.buffer(self.backbone(X))
    
    @torch.no_grad()
    def forward(self, X):
        return self.analytic_linear(self.feature_expansion(X))
    
    @torch.no_grad()
    def fit(self, X, y, *args, **kwargs):
        num_targets = int(y.max()) + 1
        Y = torch.nn.functional.one_hot(y, num_classes=num_targets).to(X.dtype)
        X = self.feature_expansion(X)
        self.analytic_linear.fit(X, Y)
    
    @torch.no_grad()
    def update(self):
        self.analytic_linear.update()


class ACILLearner(BaseLearner):
    def __init__(self, model: nn.Module, args):
        super(ACILLearner, self).__init__(model, args)
        self.learning_rate = args.learning_rate
        self.buffer_size = args.buffer_size
        self.gamma = args.gamma
        self.base_epochs = args.base_epochs
        self.warmup_epochs = args.warmup_epochs
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.label_smoothing = args.label_smoothing
        self.separate_decay = args.separate_decay
        self.feature_dim = args.feature_dim
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing).to(self.device)
        self.previous_model = None  # To store the model before replacing it

    def make_model(self):
        # Extract backbone (input_norm and encoder) from previous model
        backbone = nn.Sequential(
            self.previous_model[0],
            self.previous_model[1]
        )
        self.acil_model = ACIL(
            backbone_output=self.feature_dim,
            backbone=backbone,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            device=self.device,
            dtype=torch.double,
            linear=RecursiveLinear,
            learned_classes=self.learned_classes
        )
        self.acil_model.to(self.device)
        self.model = self.acil_model  # Overwrite self.model with ACIL model
        print(self.model)

    def before_task(self, y_train):
        self.task_now += 1
        self.classes_in_task = list(set(y_train.tolist()))
        n_new_classes = len(self.classes_in_task)
        assert n_new_classes > 1, "A task must contain more than 1 class"

       
            
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
        temp_model = nn.Sequential(
            self.model.input_norm,
            self.model.encoder,
            nn.Linear(self.feature_dim, len(self.classes_in_task)),
        ).to(self.device)
        
        optimizer = optim.SGD(
            temp_model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay if not self.separate_decay else 0.0,
        )

        # Scheduler setup remains the same
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.base_epochs - self.warmup_epochs, eta_min=1e-6
        )
        if self.warmup_epochs > 0:
            warmup_scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=self.warmup_epochs,
            )
            scheduler = lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, scheduler], [self.warmup_epochs]
            )

        early_stopping = EarlyStopping(path=self.ckpt_path, patience=self.args.patience, mode='min', verbose=self.verbose)

        for epoch in range(self.base_epochs):
            temp_model.train()
            epoch_loss, epoch_acc = self.train_epoch(temp_model, train_loader, optimizer)
            temp_model.eval()
            val_loss, val_acc = self.evaluate_epoch(temp_model, val_loader)
            if self.verbose:
                print(f'Epoch {epoch+1}/{self.base_epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')
            scheduler.step()
            early_stopping(val_loss, temp_model)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

        self.previous_model = temp_model               




    def train_epoch(self, model, dataloader, optimizer):
        total = 0
        correct = 0
        epoch_loss = 0
        model.train()
        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)
            optimizer.zero_grad()
            outputs = model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            optimizer.step()
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
            print(y)
    
    def evaluate(self, task_stream, path=None):
        self.model.eval()
        for task_id, task in enumerate(task_stream):
            (x_test, y_test), _ = task
            x_test, y_test = x_test.to(self.device), y_test.to(self.device)
            outputs = self.model(x_test)
            loss = self.criterion(outputs, y_test)
            prediction = torch.argmax(outputs, dim=1)
            acc = prediction.eq(y_test).sum().item() / y_test.size(0) * 100
            print(f"Task {task_id+1} - Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
        pass
