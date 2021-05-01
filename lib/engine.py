import torch

from lib import models, train


class Engine:
    def __init__(self, trial, params, data_loader, data_loader_test):
        self.trial = trial
        self.params = params
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test
        print(type(self.params))
        print(dir(self.params))
        print(self.params["optimizer"] == "SGD")
        ## setup
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = getattr(models, self.params["model"])(self.params["hidden_layer_size"], self.params["box_score_thresh"])
        self.paramss = [p for p in self.model.parameters() if p.requires_grad]
        self.model.to(self.device)
        print(type(self.params))

        if self.params["optimizer"] == "SGD":
            self.optimizer = optimizer = torch.optim.SGD(self.paramss, lr=self.params["lr"], momentum=self.params["momentum"], weight_decay=self.params["weight_decay"])

        if self.params["optimizer"] == "Adam":
            self.optimizer = optimizer = torch.optim.Adam(self.paramss, lr=self.params["lr"], weight_decay=self.params["weight_decay"])

        if self.params["scheduler"] == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params["step_size"], gamma=self.params["gamma"])

        if self.params["scheduler"] == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.params["milestones"], gamma=self.params["gamma"])

        if self.params["scheduler"] == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def train(self):
        return train.train_model(self.trial, self.params, self.model, self.optimizer, self.scheduler, self.device, self.data_loader, self.data_loader_test)
