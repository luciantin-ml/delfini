import torch

from lib import models, train


class Engine:
    def __init__(self, trial, params, data_loader, data_loader_test):
        self.trial = trial
        self.params = params
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test

        ## setup
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = getattr(models, params['model'])(params["hidden_layer_size"], params["box_score_thresh"])
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.model.to(self.device)

        if params["optimizer"] == "SGD":
            self.optimizer = optimizer = torch.optim.SGD(params, lr=params["lr"], momentum=params["momentum"], weight_decay=params["weight_decay"])

        if params["optimizer"] == "SGD":
            self.optimizer = optimizer = torch.optim.Adam(params, lr=params["lr"], weight_decay=params["weight_decay"])

        if params["scheduler"] == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=params["step_size"], gamma=params["gamma"])

        if params["scheduler"] == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=params["milestones"], gamma=params["gamma"])

        if params["scheduler"] == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def train(self):
        return train.train_model(self.trial, self.params["num_epochs"], self.model, self.optimizer, self.scheduler, self.device, self.data_loader, self.data_loader_test)
