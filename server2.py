from collections import OrderedDict
import torch.optim as optim

from omegaconf import DictConfig
from statistics import mean

import torch

from model2 import Net, test

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {
            "lr": config.lr,
            "beta1" : config.beta1,
            "beta2" : config.beta2,
            "local_epochs": config.local_epochs,
        }
    return fit_config_fn

def get_evaluate_fn(num_classes: int, testloader):
    def evaluate_fn(server_round: int, parameters, config):
        model = Net(num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy,precision,recall,f1 = test(model, testloader, device)
        return loss, {"accuracy": accuracy,"precision":precision,"recall":recall,"f1":f1}   
    return evaluate_fn









