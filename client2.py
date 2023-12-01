from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import torch.optim as optim

import torch
import flwr as fl

from model2 import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, trainloader, valloader,num_classes) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = valloader

        # a model that is randomly initialised at first
        self.model = Net(num_classes)
        self.model.load_pretrained_model('model2.pth')

        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config["lr"]
        beta1 = config["beta1"]
        beta2=config["beta2"]
        epochs = config["local_epochs"]
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2))
        train(self.model, self.trainloader, optimizer, epochs, self.device)
        return self.get_parameters({}), len(self.trainloader), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy,precision,recall,f1= test(self.model, self.valloader, self.device)
        return float(loss), {"accuracy": accuracy,"precision":precision,"recall":recall,"f1":f1}
    
def generate_client_fn(trainloaders, valloaders,num_classes):
        def client_fn(cid: str):
            return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            num_classes=num_classes,
        )
        return client_fn
    

        


