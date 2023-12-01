import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.autograd import Variable

def fgsm_attack(model, data, target, epsilon=0.01):
    data.requires_grad = True
    criterion=torch.nn.CrossEntropyLoss()
    output = model(data)
    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    return perturbed_data

class Net(nn.Module):
    def __init__(self, num_classes=2):
        input_dim = 115
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, 32)  
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(32, 16)  
        self.dropout2 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(16, num_classes)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

    def load_pretrained_model(self, model_path):
            checkpoint = torch.load(model_path)
            self.load_state_dict(checkpoint)
             

def train(net, trainloader, optimizer, epochs, device: str):
    epsilon=0.01
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

           
            adv_inputs = fgsm_attack(net, inputs.to(device), labels.to(device), epsilon=epsilon)

            
            outputs_clean = net(inputs.to(device))
            outputs_adv = net(adv_inputs)

           
            loss_clean = criterion(outputs_clean, labels.to(device))
            loss_adv = criterion(outputs_adv, labels.to(device))
            loss = loss_clean + loss_adv

           
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}')

    print('Finished Training with Adversarial Examples')

def test(net, testloader, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    net.to(device)
    test_loss = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for data in testloader:
            inputs, one_hot_labels = data  
            inputs, labels = inputs.to(device), torch.argmax(one_hot_labels, dim=1).to(device)
            outputs = net(inputs)

           
            loss = criterion(outputs, labels)
            test_loss += loss.item()

          
            predicted_classes = torch.argmax(outputs, dim=1)
            predictions.extend(predicted_classes.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

   
    test_loss /= len(testloader)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return test_loss,accuracy,precision,recall,f1






















    
