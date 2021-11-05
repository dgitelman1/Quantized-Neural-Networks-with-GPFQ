import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''
    Add new annotations later.
    '''
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # hidden_dim = [512, 256, 128]
        # Define layers of MLP. Using nn.Sequential is also OK.
        self.layer1 = nn.Linear(input_dim, hidden_dim[0], bias=True)
        self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[1], bias=True)
        self.layer3 = nn.Linear(hidden_dim[1], hidden_dim[2], bias=True)
        self.layer4 = nn.Linear(hidden_dim[2], out_dim, bias=True)

    def forward(self, X):
        X = X.view(-1, self.input_dim)
        X = self.layer1(X)
        X = F.relu(X)
        X = self.layer2(X)
        X = F.relu(X)
        X = self.layer3(X)
        X = F.relu(X)
        X = self.layer4(X)
        return F.log_softmax(X, dim=1)

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.log_softmax(logits, dim=1)
        return probs


class CNN(nn.Module):
    '''
    Basic CNN to test quantization.
    '''
    def __init__(self):
        super().__init__()

        # Define layers of MLP. Using nn.Sequential is also OK.
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # X = X.view(-1, self.input_dim)
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)