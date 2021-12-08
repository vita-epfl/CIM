import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, lags = 3, encoder_hidden_dim = 16, fc_hidden_dim = 16, input_dim = 2):
        super(MLP, self).__init__()

        self.encoders = []
        for i in range(input_dim):
            self.encoders.append(nn.Sequential(
                nn.Linear(lags, encoder_hidden_dim).cuda(),
                nn.LeakyReLU(True)
            ))
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim * encoder_hidden_dim, fc_hidden_dim).cuda(),
            nn.ReLU(True),
            nn.Linear(fc_hidden_dim, fc_hidden_dim).cuda(),
            nn.ReLU(True),
            nn.Linear(fc_hidden_dim, 1).cuda(),
            nn.Sigmoid()
        )
        
    def forward(self, X):
        hiddens = []
        for i in range(len(self.encoders)):
            hiddens.append(self.encoders[i](X[:, i]))
        
        hidden = torch.cat(hiddens, 1)
        return self.fc(hidden)
    
    def loss(self, X, y, criterion = nn.BCELoss()):
        y_hat = self(X)
        loss = criterion(y_hat.squeeze(), y)
        return loss
    
    def classify(self, X, p = 0.8):
        y_hat = self(X)
        return y_hat.detach().cpu().numpy() > p 


class FullMLP(nn.Module):
    def __init__(self, lags = 3, encoder_hidden_dim = 8, fc_hidden_dim = 256):
        super(FullMLP, self).__init__()

        self.encoders = []
        for i in range(128):
            self.encoders.append(nn.Sequential(
                nn.Linear(lags, encoder_hidden_dim).cuda(),
                nn.LeakyReLU(True)
            ))
        
        self.fc = nn.Sequential(
            nn.Linear(128 * encoder_hidden_dim, fc_hidden_dim).cuda(),
            nn.ReLU(True),
            nn.Linear(fc_hidden_dim, fc_hidden_dim).cuda(),
            nn.ReLU(True),
            nn.Linear(fc_hidden_dim, 1).cuda(),
            nn.Sigmoid()
        )
        
    def forward(self, X):
        hiddens = []
        for i in range(128):
            hiddens.append(self.encoders[i](X[:, i]))
        
        hidden = torch.cat(hiddens, 1)
        return self.fc(hidden)
    
    def loss(self, X, y, criterion = nn.BCELoss()):
        y_hat = self(X)
        loss = criterion(y_hat.squeeze(), y)
        return loss
    
    def classify(self, X, p = 0.8):
        y_hat = self(X)
        return y_hat.detach().cpu().numpy() > p 