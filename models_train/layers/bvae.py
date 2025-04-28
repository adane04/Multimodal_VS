import torch
import torch.nn as nn
from torch.autograd import Variable
from models_train.layers.lstmcell import StackedLSTMCell
from torch.distributions.normal import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BayesianLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianLinear, self).__init__()
        
        # mean and log variance of weights
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).normal_(0, 0.1))
        self.weight_log_var = nn.Parameter(torch.Tensor(output_dim, input_dim).normal_(-3, 0.1))
        
        #  bias mean and log variance
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim).normal_(0, 0.1))
        self.bias_log_var = nn.Parameter(torch.Tensor(output_dim).normal_(-3, 0.1))

    def forward(self, x):
        # Sample weights and bias from their distributions
        weight_sigma = torch.exp(0.5 * self.weight_log_var)
        bias_sigma = torch.exp(0.5 * self.bias_log_var)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        return nn.functional.linear(x, weight, bias)


class eLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        '''
        Encoder LSTM
        '''
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear_mu = BayesianLinear(hidden_size, hidden_size)
        self.linear_var = BayesianLinear(hidden_size, hidden_size)

    def forward(self, frame_features):
        '''
        Args:
            frame_features: [seq_len, 1, hidden_size]
        Returns:
            h_last, c_last: Encoder hidden states
        '''
        self.lstm.flatten_parameters()
        _, (h_last, c_last) = self.lstm(frame_features)
        return h_last, c_last


class dLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=2):
        '''
        Decoder LSTM
        '''
        super().__init__()
        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
        self.out = BayesianLinear(hidden_size, input_size)

    def forward(self, seq_len, init_hidden):
        '''
        Args:
            seq_len: scalar (int)
            init_hidden: initial hidden and cell states
        Returns:
            out_features: List of hidden states over sequence
        '''
        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)
        
        x = Variable(torch.zeros(batch_size, hidden_size)).to(device=device)
        h, c = init_hidden
        out_features = []

        for _ in range(seq_len):
            (h_last, c_last), (h, c) = self.lstm_cell(x, (h, c))
            x = self.out(h_last)
            out_features.append(h_last)

        return out_features


#class BayesianVAE(nn.Module):
class VAE(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        '''
        Bayesian Variational Auto-Encoder
        '''
        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)
        self.softplus = nn.Softplus()

    def reparameterize(self, mu, log_variance):
        '''
        Sampling z via reparameterization trick
        '''
        sigma = torch.exp(0.5 * log_variance)
        epsilon = Variable(torch.randn(sigma.size())).to(device=device)
        return (mu + epsilon * sigma).unsqueeze(1)

    def forward(self, features):
        '''
        Args:
            features: Input features
        Returns:
            Encoded mu, log_variance, and decoded features
        '''
        seq_len = features.size(0)
        h, c = self.e_lstm(features)
        h = h.squeeze(1)
        
        # Bayesian outputs
        h_mu = self.e_lstm.linear_mu(h)
        h_log_variance = torch.log(self.softplus(self.e_lstm.linear_var(h)))
        
        # Reparameterize and decode
        h_sampled = self.reparameterize(h_mu, h_log_variance)
        decoded_features = self.d_lstm(seq_len, init_hidden=(h_sampled, c))
        
        # Reformat decoded features
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        
        return h_mu, h_log_variance, decoded_features
