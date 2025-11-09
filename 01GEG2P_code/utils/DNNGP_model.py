import torch.nn as nn
import torch.nn.functional as F





class Net(nn.Module):
    def __init__(self,length):
        """
        Parameters
        ----------
        sequence_length : int
            Input sequence length
        n_genomic_features : int   n_targets
            Total number of features to predict
        """
        super(Net, self).__init__()
        self.nnet1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size= 4, padding=0, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(64),)
        self.pad1 = nn.ZeroPad2d(padding=(1, 2, 0, 0))
        
        self.nnet2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size= 4, padding=0, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(64))
        self.pad2 = nn.ZeroPad2d(padding=(1, 2, 0, 0))
        
        self.nnet3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size= 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64))
        self.pad3 = nn.ZeroPad2d(padding=(1, 2, 0, 0))
            
        self.dense_1 = nn.Sequential(
            nn.Linear(length*64, 3),
            nn.Dropout(p=0.3))

        self.dense_2 = nn.Sequential(
            nn.Linear(3, 1))



        #190   #121600
    def forward(self, x):
        """Forward propagation of a batch.
        """
        x = x.unsqueeze(1)
        #x = x.transpose(1, 2)
        x = self.pad1(self.nnet1(x))
        #print('x1.shape', x.shape)            x1.shape torch.Size([32, 64, 42938])
        x = self.pad2(self.nnet2(x))
        #print('x2.shape', x.shape)     x2.shape torch.Size([32, 64, 42938])
        x = self.pad3(self.nnet3(x))
        #print('x.shape', x.shape)      x.shape torch.Size([32, 64, 42938])
        x = x.reshape(x.shape[0], -1)
        x = self.dense_1(x)
        predict = self.dense_2(x)
        return predict
        

