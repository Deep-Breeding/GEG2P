import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, sequence_length):
        """
        Parameters
        ----------
        sequence_length : int

        """
        super(Net, self).__init__()
        self.nnet = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, stride=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )


        with torch.no_grad():
            dummy = torch.zeros(1, 1, sequence_length)  # [batch=1, channel=1, L]
            out = self.nnet(dummy)                      # [1, 1, L']
            flatten_size = out.shape[1] * out.shape[2]  # = 1 * L'

        self.dense_1 = nn.Sequential(
            nn.Linear(flatten_size, 5),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.dense_2 = nn.Linear(5, 1)

    def forward(self, x):
        x = x.unsqueeze(1)         # [B, 1, L]
        out = self.nnet(x)         # [B, 1, L']
        out = out.transpose(1, 2)  # [B, L', 1]
        out = out.reshape(out.shape[0], -1)  # flatten
        out = self.dense_1(out)
        predict = self.dense_2(out)
        return predict