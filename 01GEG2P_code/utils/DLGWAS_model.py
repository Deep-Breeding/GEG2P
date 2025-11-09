import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,length):
        super(Net, self).__init__()
        self.length = length
        self.num_classes = 1
        self.dropout = nn.Dropout(0.75)
        self.dnn_size_list = [1]

        self.leftcnn1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=4),
            nn.GELU())
        self.pad1 = nn.ZeroPad2d(padding=(1, 2, 0, 0))

        self.leftcnn2 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=20),
            nn.GELU())
        self.pad2 = nn.ZeroPad2d(padding=(10, 9, 0, 0))

        self.rightcnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=4),
            nn.GELU())
        self.pad3 = nn.ZeroPad2d(padding=(1, 2, 0, 0))

        self.centralcnn = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=4),
            nn.GELU())
        self.pad4 = nn.ZeroPad2d(padding=(1, 2, 0, 0))

        self.temp = nn.Linear(10 * length, self.dnn_size_list[-1])

    def forward(self, x):
        #print("x1.size:",x.shape)#[64,4,1,65118]
        #x = np.squeeze(x, axis=2)
        #x = x.transpose(1, 2)
        #x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        left_tower_x = x
        right_tower_x = x

        # left tower
        left_tower_x = self.pad1(self.leftcnn1(left_tower_x))  # [64, 10, 65118]
        left_tower_x = self.pad2(self.leftcnn2(left_tower_x))
        #print("left_tower_x.size:", left_tower_x.shape)  # [64, 10, 65118]

        # right tower
        right_tower_x = self.pad3(self.rightcnn(right_tower_x))
        #print("right_tower_x.size:", right_tower_x.shape)  # [64, 10, 65118]

        x = torch.add(left_tower_x, right_tower_x)
        #print("x2.size:", x.shape)  # [64, 10, 65118]

        # central tower
        x = self.pad4(self.centralcnn(x))
        #print("x3.size:", x.shape)  # x3.size: torch.Size([32, 10, 170000])


        x = self.dropout(x)
        x = x.view(-1, 10 * self.length)
        x = self.dropout(x)

        outputs = self.temp(x)
        return outputs