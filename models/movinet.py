import torch
import torch.nn as nn
import torch.functional as F 



MODEL_ARCHITECTURE = {
    "A0" : {
        "kernels" : [
            [(1, 5, 5)],
            [(5, 3, 3), (3, 3, 3), (3, 3, 3)],
            [(5, 3, 3), (3, 3, 3), (3, 3, 3)],
            [(5, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
            [(5, 3, 3), (1, 5, 5), (1, 5, 5), (1, 5, 5)],
        ],
        "bases" :   [
            [8],
            [32, 32, 32],
            [56, 56, 56],
            [56, 56, 56, 56],
            [104, 104, 104, 104],
        ],
        "expands" : [
            [40],
            [80, 80, 80],
            [184, 112, 184],
            [184, 184, 184, 184],
            [344, 280, 280, 344],
        ],
        "out" : 600,
        "pool_size" : 5,
    },
    # "A1" : {
    #     "kernels" : [
    #         []
    #     ]
    # }
}

class Block(nn.Module):

    def __init__(self, in_channels, kernels, bases, expands):

        super(Block, self).__init__()
        self.bases = bases
        self.kernels = kernels
        self.expands = expands
        self.in_channels = in_channels
        self.sequential = nn.ModuleList()

        self.sequential.append(self.make_layer(in_channels = self.in_channels, kernel = self.kernels[0], base = self.bases[0], expand= self.expands[0]))
        if len(self.bases) > 1:
            for i, base in enumerate(self.bases[1:]):
                self.sequential.append(self.make_layer(in_channels = self.expands[i], kernel = self.kernels[i + 1], base = self.bases[i + 1], expand= self.expands[i + 1]))


    def make_layer(self, in_channels, kernel, base, expand):

        return nn.Sequential(
            nn.Conv3d(in_channels= in_channels, out_channels=base, kernel_size=kernel),
            nn.Conv3d(in_channels = base, out_channels = expand, kernel_size= (1, 1, 1)),
            nn.BatchNorm3d(num_features= expand),
            nn.ReLU()
        )

    def forward(self, x):

        for layer in self.sequential:
            x = layer(x)
        return x


class MoviNet(nn.Module):

    def __init__(self, in_channels = 3, n_classes = 2, arch = "A0", sample_duration = 32):

        super(MoviNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.arch = arch


        self.kernels = MODEL_ARCHITECTURE[arch]['kernels']
        self.bases   = MODEL_ARCHITECTURE[arch]['bases']
        self.expands = MODEL_ARCHITECTURE[arch]['expands']

        self.out = MODEL_ARCHITECTURE[arch]['out']
        self.pool_size = MODEL_ARCHITECTURE[arch]['pool_size']

        self.sample_duration = sample_duration

        self.conv_1 = nn.Conv3d(in_channels = self.in_channels, out_channels=16, kernel_size=(1, 3, 3))
        self.block_2 = Block(in_channels = 16, kernels = self.kernels[0], bases = self.bases[0], expands=self.expands[0])
        self.block_3 = Block(in_channels = self.expands[0][-1], kernels = self.kernels[1], bases = self.bases[1], expands=self.expands[1])
        self.block_4 = Block(in_channels = self.expands[1][-1], kernels = self.kernels[2], bases = self.bases[2], expands=self.expands[2])
        self.block_5 = Block(in_channels = self.expands[2][-1], kernels = self.kernels[3], bases = self.bases[3], expands=self.expands[3])
        self.block_6 = Block(in_channels = self.expands[3][-1], kernels = self.kernels[4], bases = self.bases[4], expands=self.expands[4])

        self.conv_7 = nn.Conv3d(in_channels = self.expands[-1][-1], out_channels=self.out, kernel_size=(1, 1, 1))
        self.pool_8 = nn.MaxPool3d(kernel_size=(self.sample_duration, self.pool_size, self.pool_size))

        self.dense_9 = nn.Linear(1, 2048)
        self.dense_10 = nn.Linear(2048, self.n_classes)

    def forward(self, x):

        x = self.conv_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.conv_7(x)
        print(x.shape)
        x = nn.ReLU(self.pool_8(x))
        x = x.view(1, -1)
        x = nn.ReLU(self.dense_9(x))
        x = self.dense_10(x)


if __name__ == "__main__":
    movinet = MoviNet()
    # print(movinet)
    x = torch.randn((1, 3, 32, 224, 224))
    y = movinet(x)
