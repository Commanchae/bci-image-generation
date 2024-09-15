import torch
import torch.nn as nn

class ImageGenerator(nn.Module):
    def __init__(self):
        super(ImageGenerator, self).__init__()

        # Encoding Process #
        self.encode_conv2d_0 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1, 8), stride=1, padding=0)
        self.encode_conv2d_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2), stride=1, padding=0)
        self.encode_covn2d_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 4), stride=1, padding=0)
        self.relu = nn.ReLU()
        self.bn0 = nn.BatchNorm2d(256)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(1, 2))
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.5)
        # Decoding Process #
        linear_shape = self.check_linear_shape()

        self.fc1 = nn.Linear(linear_shape, 1024)
        self.fc2 = nn.Linear(1024, 32 * 32)

        self.sigmoid = nn.Sigmoid()

    def check_linear_shape(self):
        x = torch.rand((1, 4, 512))
        shape = self.encode(x).shape
        return shape[1]

    def encode(self, x: torch.Tensor):
        '''
        Takes in x of shape [b, 4, 512]
        '''
        # First, convert to [b, 1, 4, 512] to use convolutional layers with in_channels=1.
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Let's try having 1 convolutional layer for now that does spatial-temporal feature extraction.
        x = self.encode_conv2d_0(x) # [b, 1, 4, 512] -> [b, 4, 1, 509]
        x = self.bn0(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.avg_pool2d(x) # [b, 4, 1, 509] -> [b, 4, 1, 127]

        x = self.encode_conv2d_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.avg_pool2d(x)

        x = self.encode_covn2d_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.avg_pool2d(x)

        # Continue by adding upsampling method as well   after condensing conv output into fc.
        x = x.flatten(start_dim=1) # Results in shape [b, 4 * 1 * 127] or [b, 508]

        return x
    
    def decode(self, x:torch.Tensor):
        '''
        Takes in a latent vector of shape [b, 508].
        '''
        x = self.fc1(x) # [b, 508] -> [b, 1024]
        x = self.relu(x) 

        x = self.fc2(x) # [b, 1024] -> [b, 32*32]
        x = self.sigmoid(x)

        x = x.view(-1, 1, 32, 32)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x