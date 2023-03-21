import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception(
                f'Pooling type of {pool_type} is not supported. It must be one of "max", "avg" and "avg+max".')
        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception(
                f'Pooling type of {pool_type} is not supported. It must be one of "max", "avg" and "avg+max".')
        return x


class PANNS_CNN6(nn.Module):
    """
    The CNN14(14-layer CNNs) mainly consist of 4 convolutional blocks while each convolutional
    block consists of 1 convolutional layers with a kernel size of 5 × 5.

    Reference:
        PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
        https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 512

    def __init__(self, num_class, input_size, dropout=0.1, extract_embedding: bool = True):

        super(PANNS_CNN6, self).__init__()
        self.bn0 = nn.BatchNorm2d(input_size)
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.emb_size, num_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute([0, 3, 2, 1])
        x = self.bn0(x)
        x = x.permute([0, 3, 2, 1])

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.mean(dim=3)
        x = x.max(dim=2)[0] + x.mean(dim=2)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))

        x = self.dropout(output)
        logits = self.fc(x)

        return logits


class PANNS_CNN10(nn.Module):
    """
    The CNN10(14-layer CNNs) mainly consist of 4 convolutional blocks while each convolutional
    block consists of 2 convolutional layers with a kernel size of 3 × 3.

    Reference:
        PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
        https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 512

    def __init__(self, num_class, input_size, dropout=0.1, extract_embedding: bool = True):

        super(PANNS_CNN10, self).__init__()
        self.bn0 = nn.BatchNorm2d(input_size)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.emb_size, num_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute([0, 3, 2, 1])
        x = self.bn0(x)
        x = x.permute([0, 3, 2, 1])

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.mean(dim=3)
        x = x.max(dim=2)[0] + x.mean(dim=2)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))

        x = self.dropout(output)
        logits = self.fc(x)

        return logits


class PANNS_CNN14(nn.Module):
    """
    The CNN14(14-layer CNNs) mainly consist of 6 convolutional blocks while each convolutional
    block consists of 2 convolutional layers with a kernel size of 3 × 3.

    Reference:
        PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
        https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 2048

    def __init__(self, num_class, input_size, dropout=0.1, extract_embedding: bool = True):

        super(PANNS_CNN14, self).__init__()
        self.bn0 = nn.BatchNorm2d(input_size)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.emb_size, num_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute([0, 3, 2, 1])
        x = self.bn0(x)
        x = x.permute([0, 3, 2, 1])

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.mean(dim=3)
        x = x.max(dim=2)[0] + x.mean(dim=2)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))

        x = self.dropout(output)
        logits = self.fc(x)

        return logits
