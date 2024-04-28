import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, dropout=0.2, class_num=2):
        super(CNN, self).__init__()

        self.cov1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        self.cov2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        self.cov3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3),
            nn.ReLU()
        )
        self.fc = nn.Linear(in_features=18, out_features=class_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cov1(x)
        x = self.cov2(x)
        x = self.cov3(x)
        x = self.fc(x).squeeze(1)
        return x


class Classifier(nn.Module):
    def __init__(self, in_num=88, class_num=2, p=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=in_num, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(p)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(in_features=512, out_features=class_num),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


class MGNet(nn.Module):
    def __init__(self, in_num=88, class_num=2, p=0.5):
        super(MGNet, self).__init__()
        self.cnn = CNN()
        self.classifier = Classifier()
        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        cnn_data = self.cnn(x)
        cla_data = self.classifier(x)
        x = self.a * cnn_data + self.b * cla_data
        return x


if __name__ == '__main__':
    batch_size = 8
    channels = 1
    feature = 88
    input = torch.randn(batch_size, feature)
    cnn = MGNet()
    output = cnn(input)
    print(output.shape)

