import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_planes,out_channels=places,kernel_size=8,stride=stride,padding=0, bias=False),
        nn.BatchNorm1d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=4, stride=1, padding=0)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=in_places,out_channels=places,kernel_size=7,stride=1, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=places, out_channels=places, kernel_size=7, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=11, stride=stride, bias=False),
                nn.BatchNorm1d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class DEEPPLUS(nn.Module):
    def __init__(self,blocks, num_classes=2, expansion = 4):
        super(DEEPPLUS,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 4, places= 40)

        self.layer1 = self.make_layer(in_places = 40, places= 40, block=blocks[0], stride=2)
        self.layer2 = self.make_layer(in_places = 160,places=80, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=320,places=160, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=640,places=320, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(47*1280,2000),nn.ReLU(),nn.Linear(2000,120),nn.ReLU(),nn.Linear(120,num_classes))
        #self.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(640*110,2000),nn.ReLU(),nn.Linear(2000,120),nn.ReLU(),nn.Linear(120,num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0),-1,2)
        x = F.softmax(x,dim=2)
        return x

