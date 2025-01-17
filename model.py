import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.fft import fft2, ifft2
import torchvision.models as models
from resnetcifar import ResNet18_cifar10, ResNet50_cifar10


class Client_Model(nn.Module):
    def __init__(self, args, name):
        super().__init__()
        self.args = args
        self.name = name
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)
            
        if self.name == 'emnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, args.dims_feature)#args.dims_feature=100
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
         
        if self.name == 'mnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, args.dims_feature)#args.dims_feature=200
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
            
        if self.name == 'fmnist':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(32*4*4, 384) 
            self.fc2 = nn.Linear(384, args.dims_feature) #args.dims_feature=100
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
        
        if self.name == 'cifar10':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, args.dims_feature) #args.dims_feature=192
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
            
        if self.name == 'mixed_digit':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(1024, 384) 
            self.fc2 = nn.Linear(384, 100) #args.dims_feature=100
            self.classifier = nn.Linear(100, self.n_cls)
            
        if self.name == 'cifar100':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(1024, 384) 
            self.fc2 = nn.Linear(384, args.dims_feature) #args.dims_feature=192
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
            
        if self.name == 'resnet18':
            resnet18 = ResNet18_cifar10()
            resnet18.fc = nn.Linear(512, 512) 

            # Change BN to GN 
            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
            self.model = resnet18
            self.fc1 = nn.Linear(512, self.args.dims_feature) 
            self.classifier = nn.Linear(self.args.dims_feature, self.args.num_classes)
  
        if self.name == "Resnet50":#without replacement of bn layers by gn layers
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            self.fc1 = nn.Linear(2048, 512) 
            self.fc2 = nn.Linear(512, args.dims_feature) #args.dims_feature=192
            self.classifier = nn.Linear(args.dims_feature, self.args.num_classes)

    def forward(self, x):

        if self.name == 'Linear':
            x = self.fc(x)
        
        if self.name == 'mnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))  
            x = self.classifier(y_feature)
        
        if self.name == 'emnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            x = self.classifier(y_feature)
            
            
        if self.name == 'fmnist':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32*4*4)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            #y_feature = (self.fc2(x))
            x = self.classifier(y_feature)
        
        if self.name == 'cifar10':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            mfeature = x
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            #y_feature = self.fc2(x)
            x = self.classifier(y_feature)
            
        if self.name == 'mixed_digit':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            x = self.classifier(y_feature)

            
        if self.name == 'cifar100':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*32*32)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            x = self.classifier(y_feature)
            
        if self.name == "resnet18":
            x =  F.relu(self.model(x))
            y_feature = self.fc1(x)
            x = self.classifier(y_feature)

            
        if self.name == "Resnet50":
            x = self.features(x).squeeze().view(-1,2048)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            x = self.classifier(y_feature)
            

        return (mfeature, y_feature, x)
        #return (y_feature, x)


class RSmodel(nn.Module):
    def __init__(self):
        super(RSmodel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(144, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # No padding here
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # No padding here
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=1, stride=1),  # No padding here
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 15, kernel_size=1),
            nn.BatchNorm2d(15),
        )
        self.classifier = nn.Linear(15 * 1 * 1, 15)  # Adjust the input features to match the flattened size

    def forward(self, x, mefeatures):
        x = x.view(-1, 165, 7, 7)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        mefeatures = x
        features = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(features)
        return mefeatures, features, x
class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        #self.bn1 = nn.BatchNorm2d(64)
        self.gn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        #self.bn2 = nn.BatchNorm2d(64)
        self.gn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        #self.bn3 = nn.BatchNorm2d(128)
        self.gn3 = nn.GroupNorm(num_groups = 2, num_channels = 128)
    
        self.fc1 = nn.Linear(6272, 2048)
        #self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        #self.bn5 = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes)


    def forward(self, x):
        # x = F.relu(self.gn1(self.conv1(x)))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.gn2(self.conv2(x)))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.gn3(self.conv3(x)))
        # x = x.view(x.shape[0], -1)
        # x = F.relu(self.fc1(x))
        # # x = self.bn4(x)
        # features = F.relu(self.fc2(x))
        # x = self.classifier(features)
        
        x = F.relu((self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu((self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu((self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        x = self.classifier(features)
        
        return features, x



class Cross_fusion_CNN_avg(nn.Module):

    def __init__(self, input_channels, input_channels2, n_classes):
        super(Cross_fusion_CNN_avg, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)


        self.conv1_a = nn.Conv2d(input_channels, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_a = nn.BatchNorm2d(filters[0])
        self.conv2_a = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_a = nn.BatchNorm2d(filters[1])

        self.conv3_a = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_a = nn.BatchNorm2d(filters[2])
        self.conv4_a = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_a = nn.BatchNorm2d(filters[3])



        self.conv1_b = nn.Conv2d(input_channels2, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_b = nn.BatchNorm2d(filters[0])
        self.conv2_b = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_b = nn.BatchNorm2d(filters[1])

        self.conv3_b = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_b = nn.BatchNorm2d(filters[2])
        self.conv4_b = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_b = nn.BatchNorm2d(filters[3])


        self.conv5 = nn.Conv2d(filters[3] + filters[3], filters[3], (1, 1))
        self.bn5 = nn.BatchNorm2d(filters[3])
        self.conv6 = nn.Conv2d(filters[3], filters[2], (1, 1))
        self.bn6 = nn.BatchNorm2d(filters[2])
        self.fc = nn.Linear(64, 192)

        # self.conv7 = nn.Conv2d(filters[2], n_classes, (1, 1))

        ######################################################
        self.classifier = nn.Linear(filters[2], n_classes)
        ######################################################

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = x1.reshape(-1, 144, 7, 7)
        # print("x1",x1.shape)
        x2 = x2.reshape(-1, 21, 7, 7)
        # for image a
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        x1 = self.activation(self.bn1_a(self.conv1_a(x1)))
        x1 = self.activation(self.bn2_a(self.conv2_a(x1)))
        x1 = self.max_pool(x1)
        x1 = self.activation(self.bn3_a(self.conv3_a(x1)))


        # for image b
        x2 = self.activation(self.bn1_b(self.conv1_b(x2)))
        x2 = self.activation(self.bn2_b(self.conv2_b(x2)))
        x2 = self.max_pool(x2)
        x2 = self.activation(self.bn3_b(self.conv3_b(x2)))

        x11 = self.activation(self.bn4_a(self.conv4_a(x1)))
        x11 = self.max_pool(x11)
        x22 = self.activation(self.bn4_b(self.conv4_b(x2)))
        x22 = self.max_pool(x22)
        x12 = self.activation(self.bn4_b(self.conv4_b(x1)))
        x12 = self.max_pool(x12)
        x21 = self.activation(self.bn4_a(self.conv4_a(x2)))
        x21 = self.max_pool(x21)

        joint_encoder_layer1 = torch.cat([x11 + x21, x22 + x12], 1)
        joint_encoder_layer2 = torch.cat([x11, x12], 1)
        joint_encoder_layer3 = torch.cat([x22, x21], 1)


        fusion1 = self.activation(self.bn5(self.conv5(joint_encoder_layer1)))
        fusion1 = self.activation(self.bn6(self.conv6(fusion1)))
        fusion1 = self.avg_pool(fusion1)
        fusion1 = torch.flatten(fusion1, 1)  # Flatten for Linear layer
        y_features = fusion1
        fusion1 = self.classifier(y_features)

        fusion2 = self.activation(self.bn5(self.conv5(joint_encoder_layer2)))
        fusion2 = self.activation(self.bn6(self.conv6(fusion2)))
        fusion2 = self.avg_pool(fusion2)
        fusion2 = torch.flatten(fusion2, 1)  # Flatten for Linear layer
        fusion2 = self.classifier(fusion2)

        fusion3 = self.activation(self.bn5(self.conv5(joint_encoder_layer3)))
        fusion3 = self.activation(self.bn6(self.conv6(fusion3)))
        fusion3 = self.avg_pool(fusion3)
        fusion3 = torch.flatten(fusion3, 1)  # Flatten for Linear layer
        fusion3 = self.classifier(fusion3)




        return (y_features,fusion1, fusion2, fusion3)