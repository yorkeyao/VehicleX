from __future__ import absolute_import

from torch import nn
from torch.nn import init
from torchvision.models import resnet50, densenet121

'''
use global feat for testing
'''


class ZJU_model(nn.Module):
    def __init__(self, num_features=0, dropout=0, num_classes=0, norm=False, last_stride=2, output_feature='fc',
                 backbone='resnet50', BNneck=False):
        super(ZJU_model, self).__init__()
        # Create IDE_only model
        self.num_features = num_features
        self.num_classes = num_classes
        self.BNneck = BNneck

        if backbone == 'resnet50':
            # ResNet50: from 3*384*128 -> 2048*12*4 (Tensor T; of column vector f's)
            self.base = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
            if last_stride != 2:
                # decrease the downsampling rate
                # change the stride2 conv layer in self.layer4 to stride=1
                self.base[7][0].conv2.stride = last_stride
                # change the downsampling layer in self.layer4 to stride=1
                self.base[7][0].downsample[0].stride = last_stride
            base_channel = 2048
        elif backbone == 'densenet121':
            self.base = nn.Sequential(*list(densenet121(pretrained=True).children())[:-1])[0]
            if last_stride != 2:
                # remove the pooling layer in last transition block
                self.base[-3][-1].stride = 1
                self.base[-3][-1].kernel_size = 1
                pass
            base_channel = 1024
        else:
            raise Exception('Please select arch from [resnet50, densenet121]!')

        ################################################################################################################
        '''Global Average Pooling: 2048*12*4 -> 2048*1*1'''
        # Tensor T [N, 2048, 1, 1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        ################################################################################################################
        '''feat & feat_bn'''
        if not self.num_features:
            self.feature_fc = nn.Sequential(nn.BatchNorm1d(base_channel))
            self.feature_fc[0].bias.requires_grad_(False)  # no shift for BN
        else:
            self.feature_fc = nn.Sequential(nn.Linear(base_channel, self.num_features),
                                            nn.BatchNorm1d(self.num_features))
            init.kaiming_normal_(self.feature_fc[0].weight, mode='fan_out')
            init.constant_(self.feature_fc[0].bias, 0.0)
        init.constant_(self.feature_fc[-1].weight, 1)
        init.constant_(self.feature_fc[-1].bias, 0)
        # self.feature_fc.apply(weights_init_kaiming)

        # fc for softmax:
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features if self.num_features else base_channel,
                                        self.num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)
        pass

    def forward(self, x, eval_only=False):
        """
        Returns:
          h_s: each member with shape [N, c]
          prediction_s: each member with shape [N, num_classes]
        """
        # Tensor T [N, 2048, 12, 4]
        x = self.base(x)
        x = self.global_avg_pool(x).view(x.shape[0], -1)
        global_feat = x

        if self.BNneck:
            x = self.feature_fc(x)
        feat = x

        prediction_s = []
        if self.num_classes > 0 and not eval_only:
            prediction = self.classifier(x)
            prediction_s.append(prediction)
        if self.training:
            return global_feat if not self.num_features else feat, tuple(prediction_s)
        else:
            return feat if not self.num_features else global_feat, tuple(prediction_s)
