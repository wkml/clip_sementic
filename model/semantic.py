import torch.nn as nn
import torch.nn.functional as F
import torch

class semantic(nn.Module):
    def __init__(self, num_classes, image_feature_dim, word_feature_dim, intermediary_dim=1024):
        super(semantic, self).__init__()
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim
        self.word_feature_dim = word_feature_dim
        self.intermediary_dim = intermediary_dim
        self.fc_a = nn.Linear(self.intermediary_dim, 1)

    def forward(self, batch_size, img_feature_map, word_features):
        convsize = img_feature_map.size()[3]
        img_feature_map = img_feature_map.permute(0, 2, 3, 1)
        # img_feature_map = torch.transpose(torch.transpose(img_feature_map, 1, 2), 2, 3)
        # 展平每一个图像块的特征
        f_wh_feature = img_feature_map.reshape(batch_size * convsize * convsize, -1)
        # 对每一个图像块的特征做仿射变换，每一个块对每一个类都有一个特征  [block_count, num_classes, dim]
        f_wh_feature = f_wh_feature.reshape(batch_size * convsize * convsize, 1, -1).repeat(1, self.num_classes,1)

        # V 把每一个类的特征维度映射为1024，并对应每一个图像块复制 [block_count, num_classes, dim] 每一个块对每一个类都有一个特征
        f_wd_feature = word_features.reshape(1, self.num_classes, 1024).repeat(batch_size * convsize * convsize, 1, 1)

        # P 将每一个块对每一个类的特征展平 [ block_count * num_classes, dim]
        lb_feature = torch.tanh(f_wh_feature * f_wd_feature).reshape(-1, 1024)
        # attention coefficient 对每一个特征做pooling，转化为一个置信度
        coefficient = self.fc_a(lb_feature)
        coefficient = coefficient.reshape(batch_size, convsize, convsize, self.num_classes).permute(0, 3, 1, 2).reshape(batch_size, self.num_classes, -1)
        coefficient = F.softmax(coefficient, dim=2)

        coefficient = coefficient.reshape(batch_size, self.num_classes, convsize, convsize)
        # coefficient = torch.transpose(torch.transpose(coefficient, 1, 2), 2, 3)
        coefficient = coefficient.permute(0, 2, 3, 1)
        coefficient = coefficient.reshape(batch_size, convsize, convsize, self.num_classes, 1).repeat(1, 1, 1, 1,
                                                                                                   self.image_feature_dim)
        # weighted average pooling 给每一个块打一个权重
        img_feature_map = img_feature_map.reshape(batch_size, convsize, convsize, 1, self.image_feature_dim).repeat(1, 1, 1, self.num_classes, 1) * coefficient
        # fc
        graph_net_input = torch.sum(torch.sum(img_feature_map, 1), 1)
        return graph_net_input