import torch
import torch.nn as nn
import torch.nn.functional as F

class RGA(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True,
                 cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGA, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial
        self.use_spatial = use_spatial
        self.use_channel = use_channel

        self.inter_channel = max(in_channel // cha_ratio, 1)
        self.inter_spatial = max(in_spatial // spa_ratio, 1)

        # --- 1. 原始特征的嵌入函数 (Embedding for Original Features) ---
        if self.use_spatial:
            # ✨ THIS IS THE MISSING BLOCK THAT HAS BEEN ADDED
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

        # --- 2. 关系特征的嵌入函数 (Embedding for Relational Features) ---
        if self.use_spatial:
            self.gg_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        # --- 3. 学习注意力权重的网络 (Network to Learn Attention Weights) ---
        if self.use_spatial:
            num_channel_s = 1 + self.inter_spatial
            # ✨ THIS IS THE CORRECT, SAFEGUARDED VERSION OF W_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=max(num_channel_s // down_ratio, 1),
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(max(num_channel_s // down_ratio, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=max(num_channel_s // down_ratio, 1), out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
        if self.use_channel:
            num_channel_c = max(1 + self.inter_channel, 1)
            self.W_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_c, out_channels=max(num_channel_c // down_ratio, 1),
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(max(num_channel_c // down_ratio, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=max(num_channel_c // down_ratio, 1), out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

        # --- 4. 用于建模关系的嵌入函数 (Embedding to Model Relations) ---
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

    def forward(self, x):
        b, c, h, w = x.size()

        if self.use_spatial:
            # 空间注意力
            # Q
            theta_xs = self.theta_spatial(x)
            theta_xs = theta_xs.view(b, self.inter_channel, -1)
            theta_xs = theta_xs.permute(0, 2, 1)
            # K
            phi_xs = self.phi_spatial(x)
            phi_xs = phi_xs.view(b, self.inter_channel, -1)
            Gs = torch.matmul(theta_xs, phi_xs)

            # 以光栅扫描顺序堆叠关系得到关系向量
            # 第一部分 cat
            Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)
            Gs_out = Gs.view(b, h * w, h, w)
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)
            Gs_joint = self.gg_spatial(Gs_joint)
            # 第二部分 cat
            g_xs = self.gx_spatial(x)
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)
            ys = torch.cat((g_xs, Gs_joint), 1)

            W_ys = self.W_spatial(ys)

            if not self.use_channel:
                out = torch.sigmoid(W_ys.expand_as(x)) * x
                return out
            else:
                x = torch.sigmoid(W_ys.expand_as(x)) * x

        if self.use_channel:
            # 通道注意力
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
            # Q
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
            # K
            phi_xc = self.phi_channel(xc).squeeze(-1)
            Gc = torch.matmul(theta_xc, phi_xc)

            # 以光栅扫描顺序堆叠关系得到关系向量
            # 第一部分 cat
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
            Gc_out = Gc.unsqueeze(-1)
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)
            Gc_joint = self.gg_channel(Gc_joint)
            # 第二部分 cat
            g_xc = self.gx_channel(xc)
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            yc = torch.cat((g_xc, Gc_joint), 1)

            W_yc = self.W_channel(yc).transpose(1, 2)
            out = torch.sigmoid(W_yc) * x
            return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalExtraction(nn.Module):
    """全局特征提取模块
    参数：
        dim：输入特征图的通道数
    """
    def __init__(self, dim=None):
        super(GlobalExtraction, self).__init__()
        self.avgpool = self.globalavgchannelpool
        self.maxpool = self.globalmaxchannelpool
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, 1, 1),
            nn.BatchNorm2d(1)
        )

    def globalavgchannelpool(self, x):
        """全局平均池化
        参数：
            x：输入特征图 [N, C, H, W]
        返回：
            全局平均池化特征 [N, 1, H, W]
        """
        x = x.mean(1, keepdim=True)  # 计算通道维度的平均值
        return x

    def globalmaxchannelpool(self, x):
        """全局最大池化
        参数：
            x：输入特征图 [N, C, H, W]
        返回：
            全局最大池化特征 [N, 1, H, W]
        """
        x = x.max(dim=1, keepdim=True)[0]  # 计算通道维度的最大值
        return x

    def forward(self, x):
        x_ = x.clone()
        x = self.avgpool(x)  # 通过平均池化提取全局特征
        x2 = self.maxpool(x_)  # 通过最大池化提取全局特征
        cat = torch.cat((x, x2), dim=1)  # 连接两种池化特征
        proj = self.proj(cat)  # 投影融合特征
        return proj

class ContextExtraction(nn.Module):
    """局部上下文特征提取模块
    参数：
        dim：输入特征图的通道数
        reduction：降维倍数
    """
    def __init__(self, dim, reduction=None):
        super(ContextExtraction, self).__init__()
        self.reduction = 1 if reduction is None else 2
        self.dconv = self.DepthWiseConv2dx2(dim)
        self.proj = self.Proj(dim)

    def DepthWiseConv2dx2(self, dim):
        """深度可分离卷积
        参数：
            dim：输入特征图的通道数
        返回：
            深度可分离卷积模块
        """
        dconv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(num_features=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(num_features=dim),
            nn.ReLU(inplace=True)
        )
        return dconv

    def Proj(self, dim):
        """特征降维
        参数：
            dim：输入特征图的通道数
        返回：
            特征降维模块
        """
        proj = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // self.reduction, kernel_size=1),
            nn.BatchNorm2d(num_features=dim // self.reduction)
        )
        return proj

    def forward(self, x):
        x = self.dconv(x)  # 提取局部上下文特征
        x = self.proj(x)  # 特征降维
        return x

class MultiscaleFusion(nn.Module):
    """多尺度特征融合模块
    参数：
        dim：输入特征图的通道数
    """
    def __init__(self, dim):
        super(MultiscaleFusion, self).__init__()
        self.local = ContextExtraction(dim)
        self.global_ = GlobalExtraction(dim)
        self.bn = nn.BatchNorm2d(num_features=dim)

    def forward(self, x, g):
        x = self.local(x)  # 提取局部上下文特征
        g = self.global_(g)  # 提取全局通道注意力特征
        fuse = self.bn(x + g)  # 特征融合并进行标准化
        return fuse

class MultiScaleGatedAttn(nn.Module):
    """多尺度门控注意力模块
    参数：
        dim：输入特征图的通道数
    """
    def __init__(self, dim):
        super(MultiScaleGatedAttn, self).__init__()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.bn_2 = nn.BatchNorm2d(dim)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1)
        )

    def forward(self, x, g):
        x_ = x.clone()  # 保存输入特征的副本
        g_ = g.clone()  # 保存门控特征的副本

        # 第一阶段：多尺度特征提取与融合
        multi = self.multi(x, g)  # 融合局部和全局特征

        # 第二阶段：自适应特征选择
        multi = self.selection(multi)  # 生成特征选择权重
        attention_weights = F.softmax(multi, dim=1)  # 权重归一化
        A, B = attention_weights.split(1, dim=1)  # 分离两个特征通道的权重
        x_att = A.expand_as(x_) * x_  # 应用特征选择权重到输入特征
        g_att = B.expand_as(g_) * g_  # 应用特征选择权重到门控特征
        x_att = x_att + x_  # 残差连接
        g_att = g_att + g_  # 残差连接

        # 第三阶段：特征交互与增强
        x_sig = torch.sigmoid(x_att)  # 生成输入特征的门控信号
        g_att_2 = x_sig * g_att  # 输入特征调制门控特征

        g_sig = torch.sigmoid(g_att)  # 生成门控特征的门控信号
        x_att_2 = g_sig * x_att  # 门控特征调制输入特征

        interaction = x_att_2 * g_att_2  # 特征交互融合

        # 第四阶段：特征重校准
        projected = torch.sigmoid(self.bn(self.proj(interaction)))  # 特征投影与归一化
        weighted = projected * x_  # 特征重校准
        y = self.conv_block(weighted)  # 最终特征提取
        y = self.bn_2(y)  # 输出特征归一化
        return y