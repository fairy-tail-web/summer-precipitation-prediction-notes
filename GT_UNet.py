import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .module import *

BATCH_NORM_DECAY = 1 - 0.9  # pytorch batch norm `momentum = 1 - counterpart` of tensorflow
BATCH_NORM_EPSILON = 1e-5

def get_act(activation):
    """Only supports ReLU and SiLU/Swish."""
    assert activation in ['relu', 'silu']
    if activation == 'relu':
        return nn.ReLU()
    else:
        return nn.Hardswish()  # TODO: pytorch's nn.Hardswish() v.s. tf.nn.swish


class BNReLU(nn.Module):
    """"""

    def __init__(self, out_channels, activation='relu', nonlinearity=True, init_zero=False):
        super(BNReLU, self).__init__()

        self.norm = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_DECAY, eps=BATCH_NORM_EPSILON)
        if nonlinearity:
            self.act = get_act(activation)
        else:
            self.act = None

        if init_zero:
            nn.init.constant_(self.norm.weight, 0)
        else:
            nn.init.constant_(self.norm.weight, 1)

    def forward(self, input):
        out = self.norm(input)
        if self.act is not None:
            out = self.act(out)
        return out


class RelPosSelfAttention(nn.Module):
    """Relative Position Self Attention"""

    def __init__(self, h, w, dim, relative=True, fold_heads=False):
        super(RelPosSelfAttention, self).__init__()
        self.relative = relative
        self.fold_heads = fold_heads
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)

    def forward(self, q, k, v):
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits += self.relative_logits(q)
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        # Relative logits in width dimension.
        rel_logits_w = self.relative_logits_1d(q, self.rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
        # Relative logits in height dimension
        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_emb_h,
                                               transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        bs, heads, h, w, dim = q.shape
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits = rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, length, 2*length - 1]
        Output: [bs, heads, length, length]
        """
        bs, heads, length, _ = x.shape
        col_pad = torch.zeros((bs, heads, length, 1), dtype=x.dtype).cuda()
        x = torch.cat([x, col_pad], dim=3)
        flat_x = torch.reshape(x, [bs, heads, -1]).cuda()
        flat_pad = torch.zeros((bs, heads, length - 1), dtype=x.dtype).cuda()
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        final_x = torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x = final_x[:, :, :length, length - 1:]
        return final_x


class AbsPosSelfAttention(nn.Module):

    def __init__(self, W, H, dkh, absolute=True, fold_heads=False):
        super(AbsPosSelfAttention, self).__init__()
        self.absolute = absolute
        self.fold_heads = fold_heads

        self.emb_w = nn.Parameter(torch.Tensor(W, dkh))
        self.emb_h = nn.Parameter(torch.Tensor(H, dkh))
        nn.init.normal_(self.emb_w, dkh ** -0.5)
        nn.init.normal_(self.emb_h, dkh ** -0.5)

    def forward(self, q, k, v):
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        abs_logits = self.absolute_logits(q)
        if self.absolute:
            logits += abs_logits
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def absolute_logits(self, q):
        """Compute absolute position enc logits."""
        emb_h = self.emb_h[:, None, :]
        emb_w = self.emb_w[None, :, :]
        emb = emb_h + emb_w
        abs_logits = torch.einsum('bhxyd,pqd->bhxypq', q, emb)
        return abs_logits


class GroupPointWise(nn.Module):
    """"""

    def __init__(self, in_channels, heads=4, proj_factor=1, target_dimension=None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels = target_dimension // proj_factor
        else:
            proj_channels = in_channels // proj_factor
        self.w = nn.Parameter(
            torch.Tensor(in_channels, heads, proj_channels // heads)
        )

        nn.init.normal_(self.w, std=0.01)

    def forward(self, input):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC
        input = input.permute(0, 2, 3, 1).float()
        """
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        out = torch.einsum('bhwc,cnp->bnhwp', input, self.w)
        return out


class MHSA(nn.Module):


    def __init__(self, in_channels, heads, curr_h, curr_w, pos_enc_type='relative', use_pos=True):
        super(MHSA, self).__init__()
        self.q_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.k_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.v_proj = GroupPointWise(in_channels, heads, proj_factor=1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type == 'relative':
            self.self_attention = RelPosSelfAttention(curr_h, curr_w, in_channels // heads, fold_heads=True)
        else:
            raise NotImplementedError

    def forward(self, input):
        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        o = self.self_attention(q=q, k=k, v=v)
        return o


class BotBlock(nn.Module):

    def __init__(self, in_dimension, curr_h, curr_w, proj_factor=4, activation='relu', pos_enc_type='relative',
                 stride=1, target_dimension=None):
        super(BotBlock, self).__init__()
        if stride != 1 or in_dimension != target_dimension:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dimension, target_dimension, kernel_size=3,padding=1, stride=stride),
                BNReLU(target_dimension, activation=activation, nonlinearity=True),
            )
        else:
            self.shortcut = None

        bottleneck_dimension = target_dimension // proj_factor
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size=3,padding=1, stride=1),
            BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True)
        )

        self.mhsa = MHSA(in_channels=bottleneck_dimension, heads=4, curr_h=curr_h, curr_w=curr_w,
                         pos_enc_type=pos_enc_type)
        conv2_list = []
        if stride != 1:
            assert stride == 2, stride
            conv2_list.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))  # TODO: 'same' in tf.pooling
        conv2_list.append(BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True))
        self.conv2 = nn.Sequential(*conv2_list)

        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_dimension, target_dimension, kernel_size=3,padding=1, stride=1),
            BNReLU(target_dimension, nonlinearity=False, init_zero=True),
        )
        self.last_act = get_act(activation)


    def forward(self, x):
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        Q_h = Q_w = 4
        N, C, H, W = x.shape
        P_h, P_w = H // Q_h, W // Q_w

        x = x.reshape(N * P_h * P_w, C, Q_h, Q_w)

        out = self.conv1(x)
        out = self.mhsa(out)
        out = out.permute(0, 3, 1, 2)  # back to pytorch dim order
        out = self.conv2(out)
        out = self.conv3(out)

        N1, C1, H1, W1 = out.shape
        out = out.reshape(N, C1, int(H), int(W))

        out += shortcut
        out = self.last_act(out)

        return out

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=False),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=False)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=False)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


def _make_bot_layer(ch_in, ch_out):

    W = H = 4
    dim_in = ch_in
    dim_out = ch_out

    stage5 = []

    stage5.append(
        BotBlock(in_dimension=dim_in, curr_h=H, curr_w=W, stride=1 , target_dimension=dim_out)
    )

    return nn.Sequential(*stage5)


# 在 models/GT_UNet.py 中

class GT_U_Net(nn.Module):
    # ✨ 修改: img_ch 将是 T*C，我们不再需要 input_len
    def __init__(self,
                 atmos_channels_in,
                 global_channels_in,
                 output_ch=1,
                 unet_start_dim=64):
        """
        Args:
            atmos_channels_in (int): 大气场输入通道数 (T_in * C_atmos)
            global_channels_in (int): 全局/背景输入通道数 (T_in * C_global)
            output_ch (int): 输出通道数 (通常为 1)
            unet_start_dim (int): U-Net 第一层的通道数 (例如 64)
        """
        super(GT_U_Net, self).__init__()

        D = unet_start_dim  # 64
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- 1. 大气场 (x) 编码器 ---
        # 它将 x 转换为 D, 2D, 4D, 8D 通道
        self.atmos_enc1 = conv_block(ch_in=atmos_channels_in, ch_out=D)
        self.atmos_enc2 = conv_block(ch_in=D, ch_out=D * 2)
        self.atmos_enc3 = conv_block(ch_in=D * 2, ch_out=D * 4)
        self.atmos_enc4 = conv_block(ch_in=D * 4, ch_out=D * 8)

        # --- 2. 全局 (g) 编码器 ---
        # 它将 g 转换为 D, 2D, 4D, 8D 通道
        self.global_enc1 = conv_block(ch_in=global_channels_in, ch_out=D)
        self.global_enc2 = conv_block(ch_in=D, ch_out=D * 2)
        self.global_enc3 = conv_block(ch_in=D * 2, ch_out=D * 4)
        self.global_enc4 = conv_block(ch_in=D * 4, ch_out=D * 8)

        # --- 3. 多尺度门控模块 ---
        # 您的 MultiScaleGatedAttn 模块在每个尺度上被实例化
        self.gate1 = MultiScaleGatedAttn(dim=D)
        self.gate2 = MultiScaleGatedAttn(dim=D * 2)
        self.gate3 = MultiScaleGatedAttn(dim=D * 4)
        self.gate4 = MultiScaleGatedAttn(dim=D * 8)

        # --- 4. 瓶颈 ---
        # RGA 模块现在处理来自第4层门控的融合特征
        self.bottleneck_rga = RGA(in_channel=D * 8, in_spatial=4 * 8)  # 假设 H/8, W/8 = 4x8

        # --- 5. 解码器 (与标准 U-Net 相同) ---
        self.Up4 = up_conv(ch_in=D * 8, ch_out=D * 4)
        self.Up_conv4 = conv_block(ch_in=D * 8, ch_out=D * 4)  # 256 + 256

        self.Up3 = up_conv(ch_in=D * 4, ch_out=D * 2)
        self.Up_conv3 = conv_block(ch_in=D * 4, ch_out=D * 2)  # 128 + 128

        self.Up2 = up_conv(ch_in=D * 2, ch_out=D)
        self.Up_conv2 = conv_block(ch_in=D * 2, ch_out=D)  # 64 + 64

        # -- 输出层 --
        self.Conv_1x1 = nn.Conv2d(D, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x_atmos, x_global):
        """
        Args:
            x_atmos (torch.Tensor): 大气场 (B, T_in, C_atmos, H, W)
            x_global (torch.Tensor): 全局特征 (B, T_in, C_global, H, W)
        Returns:
            torch.Tensor: 预测图 (B, output_ch, H, W)
        """
        # --- 0. 展平时间维度 ---
        # 将 (B, T, C, H, W) -> (B, T*C, H, W)
        B, T_a, C_a, H, W = x_atmos.shape
        x_atmos_flat = x_atmos.view(B, T_a * C_a, H, W)

        B, T_g, C_g, H, W = x_global.shape
        x_global_flat = x_global.view(B, T_g * C_g, H, W)

        # --- 编码器路径 ---

        # --- 全局 'g' 金字塔 ---
        # (计算所有 'g' 特征，以便 'x' 编码器可以使用它们)
        g1 = self.global_enc1(x_global_flat)  # [B, 64, H, W]
        g2 = self.global_enc2(self.Maxpool(g1))  # [B, 128, H/2, W/2]
        g3 = self.global_enc3(self.Maxpool(g2))  # [B, 256, H/4, W/4]
        g4 = self.global_enc4(self.Maxpool(g3))  # [B, 512, H/8, W/8]

        # --- 门控大气场 'x' 金字塔 ---

        # 级别 1
        a1_in = self.atmos_enc1(x_atmos_flat)
        x1 = self.gate1(a1_in, g1)  # [B, 64, H, W] <-- 跳跃连接 1

        # 级别 2
        a2_in = self.atmos_enc2(self.Maxpool(x1))
        x2 = self.gate2(a2_in, g2)  # [B, 128, H/2, W/2] <-- 跳跃连接 2

        # 级别 3
        a3_in = self.atmos_enc3(self.Maxpool(x2))
        x3 = self.gate3(a3_in, g3)  # [B, 256, H/4, W/4] <-- 跳跃连接 3

        # 级别 4
        a4_in = self.atmos_enc4(self.Maxpool(x3))
        x4 = self.gate4(a4_in, g4)  # [B, 512, H/8, W/8]

        # --- 瓶颈 ---
        bottle = self.bottleneck_rga(x4)

        # --- 解码器路径 ---
        # 使用来自门控编码器的融合跳跃连接 (x1, x2, x3)

        d4 = self.Up4(bottle)
        d4 = torch.cat((x3, d4), dim=1)  # 跳跃连接 x3
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)  # 跳跃连接 x2
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)  # 跳跃连接 x1
        d2 = self.Up_conv2(d2)

        # -- Final Output --
        d1 = self.Conv_1x1(d2)

        return d1