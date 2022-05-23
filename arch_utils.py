import math
import collections
from collections import OrderedDict
from itertools import repeat
import pickle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1, dilation=1):
    
    return nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation)


def up_conv3x3(in_channels, out_channels, transpose=True):
    if transpose:
        return nn.Conv2DTranspose(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv3x3(in_channels, out_channels))


class ECABlock(nn.Layer):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        # self.conv = nn.Conv1D(1, 1, kernel_size=1, padding=(k_size - 1) // 2) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        
        # TODO
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y = self.conv(y)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y
      

class UpConv(nn.Layer):
    def __init__(self, in_channels, out_channels, blocks, residual=True,norm=nn.BatchNorm2D, 
        act=F.relu, concat=True,use_att=False, use_mask=False, dilations=[], out_fuse=False):
        super(UpConv, self).__init__()
        self.concat = concat
        self.residual = residual
        self.conv2 = []
        self.use_att = use_att
        self.use_mask = use_mask
        
        self.out_fuse = out_fuse
        self.up_conv = up_conv3x3(in_channels, out_channels, transpose=False)
        if isinstance(norm, str):
            if norm == 'bn':
                norm = nn.BatchNorm2D
            elif norm == 'in':
                norm = nn.InstanceNorm2D
            else:
                raise TypeError("Unknown Type:\t{}".format(norm))
        self.norm0 = norm(out_channels)
        if len(dilations) == 0: dilations = [1] * blocks

        if self.concat:
            self.conv1 = conv3x3(2 * out_channels + int(use_mask), out_channels)
            self.norm1 = norm(out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
            self.norm1 = norm(out_channels)
        for i in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels, dilation=dilations[i], padding=dilations[i]))
        
        self.bn = []
        for _ in range(blocks):
            self.bn.append(norm(out_channels))
        self.bn = nn.LayerList(self.bn)
        self.conv2 = nn.LayerList(self.conv2)
        self.act = act

    def forward(self, from_up, from_down, mask=None,se=None):
        from_up = self.act(self.norm0(self.up_conv(from_up)))
        if self.concat:
            if self.use_mask:
                x1 = paddle.concat((from_up, from_down, mask), 1)
            else:
                x1 = paddle.concat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up
        
        xfuse = x1 = self.act(self.norm1(self.conv1(x1)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)
            
            #if (se is not None) and (idx == len(self.conv2) - 1): # last 
            #    x2 = se(x2)

            if self.residual:
                x2 = x2 + x1
                
            x2 = self.act(x2)
            x1 = x2
        if self.out_fuse:
            return x2, xfuse
        else:
            return x2


class DownConv(nn.Layer):
    def __init__(self, in_channels, out_channels, blocks, pooling=True, norm=nn.BatchNorm2D,act=F.relu,residual=True, dilations=[]):
        super(DownConv, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.pool = None
        self.conv1 = conv3x3(in_channels, out_channels)
        
        if isinstance(norm, str):
            if norm == 'bn':
                norm = nn.BatchNorm2D
            elif norm == 'in':
                norm = nn.InstanceNorm2D
            else:
                raise TypeError("Unknown Type:\t{}".format(norm))
        
        self.norm1 = norm(out_channels)
        
        if len(dilations) == 0: 
            dilations = [1] * blocks
        
        self.conv2 = []
        
        for i in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels, dilation=dilations[i], padding=dilations[i]))
       
        self.bn = []
        
        for _ in range(blocks):
            self.bn.append(norm(out_channels))
        self.bn = nn.LayerList(self.bn)
        
        if self.pooling:
            self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        
        self.conv2 = nn.LayerList(self.conv2)
        self.act = act

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x1 = self.act(self.norm1(self.conv1(x)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool


class MBEBlock(nn.Layer):
    def __init__(self, in_channels=512, out_channels=3, norm=nn.BatchNorm2D, act=F.relu, blocks=1, residual=True,
                  concat=True, is_final=True, mode='res_mask'):
        super(MBEBlock, self).__init__()
        self.concat = concat
        self.residual = residual
        self.mode = mode # vanilla, res_mask

        self.up_conv = up_conv3x3(in_channels, out_channels, transpose=False)

        if isinstance(norm, str):
            if norm == 'bn':
                norm = nn.BatchNorm2D
            elif norm == 'in':
                norm = nn.InstanceNorm2D
            else:
                raise TypeError("Unknown Type:\t{}".format(norm))
        self.norm0 = norm(out_channels)

        if self.concat:
            conv1_in = 2*out_channels
        else:
            conv1_in = out_channels
        self.conv1 = conv3x3(conv1_in, out_channels)
        self.norm1 = norm(out_channels)

        # residual structure
        self.conv2 = []
        self.conv3 = []
        for i in range(blocks):
            self.conv2.append(
                nn.Sequential(*[
                    nn.Conv2D(out_channels // 2 + 1, out_channels // 4, 5, 1, 2),
                    nn.ReLU(True),
                    nn.Conv2D(out_channels // 4, 1, 5, 1, 2),
                    nn.Sigmoid()
                ])
            )
            self.conv3.append(conv3x3(out_channels // 2, out_channels))
        
        self.bn = []
        for _ in range(blocks):
            self.bn.append(norm(out_channels))
        self.bn = nn.LayerList(self.bn)
        self.conv2 = nn.LayerList(self.conv2)
        self.conv3 = nn.LayerList(self.conv3)
        self.act = act

    def forward(self, from_up, from_down, mask=None):
        from_up = self.act(self.norm0(self.up_conv(from_up)))
        if self.concat:
            x1 = paddle.concat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up
        x1 = self.act(self.norm1(self.conv1(x1)))

        # residual structure
        _,C,H,W = x1.shape
        for idx, convs in enumerate(zip(self.conv2, self.conv3)):
            mask = convs[0](paddle.concat([x1[:,:C//2], mask], axis=1))
            x2_actv = x1[:,C//2:] * mask
            x2 = convs[1](x1[:,C//2:] + x2_actv)
            x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2        
        return x2
        

class SMRBlock(nn.Layer):
    def __init__(self, ins, outs, norm=nn.BatchNorm2D,act=F.relu, blocks=1, residual=True, concat=True):
        super(SMRBlock, self).__init__()
        self.threshold = 0.5
        self.upconv = UpConv(ins, outs, blocks, residual=residual, concat=concat, norm=norm, act=act, out_fuse=True)
        self.primary_mask = nn.Sequential(*[
            nn.Conv2D(outs,1,1,1,0),
            nn.Sigmoid()
        ])
        self.refine_branch = nn.Sequential(*[
            nn.Conv2D(outs,1,1,1,0),
            nn.Sigmoid()
        ])
        self.self_calibrated = SelfAttentionSimple(outs)

    def forward(self, input, encoder_outs=None):
        # upconv features
        mask_x, fuse_x = self.upconv(input, encoder_outs)
        primary_mask = self.primary_mask(mask_x)
        mask_x, self_calibrated_mask = self.self_calibrated(mask_x, mask_x, primary_mask)
        return {"feats":[mask_x], "attn_maps":[primary_mask, self_calibrated_mask]}


class SelfAttentionSimple(nn.Layer):
    def __init__(self, in_channel, k_center=1, sim_metric='fc', project_mode='linear'):
        super(SelfAttentionSimple, self).__init__()
        self.k_center = k_center # 1 for foreground, 2 for background & foreground
        self.reduction = 1
        self.project_mode = project_mode
        self.q_conv = nn.Conv2D(in_channel, in_channel, 1,1,0)
        self.k_conv = nn.Conv2D(in_channel, in_channel*k_center, 1,1,0)
        self.v_conv = nn.Conv2D(in_channel, in_channel*k_center, 1,1,0)
   
        self.min_area = 100
        self.threshold = 0.5
        self.out_conv = nn.Sequential(
            nn.Conv2D(in_channel, in_channel//8, 3,1,1),
            nn.ReLU(True),
            nn.Conv2D(in_channel//8, 1, 3,1,1)
        ) 
        
        self.sim_func = nn.Conv2D(in_channel + in_channel, 1, 1, 1, 0)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def compute_attention(self, query, key, mask, eps=1):  
        # key [b, c, h, w]
        # mask  [b, 1, h, w]
        b,c,h,w = query.shape
        
        query = self.q_conv(query) # query [b, c, h, w]
        key_in = key
        
        key = self.k_conv(key_in)  # [b, c, h, w]   e.g. [8, 128, 128, 128]
        keys = list(key.split(c,axis=1))  # keys[0]: [8, 1, 128, 128]
        
        importance_map = paddle.where(mask >= self.threshold, paddle.ones_like(mask), paddle.zeros_like(mask))
        s_area = paddle.clip(paddle.sum(importance_map, axis=[2,3]), self.min_area)[:,0:1]
        
        if self.k_center != 2:
            keys = [paddle.sum(k*importance_map, axis=[2,3]) / s_area for k in keys] # b,c * k
        else:
            keys = [
                paddle.sum(keys[0]*importance_map, axis=[2,3]) / s_area,
                paddle.sum(keys[1]*(1-importance_map), axis=[2,3]) / (keys[1].shape[2]*keys[1].shape[3] - s_area + eps)
            ]

        f_query = query
        
        f_key = [paddle.reshape(k, (b,1,1,1)) for k in keys]    
        f_key = [paddle.tile(k, [1, c, h, w]) for k in f_key]
        
        attention_scores = []
        
        for k in f_key:
            combine_qk = paddle.concat([f_query, k],axis=1) # tanh
            sk = self.tanh(self.sim_func(combine_qk))
            attention_scores.append(sk)
            
        s = torch.cat(attention_scores, dim=1) # b,k,h,w
        s = s.permute(0,2,3,1) # b,h,w,k
        
        v = self.v_conv(key_in)
        if self.k_center == 2:
            v_fg = torch.sum(v[:,:c]*importance_map, dim=[2,3]) / s_area
            v_bg = torch.sum(v[:,c:]*(1-importance_map), dim=[2,3]) / (v.shape[2]*v.shape[3] - s_area + eps)
            v = torch.cat([v_fg, v_bg],dim=1)
        else:
            v = torch.sum(v*importance_map, dim=[2,3]) / s_area # b, c*k
        v = v.reshape(b, self.k_center, c) # b, k, c
        attn = torch.bmm(s.reshape(b,h*w,self.k_center), v).reshape(b,h,w,c).permute(0,3,1,2)
        
        s = self.out_conv(attn + query)
        return s


    def forward(self, xin, xout, xmask):
        b_num,c,h,w = xin.shape
        
        # attention_score = self.compute_attention(xin, xout, xmask) 
        # attention_score = attention_score.reshape(b_num,1,h,w)
        # return xout, attention_score.sigmoid()
        
        return xout, self.sigmoid(xmask)


## Refinement Stage
class ResDownNew(nn.Layer):
    def __init__(self, in_size, out_size, pooling=True, use_att=False, dilation=False):
        super(ResDownNew, self).__init__()
        self.model = DownConv(in_size, out_size, 3, pooling=pooling, norm=nn.InstanceNorm2D, act=F.leaky_relu, dilations=[1,2,5] if dilation else [])

    def forward(self, x):
        return self.model(x)

class ResUpNew(nn.Layer):
    def __init__(self, in_size, out_size, use_att=False):
        super(ResUpNew, self).__init__()
        self.model = UpConv(in_size, out_size, 3, use_att=use_att, norm=nn.InstanceNorm2d)

    def forward(self, x, skip_input, mask=None):
        return self.model(x,skip_input,mask)


        
class CFFBlock(nn.Layer):
    def __init__(self, down=ResDownNew, up=ResUpNew, ngf = 32):
        super(CFFBlock, self).__init__()
        self.down1 = down(ngf, ngf)
        self.down2 = down(ngf, ngf)
        self.down3 = down(ngf, ngf, pooling=False, dilation=True)

        self.conv22 = nn.Sequential(*[
            nn.Conv2D(ngf, ngf, 3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2D(ngf, ngf, 3,1,1),
            nn.LeakyReLU(0.2)
        ])

        self.conv33 = nn.Sequential(*[
            nn.Conv2D(ngf,ngf, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2D(ngf, ngf, 3,1,1),
            nn.LeakyReLU(0.2)
        ])

        self.up32 = nn.Sequential(*[
            nn.Conv2D(ngf, ngf, 3,1,1),
            nn.LeakyReLU(0.2),
        ])

        self.up31 = nn.Sequential(*[
            nn.Conv2D(ngf, ngf, 3,1,1),
            nn.LeakyReLU(0.2),
        ])

    def forward(self, inputs):
        x1,x2,x3 = inputs
        # x32 = F.interpolate(x3, size=x2.shape[2:][::-1], mode='bilinear')
        x32 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear')
        x32 = self.up32(x32)
        
        # x31 = F.interpolate(x3, size=x1.shape[2:][::-1], mode='bilinear')
        x31 = F.interpolate(x3, size=x1.shape[2:], mode='bilinear')
        x31 = self.up31(x31)

        # cross-connection
        x,d1 = self.down1(x1 + x31)
        x,d2 = self.down2(x + self.conv22(x2) + x32)
        d3,_ = self.down3(x + self.conv33(x3))
        return [d1,d2,d3]
        

class SimpleGate(nn.Layer):
    def forward(self, x):
        x1, x2 = paddle.chunk(x, chunks=2, axis=1)
        return x1 * x2


class NAFBlock(nn.Layer):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2D(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv2 = nn.Conv2D(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel)
        self.conv3 = nn.Conv2D(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1)
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2D(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv5 = nn.Conv2D(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1)

        self.norm1 = nn.BatchNorm2D(c)
        self.norm2 = nn.BatchNorm2D(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x

        x = self.conv4(self.norm2(y))
        # x = self.conv4(y)
        
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x


class NAFNet(nn.Layer):
    def __init__(self, img_channel=3, width=16, middle_blk_num=3, enc_blk_nums=[1,1], dec_blk_nums=[1,1], mask_width=3):
        super().__init__()

        self.intro = nn.Conv2D(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1)
        self.ending = nn.Conv2D(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1)

        self.encoders = nn.LayerList()
        self.decoders = nn.LayerList()
        self.middle_blks = nn.LayerList()
        self.ups = nn.LayerList()
        self.downs = nn.LayerList()
        
        self.sg = SimpleGate()
        self.mask_intro = nn.Conv2D(in_channels=mask_width, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels=width // 2, out_channels=width // 2, kernel_size=1, padding=0, stride=1,
                      groups=1)
        )
        self.mask_conv = nn.Conv2D(in_channels=width // 2, out_channels=width, kernel_size=5, padding=2, stride=1, groups=1)
        self.fusion = nn.Conv2D(in_channels=width * 2, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1)

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2D(chan, chan, 2, 2)
            )
            chan = chan

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2D(chan, chan * 4, 1),
                    nn.PixelShuffle(2)
                )
            )
            # chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, inp, mask):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        
        mask_feat = self.mask_intro(mask)
        mask_feat = self.sg(mask_feat)
        mask_feat = mask_feat * self.ca(mask_feat)
        mask_feat = self.mask_conv(mask_feat)
        
        x = self.lrelu(self.fusion(paddle.concat([x, mask_feat], axis=1)))
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


