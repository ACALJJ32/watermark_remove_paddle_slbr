import math
import collections
from collections import OrderedDict
from itertools import repeat
import pickle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from arch_utils import UpConv, DownConv, MBEBlock, SMRBlock, CFFBlock, ResDownNew, ResUpNew, ECABlock, NAFNet, SimpleGate


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)


class CoarseEncoder(nn.Layer):
    def __init__(self, in_channels=3, depth=3, blocks=1, start_filters=32, residual=True, norm=nn.BatchNorm2D, act=F.relu, pooling = True):
        super(CoarseEncoder, self).__init__()
        self.down_convs = []
        outs = None
        
        if type(blocks) is tuple:
            blocks = blocks[0]
        
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters*(2**i)
            
            down_conv = DownConv(ins, outs, blocks, pooling=pooling, residual=residual, norm=norm, act=act)
            self.down_convs.append(down_conv)
        
        self.down_convs = nn.LayerList(self.down_convs)

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)
        return x, encoder_outs


class SharedBottleNeck(nn.Layer):
    def __init__(self, in_channels=512, depth=3, shared_depth=2, start_filters=32, blocks=1, residual=True,
                 concat=True,  norm=nn.BatchNorm2D, act=F.relu, dilations=[1,2,5]):
        super(SharedBottleNeck, self).__init__()
        self.down_convs = []
        self.up_convs = []
        self.down_im_atts = []
        self.down_mask_atts = []
        self.up_im_atts = []
        self.up_mask_atts = []

        dilations = [1,2,5]
        start_depth = depth - shared_depth
        max_filters = 512
        for i in range(start_depth, depth): # depth = 5 [0,1,2,3]
            ins = in_channels if i == start_depth else outs
            outs = min(ins * 2, max_filters)
            # Encoder convs
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, blocks, pooling=pooling, residual=residual, norm=norm, act=act, dilations=dilations)
            self.down_convs.append(down_conv)

            # Decoder convs
            if i < depth - 1:
                up_conv = UpConv(min(outs*2, max_filters), outs, blocks, residual=residual, concat=concat, norm=norm,act=F.relu, dilations=dilations)
                self.up_convs.append(up_conv)
                self.up_im_atts.append(ECABlock(outs))
                self.up_mask_atts.append(ECABlock(outs))
       
        self.down_convs = nn.LayerList(self.down_convs)
        self.up_convs = nn.LayerList(self.up_convs)

        # task-specific channel attention blocks
        self.up_im_atts = nn.LayerList(self.up_im_atts)
        self.up_mask_atts = nn.LayerList(self.up_mask_atts)

        # TODO
        # reset_params(self)

    def forward(self, input):
        # Encoder convs
        im_encoder_outs = []
        mask_encoder_outs = []
        x = input
        for i, d_conv in enumerate(self.down_convs):
            # d_conv, attn = nets
            x, before_pool = d_conv(x)
            im_encoder_outs.append(before_pool)
            mask_encoder_outs.append(before_pool)
        x_im = x
        x_mask = x

        # Decoder convs
        x = x_im
        for i, nets in enumerate(zip(self.up_convs, self.up_im_atts)):
            up_conv, attn = nets
            before_pool = None
            if im_encoder_outs is not None:
                before_pool = im_encoder_outs[-(i+2)]
            x = up_conv(x, before_pool,se=attn)
        x_im = x

        x = x_mask       
        for i, nets in enumerate(zip(self.up_convs, self.up_mask_atts)):
            up_conv, attn = nets
            before_pool = None
            if mask_encoder_outs is not None:
                before_pool = mask_encoder_outs[-(i+2)]
            x = up_conv(x, before_pool, se = attn)
        x_mask = x

        return x_im, x_mask


class CoarseDecoder(nn.Layer):
    def __init__(self, in_channels=512, out_channels=3, norm='bn',act=F.relu, depth=5, blocks=1, residual=True,
                 concat=True, use_att=False):
        super(CoarseDecoder, self).__init__()
        self.up_convs_bg = []
        self.up_convs_mask = []

        # apply channel attention to skip connection for different decoders
        self.atts_bg = []
        self.atts_mask = []
        self.use_att = use_att
        outs = in_channels
        for i in range(depth): 
            ins = outs
            outs = ins // 2
            # background reconstruction branch
            up_conv = MBEBlock(ins, outs, blocks=blocks, residual=residual, concat=concat, norm='in', act=act)
            self.up_convs_bg.append(up_conv)
            if self.use_att:
                self.atts_bg.append(ECABlock(outs))
            
            # mask prediction branch
            up_conv = SMRBlock(ins, outs, blocks=blocks, residual=residual, concat=concat, norm=norm, act=act)
            
            self.up_convs_mask.append(up_conv)
            if self.use_att:
                self.atts_mask.append(ECABlock(outs))

        # final conv
        self.conv_final_bg = nn.Conv2D(outs, out_channels, 1,1,0)
        
        self.up_convs_bg = nn.LayerList(self.up_convs_bg)
        self.atts_bg = nn.LayerList(self.atts_bg)
        self.up_convs_mask = nn.LayerList(self.up_convs_mask)
        self.atts_mask = nn.LayerList(self.atts_mask)
        
        # reset_params(self)

    def forward(self, bg, fg, mask, encoder_outs=None):
        bg_x = bg
        fg_x = fg
        mask_x = mask
        mask_outs = []
        bg_outs = []
        for i, up_convs in enumerate(zip(self.up_convs_bg, self.up_convs_mask)):
            up_bg, up_mask = up_convs
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+1)]

            if self.use_att:
                mask_before_pool = self.atts_mask[i](before_pool)
                bg_before_pool = self.atts_bg[i](before_pool)
            smr_outs = up_mask(mask_x, mask_before_pool)
            mask_x= smr_outs['feats'][0]
            primary_map, self_calibrated_map = smr_outs['attn_maps']
            mask_outs.append(primary_map)
            mask_outs.append(self_calibrated_map)

            
            bg_x = up_bg(bg_x, bg_before_pool, self_calibrated_map.detach())
            bg_outs.append(bg_x)

        if self.conv_final_bg is not None:
            bg_x = self.conv_final_bg(bg_x)
            mask_x = mask_outs[-1]
            bg_outs = [bg_x] + bg_outs
        return bg_outs, [mask_x] + mask_outs, None


################################################
#           Refinement Stage
################################################

class Refinement(nn.Layer):
    def __init__(self, in_channels=3, out_channels=3, shared_depth=2, down=ResDownNew, up=ResUpNew, ngf=32, n_cff=3, n_skips=1):
        super(Refinement, self).__init__()

        self.conv_in = nn.Sequential(nn.Conv2D(in_channels, ngf, 3,1,1), nn.InstanceNorm2D(ngf), nn.LeakyReLU(0.2))
        self.down1 = down(ngf, ngf)
        self.down2 = down(ngf, ngf)
        self.down3 = down(ngf, ngf, pooling=False, dilation=True)

        self.dec_conv2 = nn.Sequential(nn.Conv2D(ngf*1,ngf*1,1,1,0))
        self.dec_conv3 = nn.Sequential(nn.Conv2D(ngf*2,ngf*1,1,1,0), nn.LeakyReLU(0.2), nn.Conv2D(ngf, ngf, 3,1,1), nn.LeakyReLU(0.2))
        self.dec_conv4 = nn.Sequential(nn.Conv2D(ngf*4,ngf*2,1,1,0), nn.LeakyReLU(0.2), nn.Conv2D(ngf*2, ngf*2, 3,1,1), nn.LeakyReLU(0.2))
        self.n_skips = n_skips

        # CFF Blocks
        self.cff_blocks = []
        for i in range(n_cff):
            self.cff_blocks.append(CFFBlock(ngf=ngf))
        
        self.cff_blocks = nn.LayerList(self.cff_blocks)

        self.out_conv = nn.Sequential(*[
            nn.Conv2D(ngf * 3, ngf, 3,1,1),
            nn.InstanceNorm2D(ngf),
            nn.LeakyReLU(0.2),
            nn.Conv2D(ngf, out_channels, 1,1,0)
        ])     
        
    def forward(self, input, coarse_bg, mask, encoder_outs, decoder_outs):
        if self.n_skips < 1:
            dec_feat2 = 0
        else:
            dec_feat2 = self.dec_conv2(decoder_outs[0])
        if self.n_skips < 2:
            dec_feat3 = 0
        else:
            dec_feat3 = self.dec_conv3(decoder_outs[1]) # 64
        if self.n_skips < 3:
            dec_feat4 = 0
        else:
            dec_feat4 = self.dec_conv4(decoder_outs[2]) # 64

        xin = paddle.concat([coarse_bg, mask], axis=1)
        x = self.conv_in(xin)
        
        x,d1 = self.down1(x + dec_feat2) # 128,256
        x,d2 = self.down2(x + dec_feat3) # 64,128
        x,d3 = self.down3(x + dec_feat4) # 32,64

        xs = [d1,d2,d3]
        for block in self.cff_blocks:
            xs = block(xs)

        # xs = [F.interpolate(x_hr, size=coarse_bg.shape[2:][::-1], mode='bilinear') for x_hr in xs]
        xs = [F.interpolate(x_hr, size=coarse_bg.shape[2:], mode='bilinear') for x_hr in xs]
        im = self.out_conv(paddle.concat(xs,axis=1))
        return im


class SLBR(nn.Layer):
    def __init__(self, in_channels=3, depth=3, shared_depth=2, blocks=1,
                 out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True,
                 concat=True, long_skip=True):
        super(SLBR, self).__init__()
        self.shared = shared_depth = 2
        self.optimizer_encoder,  self.optimizer_image, self.optimizer_wm = None, None, None
        self.optimizer_mask, self.optimizer_shared = None, None
        
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks, blocks)

        # coarse stage
        self.encoder = CoarseEncoder(in_channels=in_channels, depth = depth - shared_depth, blocks=blocks[0],
                                    start_filters=start_filters, residual=residual, norm='bn',act=F.relu)

        self.shared_decoder = SharedBottleNeck(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                               depth=depth, shared_depth=shared_depth, blocks=blocks[4], residual=residual,
                                                concat=concat, norm='in')
        
        self.coarse_decoder = CoarseDecoder(in_channels=start_filters * 2 ** (depth - shared_depth),
                                        out_channels=out_channels_image, depth=depth - shared_depth,
                                        blocks=blocks[1], residual=residual, 
                                        concat=concat, norm='bn', use_att=True,
                                        )

        self.long_skip = long_skip
        self.tanh = nn.Tanh()
        
        self.refinement = Refinement(in_channels=4, out_channels=3, shared_depth=1)
        # self.refinement = None
        
        self.tanh = nn.Tanh()
    
    def spatial_padding(self, x):
        _, c, h, w = x.shape
        pad = 8

        pad_h = (pad - h % pad) % pad
        pad_w = (pad - w % pad) % pad
        
        # padding
        lrs = x.reshape_((-1, c, h, w))
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode="reflect")

        return lrs
        

    def forward(self, synthesized):
        b, c, h_input, w_input = synthesized.shape
        synthesized = self.spatial_padding(synthesized)
    
        image_code, before_pool = self.encoder(synthesized)
        unshared_before_pool = before_pool

        im, mask = self.shared_decoder(image_code)
        ims, mask, wm = self.coarse_decoder(im, None, mask, unshared_before_pool)
        im = ims[0]
        
        reconstructed_image = self.tanh(im)
        if self.long_skip:
            reconstructed_image = paddle.clip(reconstructed_image + synthesized, min=0, max=1)

        reconstructed_mask = mask[0]
        reconstructed_wm = wm
        
        if self.refinement is not None:
            dec_feats = (ims)[1:][::-1]
            coarser = reconstructed_image * reconstructed_mask + (1-reconstructed_mask)* synthesized
            
            refine_bg = self.refinement(synthesized, coarser, reconstructed_mask, None, dec_feats)
            refine_bg = paddle.clip(self.tanh(refine_bg) + synthesized, min=0, max=1)
            return reconstructed_image[:, :, :h_input, :w_input], mask, reconstructed_wm, refine_bg[:, :, :h_input, :w_input]
        else:
            return reconstructed_image[:, :, :h_input, :w_input], mask, reconstructed_wm


class SLBRNAF(nn.Layer):
    def __init__(self, in_channels=3, depth=3, shared_depth=2, blocks=1,
                 out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True,
                 concat=True, long_skip=True):
        super(SLBRNAF, self).__init__()
        self.shared = shared_depth = 2
        self.optimizer_encoder,  self.optimizer_image, self.optimizer_wm = None, None, None
        self.optimizer_mask, self.optimizer_shared = None, None
        
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks, blocks)
        
        # long skip
        self.feat_extractor_1 = nn.Conv2D(in_channels=in_channels, out_channels=4*in_channels, kernel_size=3, padding=1, stride=1, groups=1)
        self.feat_extractor_2 = nn.Conv2D(in_channels=4*in_channels, out_channels=2*in_channels, kernel_size=3, padding=1, stride=1, groups=1)
        self.feat_extractor_3 = nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0, stride=1, groups=1)
        self.sg = SimpleGate()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                      groups=1)
        )

        # coarse stage
        self.encoder = CoarseEncoder(in_channels=in_channels, depth = depth - shared_depth, blocks=blocks[0],
                                    start_filters=start_filters, residual=residual, norm='bn',act=F.relu)

        self.shared_decoder = SharedBottleNeck(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                               depth=depth, shared_depth=shared_depth, blocks=blocks[4], residual=residual,
                                                concat=concat, norm='in')
        
        self.coarse_decoder = CoarseDecoder(in_channels=start_filters * 2 ** (depth - shared_depth),
                                        out_channels=out_channels_image, depth=depth - shared_depth,
                                        blocks=blocks[1], residual=residual, 
                                        concat=concat, norm='bn', use_att=True,
                                        )

        self.long_skip = long_skip
        self.refinement = NAFNet(img_channel=3, width=16)
        self.tanh = nn.Tanh()
    
    def spatial_padding(self, x):
        _, c, h, w = x.shape
        pad = 8

        pad_h = (pad - h % pad) % pad
        pad_w = (pad - w % pad) % pad
        
        # padding
        lrs = x.reshape_((-1, c, h, w))
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode="reflect")

        return lrs
        

    def forward(self, synthesized):
        b, c, h_input, w_input = synthesized.shape
        synthesized = self.spatial_padding(synthesized)
        
        # extract feature
        feat = self.feat_extractor_2(self.feat_extractor_1(synthesized))
        feat = self.sg(feat)
        feat = feat * self.ca(feat)
        skip = self.feat_extractor_3(feat)
    
        image_code, before_pool = self.encoder(synthesized)
        unshared_before_pool = before_pool

        im, mask = self.shared_decoder(image_code)
        ims, mask, wm = self.coarse_decoder(im, None, mask, unshared_before_pool)
        im = ims[0]
        
        reconstructed_image = self.tanh(im)
        if self.long_skip:
            reconstructed_image = paddle.clip(reconstructed_image + synthesized, min=0, max=1)
        
        reconstructed_mask = mask[0]
        if self.refinement is not None:
            dec_feats = (ims)[1:][::-1]
            coarser = reconstructed_image * reconstructed_mask + (1-reconstructed_mask)* synthesized
            
            refine_bg = self.refinement(synthesized, coarser, reconstructed_mask, None, dec_feats)
            refine_bg = paddle.clip(self.tanh(refine_bg) + synthesized, min=0, max=1)
        
        reconstructed_image = self.tanh(im)
        return reconstructed_image[:, :, :h_input, :w_input], curr_mask, wm
        
        
