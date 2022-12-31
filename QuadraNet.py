from functools import partial
from numpy import zeros
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.fft

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


class QuadraConv_GF(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 1 , padding = 1 , dilation = 1 , groups =1 , bias = True):
        super(QuadraConv_GF, self).__init__()
        self.conv1 = GlobalLocalFilter(dim = in_channel)
        self.conv2 = GlobalLocalFilter(dim = in_channel)
        self.conv3 = GlobalLocalFilter(dim = in_channel)
        #self.conv4 = nn.Conv2d(in_channel, out_channel, kernel_size , stride, padding, dilation, groups, bias)
        #self.conv5 = nn.Conv2d(in_channel, out_channel, kernel_size , stride, padding, dilation, groups, bias)
        #self.conv6 = nn.Conv2d(in_channel, out_channel, kernel_size , stride, padding, dilation, groups, bias)
#        self.bn1 = nn.BatchNorm2d(out_channel)
#        self.relu = nn.ReLU()
        #self.bn2 = nn.BatchNorm2d(out_channel)
        #self.bn3 = nn.BatchNorm2d(out_channel)
        #self.bn4 = nn.BatchNorm2d(out_channel)
        #self.bn5 = nn.BatchNorm2d(out_channel)
        #self.bn6 = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        x_a = self.conv1(x)
        x_b = self.conv2(x)
        x_2 = torch.mul(x_a,x_b)
        x_1 = self.conv3(x)
        x = x_2 + x_1
#        x = self.relu(x)
#        x = self.bn1(x)

        return x


class QuadraConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 1 , padding = 1 , dilation = 1 , groups =1 , bias = True):
        super(QuadraConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size , stride, padding, dilation, groups, bias)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size , stride, padding, dilation, groups, bias)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size , stride, padding, dilation, groups, bias)
        #self.conv4 = nn.Conv2d(in_channel, out_channel, kernel_size , stride, padding, dilation, groups, bias)
        #self.conv5 = nn.Conv2d(in_channel, out_channel, kernel_size , stride, padding, dilation, groups, bias)
        #self.conv6 = nn.Conv2d(in_channel, out_channel, kernel_size , stride, padding, dilation, groups, bias)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        #self.bn2 = nn.BatchNorm2d(out_channel)
        #self.bn3 = nn.BatchNorm2d(out_channel)
        #self.bn4 = nn.BatchNorm2d(out_channel)
        #self.bn5 = nn.BatchNorm2d(out_channel)
        #self.bn6 = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        x_a = self.conv1(x)
        x_b = self.conv2(x)
        x_2 = torch.mul(x_a,x_b)
        x_1 = self.conv3(x)
        x = x_2 + x_1
        x = self.relu(x)
        x = self.bn1(x)

        return x

class QuadraConv_dw(nn.Module):
    def __init__(self, in_channel, kernel_size = 7, bias = True):
        super(QuadraConv_dw, self).__init__()
        self.conv1 = get_dwconv(dim = in_channel, kernel= kernel_size, bias = bias)
        self.conv2 = get_dwconv(dim = in_channel, kernel= kernel_size, bias = bias)
        self.conv3 = get_dwconv(dim = in_channel, kernel= kernel_size, bias = bias)

    def forward(self,x):
        x_a = self.conv1(x)
        x_b = self.conv2(x)
        x_2 = torch.mul(x_a,x_b)
        x_1 = self.conv3(x)
        x = x_2 + x_1

        return x

class QuadraConv_dw21(nn.Module):
    def __init__(self, in_channel, kernel_size = 21, bias = True):
        super(QuadraConv_dw21, self).__init__()
        self.conv1 = get_dwconv(dim = in_channel, kernel= kernel_size, bias = bias)
        self.conv2 = get_dwconv(dim = in_channel, kernel= kernel_size, bias = bias)
        self.conv3 = get_dwconv(dim = in_channel, kernel= kernel_size, bias = bias)

    def forward(self,x):
        x_a = self.conv1(x)
        x_b = self.conv2(x)
        x_2 = torch.mul(x_a,x_b)
        x_1 = self.conv3(x)
        x = x_2 + x_1

        return x

class QuadraFC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(QuadraFC,self).__init__()
        self.linear1 = nn.Linear(in_channel, out_channel)
        self.linear2 = nn.Linear(in_channel, out_channel)
        self.linear3 = nn.Linear(in_channel, out_channel)

    def forward(self,x):
        x_a = self.linear1(x)
        x_b = self.linear2(x)
        x_2 = torch.mul(x_a, x_b)
        x_1 = self.linear3(x)
        x = x_2 + x_1

        return x

class PolyConv(nn.Module):
    def __init__(self, in_channel, out_channel, order = 3 , kernel_size = 3, stride = 1 , padding = 1 , dilation = 1 , groups =1 , bias = True):
        super(PolyConv,self).__init__()
        self.order = order
        self.convlist = nn.ModuleList(
            [nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias) for i in range(order)]
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = torch.zeros_like(self.convlist[0](x))
        for i in range(self.order):
            x_i = self.convlist[i](x)
            y_i = torch.mul(y, x_i)
            y = y_i + x_i
        y = self.relu(y)
        y = self.bn(y)
        
        return y


    
class PolyFC(nn.Module):
    def __init__(self, in_size, out_size, order):
        super(PolyFC).__init__()
        self.order = order
        self.FClist = nn.ModuleList(
            [nn.linear(in_size, out_size) for i in range(order)]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = torch.zeros_like(self.FClist[0](x))
        for i in range(self.order):
            x_i = self.FClist[i](x)
            y_i = torch.mul(y, x_i)
            y = y_i + x_i
        y = self.relu(y)

        return y

class QuadraConvMixer(nn.Module):
    def __init__(self, dim, drop_path = 0. ,layer_scale_init_value=1e-6, QuadraConv=QuadraConv_dw):
        super(QuadraConvMixer,self).__init__()
        self.dwconv = get_dwconv(dim = dim, kernel = 3, bias= True)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.pwconv = QuadraFC(dim,dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.norm1(self.act1(self.QuadraConv(x))))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv(x)
        x = self.act(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.norm2(x)

        x = input + self.drop_path(x)

        


        
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class QuadraBlock1(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, QuadraConv=QuadraConv):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.QuadraConv = QuadraConv(dim,dim) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.QuadraConv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class QuadraBlock1_GF(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, QuadraConv=QuadraConv_GF):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.QuadraConv = QuadraConv(dim,dim) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.QuadraConv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class QuadraBlock1_dw(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, QuadraConv=QuadraConv_dw):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.QuadraConv = QuadraConv(dim) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.QuadraConv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Block1_dw(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, QuadraConv=nn.Conv2d):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.QuadraConv = QuadraConv(dim,dim,kernel_size=7,padding=3,groups=dim) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.QuadraConv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class QuadraBlock2_dw(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, QuadraConv=QuadraConv_dw):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.QuadraConv = QuadraConv(dim) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.QuadraConv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class QuadraBlock1_dw_fc(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, QuadraConv=QuadraConv_dw):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.QuadraConv = QuadraConv(dim) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = QuadraFC(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.QuadraConv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class QuadraBlock2(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, QuadraConv=QuadraConv, QuadraFC = QuadraFC):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.QuadraConv = QuadraConv(dim) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = QuadraFC(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = QuadraFC(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.QuadraConv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class QuadraBlock3(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, QuadraConv=QuadraConv, QuadraFC = QuadraFC):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.QuadraConv = QuadraConv(dim) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
#        x = x + self.drop_path(gamma1 * self.QuadraConv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class PolyBlock1(nn.Module):
    def __init__(self, dim, drop_path=0., order = 3,layer_scale_init_value=1e-6, PolyConv=PolyConv):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.PolyConv = PolyConv(dim,dim,order) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.PolyConv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class QuadraNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], base_dim=96, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 QuadraConv = QuadraConv, block=QuadraBlock1, uniform_init=False, **kwargs
                 ):
        super().__init__()
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 


        if not isinstance(QuadraConv, list):
            QuadraConv = [QuadraConv, QuadraConv, QuadraConv, QuadraConv]
        else:
            QuadraConv = QuadraConv
            assert len(QuadraConv) == 4

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, QuadraConv=QuadraConv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.uniform_init = uniform_init

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)            

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            for j, blk in enumerate(self.stages[i]):
                    x = blk(x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class QuadraNet2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], base_dim=96, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 QuadraConv = QuadraConv, block=QuadraBlock1, uniform_init=False, **kwargs
                 ):
        super().__init__()
        dims = [40, 80, 128, 256]

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 


        if not isinstance(QuadraConv, list):
            QuadraConv = [QuadraConv, QuadraConv, QuadraConv, QuadraConv]
        else:
            QuadraConv = QuadraConv
            assert len(QuadraConv) == 4

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, QuadraConv=QuadraConv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.uniform_init = uniform_init

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)            

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            for j, blk in enumerate(self.stages[i]):
                    x = blk(x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class PolyNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], base_dim=96, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 PolyConv = PolyConv, block=PolyBlock1, uniform_init=False, **kwargs
                 ):
        super().__init__()
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 


        if not isinstance(PolyConv, list):
            PolyConv = [PolyConv, PolyConv, PolyConv, PolyConv]
        else:
            PolyConv = PolyConv
            assert len(PolyConv) == 4

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, PolyConv=PolyConv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.uniform_init = uniform_init

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)            

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            for j, blk in enumerate(self.stages[i]):
                    x = blk(x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
            
            
@register_model
def QuadraNet_tiny_GF(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 3, 18, 2], base_dim=64, block=QuadraBlock1_GF,
    QuadraConv=[
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_tiny_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 18, 2], base_dim=64, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_nas_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 18, 2], base_dim=64, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_tiny_dw2(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 3, 18, 2], base_dim=64, block=QuadraBlock2_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model


@register_model
def QuadraNet_tiny_dw_fc(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[3, 3, 27, 3], base_dim=64, block=QuadraBlock1_dw_fc,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_tiny36_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[3, 3, 27, 3], base_dim=64, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_xs36_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[3, 3, 27, 3], base_dim=32, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def Net_xs36_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[3, 3, 27, 3], base_dim=32, block=Block1_dw,
    QuadraConv=[
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
    ],
    **kwargs
    )
    return model

@register_model
def Net_xxs36_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[3, 3, 27, 3], base_dim=16, block=Block1_dw,
    QuadraConv=[
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
    ],
    **kwargs
    )
    return model

@register_model
def Net_tiny36_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[3, 3, 27, 3], base_dim=64, block=Block1_dw,
    QuadraConv=[
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
    ],
    **kwargs
    )
    return model

@register_model
def Net_tiny12_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 6, 2], base_dim=64, block=Block1_dw,
    QuadraConv=[
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
    ],
    **kwargs
    )
    return model

@register_model
def Net_base_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 18, 2], base_dim=128, block=Block1_dw,
    QuadraConv=[
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
    ],
    **kwargs
    )
    return model

@register_model
def Net_xs12_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 6, 2], base_dim=32, block=Block1_dw,
    QuadraConv=[
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
    ],
    **kwargs
    )
    return model

@register_model
def Net_xxs12_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 6, 2], base_dim=16, block=Block1_dw,
    QuadraConv=[
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
        partial(nn.Conv2d),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_xxs36_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[3, 3, 27, 3], base_dim=16, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_supertiny12_32_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 6, 2], base_dim=32, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_supertiny12_16_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 6, 2], base_dim=16, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_supertiny12_64_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 6, 2], base_dim=64, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_tiny(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 3, 18, 2], base_dim=64, block=QuadraBlock1,
    QuadraConv=[
        partial(QuadraConv),
        partial(QuadraConv),
        partial(QuadraConv),
        partial(QuadraConv),
    ],
    **kwargs
    )
    return model

@register_model
def PolyNet_tiny(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = PolyNet(depths=[2, 3, 18, 2], base_dim=64, block=PolyBlock1,
    QuadraConv=[
        partial(PolyConv, order = 3),
        partial(PolyConv, order = 3),
        partial(PolyConv, order = 3),
        partial(PolyConv, order = 3),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_small_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 3, 18, 2], base_dim=96, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_small_gf(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 3, 18, 2], base_dim=96, block=QuadraBlock1_GF,
    QuadraConv=[
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_base_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 18, 2], base_dim=128, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def FNNet_base_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 18, 2], base_dim=128, block=QuadraBlock3,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_base_dw36(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[3, 3, 27, 3], base_dim=128, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_base_dw_fc(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 3, 18, 2], base_dim=128, block=QuadraBlock1_dw_fc,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_base_dw21(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 3, 18, 2], base_dim=128, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw21),
        partial(QuadraConv_dw21),
        partial(QuadraConv_dw21),
        partial(QuadraConv_dw21),
    ],
    **kwargs
    )
    return model



@register_model
def QuadraNet_base_gf(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 2, 18, 2], base_dim=128, block=QuadraBlock1_GF,
    QuadraConv=[
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_base_gf_img384(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 3, 18, 2], base_dim=128, block=QuadraBlock1_GF,
    QuadraConv=[
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_large_dw(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[3, 3, 27, 3], base_dim=192, block=QuadraBlock1_dw,
    QuadraConv=[
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
        partial(QuadraConv_dw),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_large_gf(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 3, 18, 2], base_dim=192, block=QuadraBlock1_GF,
    QuadraConv=[
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
    ],
    **kwargs
    )
    return model

@register_model
def QuadraNet_large_gf_img384(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = QuadraNet(depths=[2, 3, 18, 2], base_dim=192, block=QuadraBlock1_GF,
    QuadraConv=[
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
        partial(QuadraConv_GF),
    ],
    **kwargs
    )
    return model

            



       


        


