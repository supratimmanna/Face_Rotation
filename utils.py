import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
import numpy as np


####################################################

def weigth_initialization(conv_weigths, init):
    if (init == 'xavier_normal'):
        nn.init.xavier_normal_(conv_weigths)
    if (init == 'xavier_uniform'):
        nn.init.xavier_uniform_(conv_weigths)
    return


##################### conv2d ###########################
def conv(input_channels, 
         output_channels,
         kernel_size, stride, 
         padding, weigth_init = 'xavier_normal', 
         batch_norm = False,
         activation=nn.ReLU()):
    a = []
    a.append(nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    weigth_initialization(a[-1].weight, weigth_init)
    if activation is not None:
        a.append(activation)
    if batch_norm:
        a.append(nn.BatchNorm2d(output_channels))
    return nn.Sequential(*a)


##################### deconv2d ###########################
def deconv(input_channels, 
         output_channels,
         kernel_size, stride, 
         padding, weigth_init = 'xavier_normal', 
         batch_norm = False,
         activation=nn.ReLU()):
    a = []
    a.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    weigth_initialization(a[-1].weight, weigth_init)
    if activation is not None:
        a.append(activation)
    if batch_norm:
        a.append(nn.BatchNorm2d(output_channels))
    return nn.Sequential(*a)


##################### Residual Block ###########################
class ResidualBlock(nn.Module):
    def __init__(self, input_channels,
                output_channels = None, 
                kernel_size = 3, stride = 1, 
                padding = None, weight_init = 'xavier_normal', 
                batch_norm = False,
                activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        if output_channels is None:
            output_channels = input_channels // stride
        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.conv(input_channels, output_channels, 1, stride, 0, None, False, None)
            
        a = []
        a.append( conv( input_channels , input_channels  , kernel_size , 1 , padding if padding is not None else (kernel_size - 1)//2 , weight_init ,  False, activation))
        a.append( conv( input_channels , output_channels , kernel_size , 1 , padding if padding is not None else (kernel_size - 1)//2 , None , False, None))
        self.model = nn.Sequential(*a)
        
    def forward(self, x):
        return self.activation(self.model(x) + self.shortcut(x))
    

#################### Calculate the shape of the output of the convolution block ###############    
def calc_conv_outsize(input_size, filter_size, stride, pad):
    return math.floor((input_size - filter_size + 2 * pad) / stride) + 1



def round_half_up(n):
    return math.floor(n + 0.5)


############ Calculate the padding size ############################
def calc_conv_pad(input_size, output_size, filter_size, stride):
    return round_half_up((stride * (output_size - 1) + filter_size - input_size) / 2)


def calc_deconv_pad(input_size, output_size, filter_size, stride):
    return round_half_up((stride * (input_size - 1) + filter_size - output_size) / 2)


def same_padding(size, kernel_size, stride, dilation):
    return ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2

#######################################################################

def show_feature_map(c):
    s = int(c.size()[1] / 4)
    fig, ax = plt.subplots(s, 4, figsize=(15, 10))
    for i in range(s):
        for j in range(4):
            ax[i, j].imshow(c[0][j*s + i].cpu().detach())
    
def show(tensor):
    img = transforms.ToPILImage()(tensor)
    plt.imshow(img, cmap='gray')
    plt.show()
    
###################################################################
def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

######################################################################
def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

####################################################################
######################################################################
def inverse_transform(X):
  return (X + 1.) / 2.

#####################################################################
class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out

########################################################################
class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

#######################################################################
class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
#####################################################################