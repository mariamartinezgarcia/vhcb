import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

class DifferentiableClamp(torch.autograd.Function):
    """
    https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


class ScaledTanh(nn.Module):
    
    """
    Class implementing the Scaled Tanh activation.
    """

    def __init__(self, factor):
        super(ScaledTanh, self).__init__()

        """
        Initialize an instance of the class.

        Parameters
        ----------
        factor: float
            Value by which the output of the tanh is multiplied.
        """

        self.factor = factor
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.tensor
            Batch of data.
        """
        return self.factor*self.tanh(x)


class LambdaLayer(nn.Module):
    """
    Class implementing the Lambda Layer. Applies a transformation given by lambd to the inputs.
    """

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()

        """
        Initialize an instance of the class.

        Parameters
        ----------
        lambd: callable function
            Transformation.
        """

        self.lambd = lambd
    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.tensor
            Batch of data.
        """
        return self.lambd(x)

class FullyConnected(nn.Module):
    """
    Class implementing a fully connected (MLP) module.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation=None, output_activation=None):
        super(FullyConnected, self).__init__()

        """
        Initialize an instance of the class.

        Parameters
        ----------
        input_dim: int
            Input dimension.
        hidden_dims: list
            Hidden dimensions.
        output_dim: int
            Output dimension.
        activation: callable function, optional
            Activation used in hidden dimensions. If no activation is indicated, a ReLU is used by default.
        output_activation: callable function, optional
            Activation used at the output. If no output activation is indicated, the output of the module is the output of the last linear layer.
        """
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                if activation is None:
                    layers.append(nn.ReLU())
                else:
                    layers.append(activation)
     
        if not output_activation is None:
            layers.append(output_activation)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.tensor
            Batch of data.
        """
        return self.network(x)
    
    
class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation layer.
    """

    def __init__(self, channels, ratio=16):
        super(SqueezeExcitation, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)
    

class ConvTrSE(nn.Module):

    """
    Transpose Convolution with Squeeze-and-Excitation layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None):
        super(ConvTrSE, self).__init__()

        if activation is None:
            activation = nn.ReLU()

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            activation
        )
        self.se_block = SqueezeExcitation(out_channels)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.se_block(out)
        return out
    

    

class CNNSkip(nn.Module):

    """
    CNN-Skip architecture
    """

    def __init__(self):
        super(CNNSkip, self).__init__()

        self.lrelu = nn.LeakyReLU()

        self.conv1 = ConvTrSE(in_channels=1024, out_channels=512, kernel_size=(1,1), stride=1, padding=0, activation=self.lrelu) #[512, 3, 3]
        self.conv2 = ConvTrSE(in_channels=512, out_channels=256, kernel_size=(4,4), stride=2, padding=1, activation=self.lrelu) #[256, 6, 6]
        self.conv3 = ConvTrSE(in_channels=256, out_channels=128, kernel_size=(4,4), stride=2, padding=0, activation=self.lrelu) #[128, 14, 14]
        self.conv4 = ConvTrSE(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1, activation=self.lrelu) #[64, 28, 28]

        self.skip1 = ConvTrSE(in_channels=1024, out_channels=256, kernel_size=(4,4), stride=2, padding=1, activation=self.lrelu) 
        self.skip2 = ConvTrSE(in_channels=512, out_channels=128, kernel_size=(6,6), stride=4, padding=0, activation=self.lrelu)


    def forward(self, x):

        s1 = self.skip1(x)
        out1 = self.lrelu(self.conv1(x))
        s2 = self.skip2(out1)
        out2 = self.lrelu(self.conv2(out1))
        out3 = self.conv3(self.lrelu(out2+s1))
        out4 = self.conv4(self.lrelu(out3+s2))

        return self.lrelu(out4)
    
# ---- Residual blocks ---- #
    
class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=(3,3),
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=(1,1),
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        print(x.shape)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x
       

# ---- Encoder and Decoder Architectures ---- #

def get_encoder(enc_type, output_dim, dataset):

    if enc_type == 'cnn':
        enc = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(3,3)), # out 26x26
                nn.LeakyReLU(),
                nn.Conv2d(8, 16, kernel_size=(3,3)), # out 24x24
                nn.LeakyReLU(),
                nn.Conv2d(16, 32, kernel_size=(3,3)), # out 22x22
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, kernel_size=(3,3)), # out 20x20
                nn.LeakyReLU(),
                nn.Flatten(1,3),
                FullyConnected(20*20*32, [512], output_dim, activation = nn.Tanh()),
        )

    if enc_type == 'dcgan':
        if dataset == 'color_mnist' or  dataset == 'confounded_color_mnist':

            enc = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1), # out 28x28
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=2), # out 14x14
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2,2), stride=2), # out 7x7
                    nn.LeakyReLU(),
                    nn.Flatten(1,3),
                    #FullyConnected(7*7*256, [522], output_dim, activation = nn.LeakyReLU()),
                    FullyConnected(7*7*256, [512], output_dim, activation = nn.LeakyReLU()),
            )

        if dataset=='MNIST' or dataset=='FMNIST':
            enc = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), padding=1), # out 28x28
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=2), # out 14x14
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2,2), stride=2), # out 7x7
                    nn.LeakyReLU(),
                    nn.Flatten(1,3),
                    #FullyConnected(7*7*256, [522], output_dim, activation = nn.LeakyReLU()),
                    FullyConnected(7*7*256, [512], output_dim, activation = nn.LeakyReLU()),
            )
        if dataset=='CIFAR10' or dataset=='SVHN':
            enc = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1), # out 32x32 
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=2), # out 16x16
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2,2), stride=2), # out 8x8
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(2,2), stride=2), # out 4x4
                    nn.Flatten(1,3),
                    FullyConnected(4*4*512, [512], output_dim, activation = nn.LeakyReLU()),
            )
        
        if dataset=='IMAGENET' or dataset=='celeba':
            enc = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1), # out 64x64
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), stride=2), # out 32x32
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=2), # out 16x16
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2,2), stride=2), # out 8x8
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(2,2), stride=2), # out 4x4
                    nn.Flatten(1,3),
                    FullyConnected(4*4*512, [512], output_dim, activation = nn.LeakyReLU()),
            )

def get_decoder(dec_type, input_dim, dataset):

    if dec_type=='cnn':
        dec = nn.Sequential(
            FullyConnected(input_dim, [512], 20*20*32, activation = nn.Tanh(), output_activation= nn.Tanh()),
            nn.Unflatten(dim=1, unflattened_size=(32, 20, 20)),
            nn.ConvTranspose2d(32, 32, kernel_size=(3,3)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(3,3)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(3,3)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=(3,3)),
            nn.Sigmoid())

    if dec_type == 'cnn_deeper':
        dec = nn.Sequential(
            FullyConnected(input_dim, [1024], 16*16*128, activation = nn.Tanh(), output_activation= nn.Tanh()),
            nn.Unflatten(dim=1, unflattened_size=(128, 16, 16)),
            nn.ConvTranspose2d(128, 128, kernel_size=(3,3)), # 18
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(3,3)), # 20
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3)), # 22
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(3,3)), # 24
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(3,3)), # 26
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=(3,3)), # 28
            nn.Sigmoid()
            )

    if dec_type == 'dcgan':
        dec = nn.Sequential(
            FullyConnected(input_dim, [1024], 3*3*1024, activation = nn.Tanh(), output_activation= nn.Tanh()),
            nn.Unflatten(dim=1, unflattened_size=(1024, 3, 3)), 
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(1,1), stride=1, padding=0), #[512, 3, 3]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4,4), stride=2, padding=1), #[512, 6, 6]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4), stride=2, padding=0), #[128, 14, 14]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1), #[64, 28, 28],
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=1, padding=0), #[1, 28, 28],
            nn.Sigmoid()
        )

    if dec_type == 'cnnskip':

        if dataset == 'color_mnist' or  dataset == 'confounded_color_mnist':

            dec = nn.Sequential(
                FullyConnected(input_dim, [1024], 3*3*1024, activation = nn.Tanh(), output_activation= nn.Tanh()),
                #FullyConnected(input_dim, [1050], 3*3*1024, activation = nn.Tanh(), output_activation= nn.Tanh()),
                nn.Unflatten(dim=1, unflattened_size=(1024, 3, 3)), 
                CNNSkip(),
                nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.Tanh()
            )
             
        if dataset == 'MNIST' or dataset=='FMNIST':
            dec = nn.Sequential(
                FullyConnected(input_dim, [1024], 3*3*1024, activation = nn.Tanh(), output_activation= nn.Tanh()),
                #FullyConnected(input_dim, [1050], 3*3*1024, activation = nn.Tanh(), output_activation= nn.Tanh()),
                nn.Unflatten(dim=1, unflattened_size=(1024, 3, 3)), 
                CNNSkip(),
                nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        if dataset == 'CIFAR10' or dataset=='SVHN':
            dec = nn.Sequential(
                FullyConnected(input_dim, [1024], 3*3*1024, activation = nn.Tanh(), output_activation= nn.Tanh()),
                nn.Unflatten(dim=1, unflattened_size=(1024, 3, 3)), 
                CNNSkip(), # [64, 28, 28]
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.Tanh()
            )
        if dataset == 'IMAGENET' or dataset=='celeba':
            dec = nn.Sequential(
                FullyConnected(input_dim, [1024], 3*3*1024, activation = nn.Tanh(), output_activation= nn.Tanh()),
                nn.Unflatten(dim=1, unflattened_size=(1024, 3, 3)), 
                CNNSkip(), # [64, 28, 28]
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=24, kernel_size=3, stride=1, padding=0),
                #nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=0), # [32, 32, 32]
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=24, out_channels=8, kernel_size=2, stride=2, padding=0), # [3, 64, 64]
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.Tanh()
            )
    
    if dec_type == 'cnnskipcifar10':
            dec = nn.Sequential(
                FullyConnected(input_dim, [1024], 3*3*1024, activation = nn.Tanh(), output_activation= nn.Tanh()),
                nn.Unflatten(dim=1, unflattened_size=(1024, 3, 3)), 
                CNNSkip(),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0), #[32, 30, 30]
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0), #[32, 30, 30]
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=0), #[3, 32, 32]
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0), #[3, 32, 32]
                nn.Tanh()
            )

    if dec_type == 'pixelcnn':
        dec = nn.Sequential(
            FullyConnected(input_dim, [1024], 3*3*1024, activation = nn.Tanh(), output_activation= nn.Tanh()),
            nn.Unflatten(dim=1, unflattened_size=(1024, 3, 3)), 
            CNNSkip(),
            PixelCNN(in_channels=64, out_channels=64, channels=64, kernel=5, n_layers=5, activation=nn.LeakyReLU()),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),

            nn.Sigmoid()
        )

    # DEBUGGEAR
    if dec_type == 'vqvae':
        "https://github.com/google-deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb"
        dec = nn.Sequential(
            FullyConnected(input_dim, [512], 7*7*128, activation = nn.ReLU(), output_activation=None),
            nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=1),
            ResidualStack(in_dim=128,h_dim=128, res_h_dim=32, n_res_layers=2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=1, kernel_size=(2,2), stride=2),
            nn.Sigmoid()
        )

    return dec