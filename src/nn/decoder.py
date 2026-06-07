
from torch import nn

class Decoder(nn.Module):
    """
    General class implementing the decoder of the model.
    """

    def __init__(self, dec):
        super(Decoder, self).__init__()

        """
        Initialize an instance of the class.

        Parameters
        ----------
        dec: torch.nn.Module
            Module with the architecture of the decoder neural network.
        """

        if not dec:
            raise Exception("Invalid decoder.")
        else:
            self.dec = dec
    

    def forward(self, x):

        """
        Forward pass.

        Parameters
        ----------
        x: torch.tensor
            Batch of data.
        """
        # Forward the decoder
        out = self.dec(x)  
        
        return out


