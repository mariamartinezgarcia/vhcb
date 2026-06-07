import torch
from torch import nn
from src.nn.modules import LambdaLayer, ScaledTanh
from src.nn.modules import dclamp

class Encoder(nn.Module):
    """
    General class implementing the decoder of the model.
    """

    def __init__(self, enc, concept_bits, sc_type=None):
        """
        Initialize an instance of the class.

        Parameters
        ----------
        enc: torch.nn.Module
            Module with the architecture of the encoder neural network without the output activation.
        concept_bits: int
            Number of concept bits.
        sc_type: str, optional
            Type of side channel. Default None.
            - None: do not use any side channel.
            - 'binary: binary side channel.
            - 'continuous': continuous side channel.
        """

        super(Encoder, self).__init__()

        self.concept_bits = concept_bits

        self.enc = enc
        self.sc_type = sc_type

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        """
        Forward pass.

        Parameters
        ----------
        x: torch.tensor
            Batch of data.
        """
        
        # Forward the encoder
        out = self.enc(x)

        # The encoder outputs bit probabilities q(bit=1|x)
        concept_probs = self.sigmoid(out[:,:self.concept_bits])
        # Clamp the output to avoid numerical instabilities
        concept_probs = dclamp(concept_probs, 0.001, 0.999) 

        side_channel = None
        if self.sc_type == 'binary':
            # Compute the mean of the Gaussian distribution
            side_channel = self.sigmoid(out[:,self.concept_bits:])
            # Clamp the output to avoid numerical instabilities
            side_channel = dclamp(side_channel, 0.001, 0.999) 
        if self.sc_type == 'continuous':
            # Compute the mean of the Gaussian distribution
            side_channel = out[:,self.concept_bits:]
      
        return concept_probs, side_channel


