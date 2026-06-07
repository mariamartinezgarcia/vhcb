import torch
import numpy as np

def check_args(inference, code_words=None, G=None, H=None):

    """
    Check arguments.

    Parameters
        ----------
        inference : string
            Inference type.
        code_words : torch.tensor, optional
            Codebook.
        G: torch.tensor, optional
            Metrix to encode information words when using repetition codes.
        H: torch.tensor, optional
            Matrix to decode coded words when using repetition codes,
    """

    # Check if the inference mode indicated is valid
    valid_inference = ['uncoded', 'word', 'rep', 'hier']
    assert inference in valid_inference, "Please, indicate a valid inference mode ['uncoded' for the uncoded case, 'word' for the coded case with inference at word level with random codes, 'rep' for the coded case with inference at bit level with repetition codes]"

    # If the inference mode selected is 'word', check if a codebook is provided
    if inference == 'word':
        assert (code_words is None)==False, "It is necessary to indicate a codebook for inference at word level."
        assert torch.all(torch.logical_or(code_words == 0, code_words == 1)), "Invalid word encountered. All words should be binary vectors."

    # If the inference mode selected is 'rep', check if the code matrices are provided
    if inference == 'rep':
        assert (G is None)==False, 'It is necessary to indicate a matrix G (encoder) for inference at bit level with repetition codes.'
        assert (H is None)==False, 'It is necessary to indicate a matrix H (decoder) for inference at bit level with repetition codes.'


         
def set_random_seed(seed):

    """
    Set seed for reproducibility. 

    Parameters
    ----------
    seed : int
        Seed value.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
