import torch
from torch.nn.functional import one_hot
from src.nn.modules import dclamp

def sample_from_qc_given_x(logits, code_words, n_samples=1):

    """
    Obtain samples from q(c|x), i.e., obtain samples from the posterior distribution over words.

    Parameters
        ----------
        logits : torch.tensor
            Code words log-probabilities (can be unnormalized).
        code_words : torch.tensor
            Codebook.
        n_samples: int, optional
            Number of samples used to estimate the ELBO. Default to 1.

        Returns
        -------
        word_sample : torch.tensor
            Words sampled.
        c_sample_logprob : torch.tensor
            Log-probability of the sampled words.
    """

    n_words = code_words.shape[0]

    # Define the distribution over code words
    q_c_given_x = torch.distributions.categorical.Categorical(logits=logits, validate_args=False)
    c_sample = q_c_given_x.sample((n_samples,)).transpose(0,1) # shape [N,n_samples] 

    # Sanity check
    assert torch.any(c_sample >= n_words)==False, "Invalid values encountered in c_sample (c_sample >= n_words)."
    assert torch.any(c_sample < 0)==False, "Invalid values encountered in c_sample (c_sample < 0)."

    c_sample_logprob = q_c_given_x.log_prob(c_sample.transpose(0,1))
    c_ohe = one_hot(c_sample, num_classes=n_words) # shape [N, n_samples, n_words]
    words_sample = (c_ohe.float()@code_words).transpose(1,2) # shape [N, n_bits, n_samples]

    return words_sample, c_sample_logprob


def sample_from_qz_given_x(qi, beta=torch.tensor(10), n_samples=1): 

    """
    This method implements the DVAE's reparameterization trick.

    Parameters
        ----------
        qi : torch.tensor
            Probability of bits being 1.
        beta : torch.tensor
            Temperature term that controls the decay of the exponentials in the smoothing transformation. Default to 10.
        n_samples: int, optional
            Number of samples used to estimate the ELBO. Default to 1.

        Returns
        -------
        q_z: torch.tensor
            Sampled z.
    """

    # Here we are implementing the reparameterization trick from the DiscreteVAE

    # Sanity check
    assert torch.any(qi < 0)==False, "Negative value encountered in bit probabilities."
    assert torch.any(qi > 1)==False, "Value larger than 1 encountered in bit probabilities."


    # Obtain n_samples from q(z|x) {REPARAMETERIZATION}
    epsilon = 1e-6

    # Bit probabilities q(c_i=1|x)
    qi = qi.unsqueeze(2).repeat(1, 1, n_samples)
    ones = torch.ones((qi.shape)).to(qi.device)

    # Clamp to avoid divisions by 0 in the reparameterization
    qi = dclamp(qi, 0, 1-1e-3)  
    
    # Sample from U(0,1)
    rho = torch.rand(qi.shape).to(qi.device)

    # Reparameterization
    b = (rho+torch.exp(-beta)*(qi-rho))/(ones-qi) - ones
    c = -(qi*torch.exp(-beta))/(ones-qi)

    dif = torch.sqrt(torch.pow(b, 2) - 4*c) - b
    dif = torch.where(dif <= 0, epsilon, dif)   # avoid negative or zero values due to numerical imprecission

    q_z = (-1/beta)*torch.log(dif/2) # shape [N, K, n_samples]

    # Sanity check
    assert torch.any(torch.isinf(q_z))==False, "Invalid q(z|x) value (inf)."
    assert torch.any(torch.isnan(q_z))==False, "Invalid q(z|x) value (nan)."
  
    return q_z


def modulate_words(words, beta=10):

    """
    Modulate words using the smoothing transformations.

    Parameters
        ----------
        words: torch.tensor
            Words to be modulated.
        beta : float
            Temperature term that controls the decay of the exponentials in the smoothing transformation. Default to 10.
    
        Returns
        -------
        z: torch.tensor
            Modulated words.
    """
    
    # Here we are modulating C using the exponentials
    # Obtain sample from p(z|c) in the coded case -> we fix the code word (it was previously sampled from q(c|x,C))
    # c_sample has shape [N, n_bits, n_samples] -> we obtain a sample from p(z|c) for each sampled word c

    # Sanity check
    assert torch.all(torch.logical_or(words == 0, words == 1)), "Invalid word encountered. All words should be binary vectors."

    epsilon = 1e-6

    # Create masks with the sampled word to select the bit samples
    not_c_sample = torch.logical_not(words).type(torch.float32) # shape [N, n_bits, n_samples], ones where ci=0

    # Auxiliar variable
    ones = torch.ones((words.shape)).to(words.device)

    # Sample from U(0,1)
    rho = torch.rand(words.shape).to(words.device)

    # Sample from F^(-1)(zi|ci=0)
    z0 = -(1/beta)*torch.log((ones-rho*(ones-ones*torch.exp(-beta)))+epsilon)
    # Apply ci=0 mask
    z0 = z0*not_c_sample

    # Sample from F^(-1)(zi|ci=1)
    z1 = (1/beta)*torch.log((rho*(ones-ones*torch.exp(-beta) + ones*torch.exp(-beta)))+epsilon) + 1
    # Apply ci=1 mask
    z1 = z1*words

    z = z0+z1

    # Sanity check
    assert torch.any(torch.isinf(z))==False, "Invalid z_sample value (inf)."
    assert torch.any(torch.isnan(z))==False, "Invalid z_sample value (nan)."

    return z  # shape [N, n_bits, n_samples]