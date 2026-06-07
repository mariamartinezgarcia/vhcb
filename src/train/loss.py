import torch
from torch import nn
import torch.distributions as dist
from src.utils.sampling import sample_from_qz_given_x
from src.nn.modules import dclamp
import torch.nn.functional as F
from src.nn.iternorm import IterNorm



def OrthogonalProjectionLoss(embed1, embed2):

    """
    Compute the orthogonality loss between two embeddings (latent representations).
        
        Parameters
        ----------
        embed1 : torch.tensor
            Embedding.
        embed2 : torch.tensor
            Embedding.
    
        Returns
        -------
        Orthogonality loss.
    """

    #  features are normalized
    embed1 = F.normalize(embed1, dim=1)
    embed2 = F.normalize(embed2, dim=1)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    output = torch.abs(cos(embed1, embed2))

    return output.mean()



def log_bernoulli(probs, observation):

    """
    Evaluate a Bernoulli distribution.
        
        Parameters
        ----------
        probs : torch.tensor
            Tensor with the probabilities to define the Bernoulli. [shape (batch_size, dimension)]
        observation: torch.tensor
            Batch of data.
    
        Returns
        -------
        Log probability.
    """

    bce = torch.nn.BCELoss(reduction='none')

    return -torch.sum(bce(probs, observation), dim=1)


def log_gaussian(x, mean, covar):

    """
    Evaluate a Multivariate Gaussian distribution with diagonal covariance matrix.
        
        Parameters
        ----------
        x : torch.tensor
            Batch of data.
        mean : torch.tensor
            Means of the distribution.
        covar : torch.tensor
            Value of the diagonal.

        Returns
        -------
        Log probability.
    """

    # MVN INDEPENDEN NORMAL DISTRIBUTIONS
    # Create a multivariate normal distribution with diagonal covariance
    gaussian = dist.independent.Independent(dist.Normal(mean, torch.sqrt(covar)), 1)
    
    return gaussian.log_prob(x)


def kl_div_bernoulli(q_probs, p_probs):

    """
    Compute KL Divergence D_KL(q|p) between two Bernoulli distributions.

    Parameters
        ----------
        q_probs : torch.tensor
            Probabilities that define the q distribution.
        p_probs : torch.tensor
           Probabilities that define the p distribution.

        Returns
        -------
        kl_div : torch.tensor
            Kullback-Leibler divergence between the given distributions.
    """

    q = dist.Bernoulli(dclamp(q_probs, min=1e-3, max=1-1e-3)) # clamp to avoid numerical instabilities
    p = dist.Bernoulli(dclamp(p_probs, min=1e-3, max=1-1e-3))

    kl_div = dist.kl.kl_divergence(q, p)

    kl_div = torch.sum(kl_div, dim=1)

    #kl_div = q_probs*(torch.log(q_probs)-torch.log(p_probs))+(1-q_probs)*(torch.log(1-q_probs)-torch.log(1-p_probs))
    #kl_div = torch.sum(kl_div, dim=1)

    return kl_div

def skl_div_bernoulli(q_probs, p_probs):

    """
    Compute Symmetric KL Divergence SD_KL(q,p) between two Bernoulli distributions.

    Parameters
        ----------
        q_probs : torch.tensor
            Probabilities that define the q distribution.
        p_probs : torch.tensor
           Probabilities that define the p distribution.

        Returns
        -------
        skl_div : torch.tensor
            Symmetric Kullback-Leibler divergence between the given distributions.
    """

    p_probs = dclamp(p_probs, min=1e-3, max=1-1e-3)
    q_probs = dclamp(q_probs, min=1e-3, max=1-1e-3)

    skl_div = (p_probs-q_probs)*(torch.log(p_probs)-torch.log(q_probs))+(q_probs-p_probs)*(torch.log(1-p_probs)-torch.log(1-q_probs))

    skl_div = torch.sum(skl_div, dim=1)

    return skl_div

def kl_div_gaussian(mean, logvar):
    
    """
    Compute KL Divergence D_KL(q|p) between our approximate posterior and a standard gaussian.

    Parameters
        ----------
        mean : torch.tensor
            Mean of the q distribution.
        logvar : torch.tensor
            Log variance of the q distribution.

        Returns
        -------
        kl_div : torch.tensor
            Kullback-Leibler divergence between the given distributions.
    """
    #prior = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(mean.shape))
    #logvar = torch.clamp(logvar, min=-30.0, max=5.0)  # clamp to avoid numerical instabilities
    kl_div = -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1)

    return kl_div


def compute_word_logprobs(bit_probs, code_words):

    """
    Compute the log probability of the words in the codebook.

    Parameters
        ----------
        bit_probs : torch.tensor
            Bit probabilities.
        code_words : torch.tensor
            Matrix containing the codebook.

        Returns
        -------
        logq : torch.tensor
            Unnormalized distribution over words.
        logq_norm : torch.tensor
            Normalized distribution over words.
    """

    # Sanity check
    assert torch.any(bit_probs < 0)==False, "Negative value encountered in bit probabilities."
    assert torch.any(bit_probs > 1)==False, "Value larger than 1 encountered in bit probabilities."
    assert torch.all(torch.logical_or(code_words == 0, code_words == 1)), "Invalid word encountered. All words should be binary vectors."

    # === Compute log(q(c|x,C)) [evaluate log(q_uncoded(c|x)) for code words] === #

    # 1. Extend the output of the encoder in a third dimension to obtain a tensor of shape [batch_size, K, n_words]
    # 2. Extend the code words matrix in a third dimension to obtain a tensor of shape [batch_size, K, n_words]
    # 3. Reduce the logq in dim=1 to obtain a matrix of shape [batch_size, n_words] containing the evaluation of log(q(c|x,C)) for each code word
    
    n_words = code_words.shape[0] 
    batch_size = bit_probs.shape[0]

    # Clamp to avoid numerical instabilities
    bit_probs = dclamp(bit_probs, min=0.001, max=0.999)

    # Evaluate log(q_uncoded(c|x)) for code words
    logq = log_bernoulli(bit_probs.unsqueeze(2).repeat(1, 1, n_words), code_words.T.unsqueeze(0).repeat(batch_size,1,1))
    
    # Clamp to avoid numerical instabilities
    logq = dclamp(logq, min=-100, max=1)

    # Sanity check
    assert torch.any(torch.isinf(logq))==False, "Invalid logq value (inf)."
    assert torch.any(torch.isnan(logq))==False, "Invalid logq value (nan)."

    # Normalization
    logq_norm = logq - logq.logsumexp(dim=-1, keepdim=True)

    # Sanity check
    assert torch.all(torch.exp(logq_norm) >= 0), "Negative value encountered in normalized probs."
    assert torch.all((torch.exp(logq_norm).sum(-1) - 1).abs() < 1e-5), "Normalized probabilities do not sum 1."

    return logq, logq_norm



def infer_and_sample_concepts(encoder_out, inf='uncoded', beta=10, n_samples=1, G=None, H=None):

    """
    Obtain the marginal posterior probabilities of the concepts and sample the soft representation z accordingly.

    Parameters
        ----------
        encoder_out : torch.tensor
            Output of the encoder NN. 
        inf : str, optional
            Inference type for the binary concepts. Default 'uncoded'.
            - 'uncoded' for the uncoded case.
            - 'rep' for the coded case with inference at bit level using repetition codes.
        beta: float, optional
            Temperature term that controls the decay of the exponentials in the smoothing transformation. Default to 10.
        n_samples: int, optional
            Number of samples drawn from q(z|x). Default to 1.
        G : torch.tensor, optional
            Matrix used to encode information words. Default None.
        H : torch.tensor, optional
            Matrix used to decode code words. Default None.
        Returns
        -------
        concept_probs : torch.tensor
            Posterior concept probabilities.
        qz_sample: torch.tensor
            Sample from drawn from q(z|x)
    """

    if inf=='uncoded':
        concept_probs = encoder_out
        # Obtain n_samples from q(z|x) for each observed x
        qz_sample = sample_from_qz_given_x(concept_probs, beta=beta, n_samples=n_samples)

    if inf=='rep':

        logpm1 = torch.matmul(torch.log(encoder_out), H.to(encoder_out.device))
        logpm0 = torch.matmul(torch.log(1-encoder_out), H.to(encoder_out.device))

        log_marginals = torch.stack((logpm0, logpm1), dim=2)

        log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

        concept_probs = torch.exp(log_marginals_norm[:,:,1])

        # Introduce code structure
        qc = torch.matmul(concept_probs, G.to(encoder_out.device))
    
        # Obtain n_samples from q(z|x) for each observed x
        qz_sample = sample_from_qz_given_x(qc, beta=beta, n_samples=n_samples)  # shape [N, K, n_samples]


    return concept_probs, qz_sample


def infer_and_sample_side_channel(encoder_out, type='continuous', inf=None, beta=10, n_samples=1, G=None, H=None):

    """
    Obtain the marginal posterior distribution of the side channel and sample the soft representation z accordingly.

    Parameters
        ----------
        encoder_out : torch.tensor
            Output of the encoder NN. 
        type: str, optional
            Type of side channel. Default None.
            - None: do not use any side channel.
            - 'binary: binary side channel.
            - 'continuous': continuous side channel.
        inf : str, optional
            Inference type for the binary side channel. Default None.
            - 'uncoded' for the uncoded case.
            - 'rep' for the coded case with inference at bit level using repetition codes.
        beta: float, optional
            Temperature term that controls the decay of the exponentials in the smoothing transformation. Default to 10.
        n_samples: int, optional
            Number of samples drawn from q(z|x). Default to 1.
        G : torch.tensor, optional
            Matrix used to encode information words. Default None.
        H : torch.tensor, optional
            Matrix used to decode code words. Default None.
        Returns
        -------
        sc_probs : torch.tensor
            Posterior side channel bit probabilities if type='binary, None otherwise. 
        qz_sample: torch.tensor
            Sample from drawn from q(z|x)
    """

    if type=='continuous':
        sc_probs = None
        sc_dim = (encoder_out.shape[1]//2)
        # Sample from the side channel
        mean = encoder_out[:,:sc_dim].unsqueeze(-1).repeat(1,1,n_samples) # shape [N, sc_dim, n_samples]
        var = torch.exp(encoder_out[:,sc_dim:]).unsqueeze(-1).repeat(1,1,n_samples) # shape [N, sc_dim, n_samples]
        sc_sample = torch.randn(encoder_out.shape[0], sc_dim, n_samples).to(encoder_out.device)*var + mean # shape [N, sc_dim, n_samples]

    if type=='binary' and inf=='uncoded':
        sc_probs = encoder_out
        sc_sample = sample_from_qz_given_x(sc_probs, beta=beta, n_samples=1)

    if type=='binary' and inf=='rep':

        logpm1 = torch.matmul(torch.log(encoder_out), H.to(encoder_out.device))
        logpm0 = torch.matmul(torch.log(1-encoder_out), H.to(encoder_out.device))

        log_marginals = torch.stack((logpm0, logpm1), dim=2)

        log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

        sc_probs = torch.exp(log_marginals_norm[:,:,1])

        # Introduce code structure
        qc = torch.matmul(sc_probs, G.to(encoder_out.device))

        # Obtain n_samples from q(z|x) for each observed x
        sc_sample = sample_from_qz_given_x(qc, beta=beta, n_samples=n_samples)  # shape [N, K, n_samples]

    return sc_probs, sc_sample


def get_losses(x, concepts, encoder, decoder, prior_bits=0.5, beta=10, concept_inf='uncoded', sc_type="continuous", sc_inf=None, G_concept=None, H_concept=None, G_sc=None, H_sc=None, likelihood='gauss', n_samples=1, whitening=None):

    """
    Compute the losses for training, i.e., the ELBO, concept loss and orthogonality loss.

    Parameters
        ----------
        x : torch.tensor
            Batch of data.
        concepts : torch.tensor
            Batch of concept vectors.
        encoder : Encoder instance
            Model's encoder neural network.
        decoder : Decoder instance
            Model's decoder neural network.
        prior_bits: float, optional
            Prior probability of each bit in an independent Bernoulli distribution. Default to 0.5.
        beta: float, optional
            Temperature term that controls the decay of the exponentials in the smoothing transformation. Default to 10.
        concept_inf : str, optional
            Inference type for the binary concepts. Default 'uncoded'.
            - 'uncoded' for the uncoded case.
            - 'rep' for the coded case with inference at bit level using repetition codes.
        sc_type: str, optional
            Type of side channel. Default None.
            - None: do not use any side channel.
            - 'binary: binary side channel.
            - 'continuous': continuous side channel.
        sc_inf: str, optional
            Inference type for the binary side channel. Default 'uncoded'.
            - 'uncoded' for the uncoded case.
            - 'rep' for the coded case with inference at bit level using repetition codes.
        G_concept : torch.tensor, optional
            Matrix used to encode concept vectors.
        H_concept : torch.tensor, optional
            Matrix used to decode concept code words. Default None.
        G_sc : torch.tensor, optional
            Matrix used to encode information words in the binary side channel.
        H_sc : torch.tensor, optional
            Matrix used to decode code words in the binary side channel. Default None.
        likelihood: string, optional
            Distribution used to compute the reconstruction term. Default 'gauss'.
            - 'gauss': Gaussian likelihood.
            - 'ber': Bernoulli likelihood.
        n_samples: int, optional
            Number of samples used to estimate the ELBO. Default to 1.
        whitening: IterNorm instance, optional
            Whitening module used to normalize the latent representation. Default None.

    Returns
        -------
        elbo : torch.tensor
            Value of the ELBO.
        concept_loss : torch.tensor
            Binary Cross Entropy evaluated with the posterior concept probabilities and the ground truth concepts.
        orth_loss:
            Orthogonality loss between the side channel and the concept soft representation. Only computed in the case of a continuous side channel.
        kl_div_concepts : torch.tensor
            Value of the concept Kullback-Leibler divergence term in the ELBO.
        kl_div_sc : torch.tensor
            Value of the side channel Kullback-Leibler divergence term in the ELBO.
        reconstruction: torch.tensor
            Value of the reconstruction term in the ELBO.
    """

    # Input
    N = x.shape[0]
    x_flat = x.view(N,-1)

    #BCE Loss
    bce = nn.BCELoss(reduction='mean')

    # Forward encoder
    encoder_out, side_channel = encoder.forward(x)

    # Infer concept posteriors and sample z_concept
    concept_probs, z_concept = infer_and_sample_concepts(encoder_out, inf=concept_inf, beta=beta, n_samples=n_samples, G=G_concept, H=H_concept)

    # Compute the KL Divergence term for the concepts
    prior_concepts = (torch.ones(concept_probs.shape)*prior_bits).to(encoder_out.device)
    kl_div_concepts = kl_div_bernoulli(concept_probs, prior_concepts)

    # Obtain Concept Loss
    concept_loss = bce(concept_probs, concepts.type(torch.FloatTensor).to(concept_probs.device))

    # Infer side channel posterior, sample z_sc, and compute the KL Divergence term for the side channel
    kl_div_sc = torch.tensor(0.).to(x_flat.device)
    if sc_type:
        sc_probs, z_sc = infer_and_sample_side_channel(side_channel, type=sc_type, inf=sc_inf, beta=beta, n_samples=n_samples, G=G_sc, H=H_sc)

        if sc_type == 'continuous':
            # Sample from the side channel
            sc_dim = (side_channel.shape[1]//2)
            kl_div_sc = kl_div_gaussian(side_channel[:,:sc_dim], side_channel[:,sc_dim:])
        if sc_type == 'binary':
            prior_probs = (torch.ones(sc_probs.shape)*prior_bits).to(side_channel.device)
            kl_div_sc =  kl_div_bernoulli(sc_probs, prior_probs)


    # Compute the reconstruction term E_{q(z|x)}[log p(x|z)] and orthogonality loss (only computed in the continuous side channel case
    
    reconstruction_sum = 0
    orth_sum = torch.tensor(0.).to(encoder_out.device)

    if z_sc is None:
        latent_sample = z_concept[:,:,:]
    else:
        latent_sample = torch.cat((z_concept[:,:,:], z_sc[:,:,:]), dim=1)

    for n in range(n_samples):
        
        # Forward decoder
        n_sample = latent_sample[:,:,n]
        if (sc_type == 'continuous') and (whitening is not None):
            n_sample = whitening(n_sample.view(n_sample.shape[0], n_sample.shape[1]))
        #out_decoder = decoder.forward(latent_sample).view(-1, x_flat.shape[1])
        out_decoder = decoder.forward(n_sample).view(-1, x_flat.shape[1])

        # Binary observation model
        if likelihood.lower() == 'ber':
            reconstruction_sum += log_bernoulli(out_decoder, x_flat)
        # Real observation model
        elif likelihood.lower() == 'gauss':
            covar = torch.ones(out_decoder.shape[1]).to(x_flat.device) * 0.1
            reconstruction_sum += log_gaussian(x_flat, out_decoder, covar)    # Fixed variance

        #if sc_type == 'continuous':
            #orth_sum += OrthogonalProjectionLoss(z_concept[:,:,n], z_sc[:,:,n])

    reconstruction = reconstruction_sum/n_samples
    orth_loss = orth_sum/n_samples

    # Obtain the ELBO 
    elbo = torch.sum((reconstruction - kl_div_concepts - kl_div_sc), dim=0)/N

    return elbo, concept_loss, orth_loss, torch.sum(kl_div_concepts, dim=0)/N, torch.sum(kl_div_sc, dim=0)/N, torch.sum(reconstruction, dim=0)/N

