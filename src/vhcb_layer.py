import os
import torch
from torch import nn
import pickle
from src.nn.encoder import Encoder
from src.nn.decoder import Decoder
from src.utils.sampling import sample_from_qz_given_x, modulate_words
from src.train.loss import skl_div_bernoulli, kl_div_bernoulli, kl_div_gaussian, infer_and_sample_concepts, infer_and_sample_side_channel, log_bernoulli, log_gaussian

class VHCB_layer(nn.Module):
    def __init__(self, noise_dim, n_concepts, enc=None, dec=None,concept_inf='uncoded', sc_type=None, sc_inf='uncoded', sc_dim=None, G_concept=None, G_sc=None, beta=10, num_ws=14):
        super().__init__()

        # Hyperparameters
        self.beta = torch.tensor(beta)
     
        # Main Configuration
        self.n_concepts = n_concepts
        self.concept_inf = concept_inf
        
        # Side Channel Configuration
        self.sc_type = sc_type
        print('Side channel type: ', sc_type)
        self.sc_dim = sc_dim
        self.sc_inf = sc_inf

        # Concept Code
        self.G_concept = G_concept
        if G_concept != None:
            # G is not a list of tensors but rather one tensor
            self.H_concept = G_concept.T
            self.bits_code_concept = G_concept.shape[1]
        else:
            self.H_concept = None
            self.bits_code_concept = n_concepts

        # Side Channel Code
        self.G_sc = G_sc
        if G_sc != None:
            # G is not a list of tensors but rather one tensor
            self.H_sc= G_sc.T
            self.bits_code_sc = G_sc.shape[1]
        else:
            self.H_sc = None
            self.bits_code_sc = sc_dim

        # Dimension of the latent space
        latent_dim =  self.bits_code_concept
        if self.sc_type == 'continuous':  
            latent_dim = self.bits_code_concept + sc_dim*2
        elif self.sc_type == 'binary':  
            if sc_inf == 'uncoded':
                latent_dim = self.bits_code_concept + sc_dim
            if sc_inf == 'rep':
                latent_dim = self.bits_code_concept + self.bits_code_sc
  
        # Define the encoder and decoder networks
        if enc is None:
            enc = nn.Sequential(
                nn.Linear(noise_dim, noise_dim),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(noise_dim),
                nn.Linear(noise_dim, noise_dim),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(noise_dim),
                nn.Linear(noise_dim, noise_dim),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(noise_dim),
                nn.Linear(noise_dim, latent_dim),
            )
        # Define the decoder network
        if  self.sc_type == 'continuous':
            latent_dim = self.bits_code_concept + sc_dim
        if dec is None:
            dec = nn.Sequential(
                nn.Linear(latent_dim, noise_dim),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(noise_dim),
                nn.Linear(noise_dim, noise_dim),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(noise_dim),
                nn.Linear(noise_dim, noise_dim),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(noise_dim),
                nn.Linear(noise_dim, noise_dim),
            )

        # Encoder
        self.encoder = Encoder(enc, self.bits_code_concept, sc_type=self.sc_type)
        # Decoder
        self.decoder = Decoder(dec)

        # For StyleGAN Latent 
        self.num_ws = num_ws

        print('Number of layers in the VHCB Module:', len(self.encoder.enc) + len(self.decoder.dec))


    def encode(self, x):
        """
        Encode the input into concept and side channel codes.
        """

        # the latent vector will be like (batch_size, 14, 512)
        # where the 512 vector is repeated 14 times since there are 14 layers in the stylegan synthesis network
        # so we can take any one of the 14 (but use mean to maintain differentiability)

        # Account for num_ws in StyleGAN
        if len(x.shape) == 3:
            x = torch.mean(x, dim=1)
        
        # Forward encoder
        out_concept, out_sc = self.encoder.forward(x)

        # Sanity check
        assert torch.any(torch.isinf(out_concept))==False, "Invalid probs value (inf)."
        assert torch.any(torch.isnan(out_concept))==False, "Invalid probs value (nan)."

        # Uncoded case
        if self.concept_inf == 'uncoded':
            # Sample z 
            concept_probs = out_concept

        # Coded case
        if self.concept_inf == 'rep':
            # Compute the information marginals
            logpm1 = torch.matmul(torch.log(out_concept), self.H_concept.to(x.device))
            logpm0 = torch.matmul(torch.log(1-out_concept), self.H_concept.to(x.device))
            log_marginals = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

            # Introduce code structure
            concept_probs = torch.exp(log_marginals_norm[:,:,1])

        # Side Channel
        sc = None
        if self.sc_type == 'continuous':
            # Sample z 
            mean = out_sc[:,:self.sc_dim]             # First half of the output corresponds to the mean vector
            var = torch.exp(out_sc[:,self.sc_dim:])   # Second half of the output corresponds to the logvar vector
            sc = torch.cat((mean, var), dim=1)

        if self.sc_type == 'binary':

            if self.sc_inf == 'uncoded':
                sc = out_sc

            if self.sc_inf == 'rep':
                # Compute the information marginals
                logpm1 = torch.matmul(torch.log(out_sc), self.H_sc.to(x.device))
                logpm0 = torch.matmul(torch.log(1-out_sc), self.H_sc.to(x.device))
                log_marginals = torch.stack((logpm0, logpm1), dim=2)
                log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

                # Introduce code structure
                sc = torch.exp(log_marginals_norm[:,:,1])

        return concept_probs, sc
    
    def decode(self, x):
        """
        Forward the decoder.
        """

        # Forward decoder
        out = self.decoder.forward(x)

        # Sanity check
        assert torch.any(torch.isinf(out))==False, "Invalid output value (inf)."
        assert torch.any(torch.isnan(out))==False, "Invalid output value (nan)."
        # Reshape the output to match the input shape in the case of StyleGAN
        if not (self.num_ws is None):
            out = out.unsqueeze(1).repeat([1, self.num_ws, 1])

        return out
    
    def generate(self, n_samples=100, m_probs=None, m=None, device='cuda'):

        """
        Generate new samples following the generative model.

        Parameters
        ----------
        n_samples: int, optional
            Number of samples to generate.
        m_probs: list, optional
            List with fixed concept pribability vectors. 

        Returns
        -------
        Generated samples and the sampled concepts used for generation.

        """

        if device == 'cuda' and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        if m is None:

            if m_probs is None:
                m_probs = torch.ones((n_samples, self.n_concepts))*0.5
            else:
                m_probs = torch.flatten(m_probs).repeat(n_samples,1)

            # Sample m
            m_sample = m_probs.bernoulli()

        else:
            m_sample = m.clone()

        # Uncoded case
        if self.concept_inf == 'uncoded':
            
            # Sample z
            z_sample = modulate_words(m_sample.to(device), beta=self.beta)

        # Coded case
        if self.concept_inf == 'rep':
            # Obtain a codeword
            c = torch.matmul(m_sample, self.G_concept)
            # Sample z
            z_sample = modulate_words(c.to(device), beta=self.beta)
    

        # Sample from the side channel
        sc_sample = None
        sc_z_sample = None
        if self.sc_type == 'continuous':
            # Sample from the side channel
            sc_z_sample = torch.randn(n_samples, self.sc_dim).to(device)
       
        if self.sc_type == 'binary':
            sc_probs = torch.ones((n_samples, self.sc_dim))*0.5

            if self.sc_inf == 'uncoded':
                # Sample m
                sc_sample = sc_probs.bernoulli()
                # Sample z
                sc_z_sample = modulate_words(sc_sample.to(device), beta=self.beta)

            if self.sc_inf == 'rep':
                # Sample m
                sc_sample = sc_probs.bernoulli()
                # Obtain a codeword
                c = torch.matmul(sc_sample, self.G_sc)
                # Sample z
                sc_z_sample = modulate_words(c.to(device), beta=self.beta)

        # Forward decoder
        if self.sc_type is None:
            latent_sample = z_sample
        else:
            latent_sample = torch.cat((z_sample, sc_z_sample), dim=1)
        generated = self.decode(latent_sample.to(device))

        return generated, m_sample
    
    def forward(self, x):

        """
        Forward pass (encoder and decoder).
        """
        batch_size = x.shape[0]

        concept_probs, out_sc = self.encode(x)

        # Sanity check
        assert torch.any(torch.isinf(concept_probs))==False, "Invalid probs value (inf)."
        assert torch.any(torch.isnan(concept_probs))==False, "Invalid probs value (nan)."

        # Sample from the concept latent distribution

        # Uncoded case
        if self.concept_inf == 'uncoded':
            # Sample z 
            concept_probs = concept_probs
            z_sample = sample_from_qz_given_x(concept_probs, beta=self.beta, n_samples=1)

        # Coded case
        if self.concept_inf == 'rep':
            # Introduce code structure
            qc = torch.matmul(concept_probs, self.G_concept.to(x.device))
            # Sample z 
            z_sample = sample_from_qz_given_x(qc, beta=self.beta, n_samples=1)

        # Sample from the side channel
        sc_sample = None
        if self.sc_type == 'continuous':
            # Sample z 
            mean = out_sc[:,:self.sc_dim]             # First half of the output corresponds to the mean vector
            var = torch.exp(out_sc[:,self.sc_dim:])   # Second half of the output corresponds to the logvar vector
            sc_sample = torch.randn(batch_size, self.sc_dim).to(x.device)*var + mean
       
        if self.sc_type == 'binary':

            if self.sc_inf == 'uncoded':
                #Sample z
                sc_sample = sample_from_qz_given_x(out_sc, beta=self.beta, n_samples=1)[:,:,0]

            if self.sc_inf == 'rep':
                # Introduce code structure
                qc = torch.matmul(out_sc, self.G_sc.to(x.device))
                # Sample z
                sc_sample = sample_from_qz_given_x(qc, beta=self.beta, n_samples=1)[:,:,0]

        # Forward decoder
        if self.sc_type is None:
            latent_sample = z_sample[:,:,0]
        else:
            latent_sample = torch.cat((z_sample[:,:,0], sc_sample), dim=1)
        reconstructed = self.decode(latent_sample)

        return latent_sample, concept_probs, reconstructed
    
    def sample_from_latent_dist(self, concept_probs, out_sc=None, test=True, map=False):
        
        batch_size = concept_probs.shape[0]

        # Sample from the concept latent distribution

        # Uncoded case
        if self.concept_inf == 'uncoded':
            if test:
                if map:
                    m_concept = (concept_probs.clone() >= 0.5).float()
                else:
                    m_concept = concept_probs.bernoulli()
                z_sample = modulate_words(m_concept, beta=self.beta)
            else:
                # Sample z 
                z_sample = sample_from_qz_given_x(concept_probs, beta=self.beta, n_samples=1)

        # Coded case
        if self.concept_inf == 'rep':
            if test:
                if map:
                    m_concept = (concept_probs.clone() >= 0.5).float()
                else:
                    m_concept = concept_probs.bernoulli()
                # Obtain a codeword
                c = torch.matmul(m_concept, self.G_concept.to(concept_probs.device))
                z_sample = modulate_words(c, beta=self.beta)
            else:
                # Introduce code structure
                qc = torch.matmul(concept_probs, self.G_concept.to(concept_probs.device))
                # Sample z 
                z_sample = sample_from_qz_given_x(qc, beta=self.beta, n_samples=1)[:,:,0]

        # Sample from the side channel
        sc_sample = None
        if self.sc_type == 'continuous':
            # Sample z 
            mean = out_sc[:,:self.sc_dim]             # First half of the output corresponds to the mean vector
            var = torch.exp(out_sc[:,self.sc_dim:])   # Second half of the output corresponds to the logvar vector
            sc_sample = torch.randn(batch_size, self.sc_dim).to(concept_probs.device)*var + mean
        
        if self.sc_type == 'binary':

            if self.sc_inf == 'uncoded':
                if test:
                    m_sc = out_sc.bernoulli()
                    sc_sample = modulate_words(m_sc, beta=self.beta)
                else:
                    #Sample z
                    sc_sample = sample_from_qz_given_x(out_sc, beta=self.beta, n_samples=1)[:,:,0]

            if self.sc_inf == 'rep':

                if test:
                    m_sc = out_sc.bernoulli()
                    c_sc = torch.matmul(m_sc, self.G_sc.to(out_sc.device))
                    sc_sample = modulate_words(c_sc, beta=self.beta)
                else:
                    # Introduce code structure
                    qc = torch.matmul(out_sc, self.G_sc.to(concept_probs.device))
                    # Sample z
                    sc_sample = sample_from_qz_given_x(qc, beta=self.beta, n_samples=1)[:,:,0]
        
        return m_concept, z_sample, sc_sample
    
    def get_elbo(self, x, target_probs, beta_concepts=1.,  beta_sc=1.):

        # Input
        batch_size = x.shape[0]

        # Forward encoder
        concept_probs, out_sc = self.encode(x)
        target_probs = target_probs.to(x.device)

        # Uncoded case
        if self.concept_inf == 'uncoded':
            # Sample z 
            z_concept = sample_from_qz_given_x(concept_probs, beta=self.beta, n_samples=1)

        # Coded case
        if self.concept_inf == 'rep':
            # Introduce code structure
            qc = torch.matmul(concept_probs, self.G_concept.to(x.device))
            assert torch.any(torch.isinf(qc))==False, "Invalid qc value (inf)."
            assert torch.any(torch.isnan(qc))==False, "Invalid qc value (nan)."
            # Sample z 
            z_concept = sample_from_qz_given_x(qc, beta=self.beta, n_samples=1)

        # Compute the KL Divergence term for the concepts
        kl_div_concepts = skl_div_bernoulli(concept_probs, target_probs)
        assert torch.any(torch.isinf(kl_div_concepts))==False, "Invalid kl_div_concepts value (inf)."
        assert torch.any(torch.isnan(kl_div_concepts))==False, "Invalid kl_div_concepts value (nan)."

        # Infer side channel posterior, sample z_sc, and compute the KL Divergence term for the side channel
        z_sc = None
        kl_div_sc = torch.zeros(batch_size).to(x.device)
        if self.sc_type == 'continuous':
            # Sample z 
            mean = out_sc[:,:self.sc_dim]  # First half of the output corresponds to the mean vector
            log_var = out_sc[:,self.sc_dim:]
            log_var = torch.clamp(log_var, -10, 5) # Clamp log_var to avoid numerical issues
            var = torch.exp(log_var)   # Second half of the output corresponds to the logvar vector
            z_sc = torch.randn(batch_size, self.sc_dim).to(x.device)*var + mean

            kl_div_sc = kl_div_gaussian(mean, log_var)
            assert torch.any(torch.isinf(kl_div_sc))==False, "Invalid KL DIV side channel value (inf)."
            assert torch.any(torch.isnan(kl_div_sc))==False, "Invalid KL DIV side channel value (nan)."
       
        if self.sc_type == 'binary':

            if self.sc_inf == 'uncoded':
                #Sample z
                z_sc = sample_from_qz_given_x(out_sc, beta=self.beta, n_samples=1)

            if self.sc_inf == 'rep':
                # Introduce code structure
                qc = torch.matmul(out_sc, self.G_sc.to(x.device))
                assert torch.any(torch.isinf(qc))==False, "Invalid qc side channel value (inf)."
                assert torch.any(torch.isnan(qc))==False, "Invalid qc side channel value (nan)."
                # Sample z
                z_sc = sample_from_qz_given_x(qc, beta=self.beta, n_samples=1)
            
            prior_probs = (torch.ones(out_sc.shape)*0.5).to(x.device)
            kl_div_sc =  kl_div_bernoulli(out_sc, prior_probs)
            assert torch.any(torch.isinf(kl_div_sc))==False, "Invalid KL DIV side channel value (inf)."
            assert torch.any(torch.isnan(kl_div_sc))==False, "Invalid KL DIV side channel value (nan)."


        # Compute the reconstruction term E_{q(z|x)}[log p(x|z)]

        if z_sc is None:
            latent_sample = z_concept[:,:,0]
        else:
            if self.sc_type == 'continuous':
                latent_sample = torch.cat((z_concept[:,:,0], z_sc), dim=1)
            else:
                latent_sample = torch.cat((z_concept[:,:,0], z_sc[:,:,0]), dim=1)

        #out_decoder = decoder.forward(latent_sample).view(-1, x_flat.shape[1])
        out_decoder = self.decode(latent_sample)
        # Real observation model
        covar = torch.ones(out_decoder.shape[-1]).to(x.device) * 0.1
        reconstruction_term = log_gaussian(torch.mean(x, dim=1), torch.mean(out_decoder, dim=1), covar)    # Fixed variance

        # Obtain the ELBO 
        elbo = torch.sum((reconstruction_term - beta_concepts*kl_div_concepts - beta_sc*kl_div_sc), dim=0)/batch_size
        
        return elbo, torch.sum(kl_div_concepts, dim=0)/batch_size, torch.sum(kl_div_sc, dim=0)/batch_size, torch.sum(reconstruction_term, dim=0)/batch_size, concept_probs, out_sc, out_decoder, z_sc, z_concept