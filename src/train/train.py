import torch
import wandb
from src.train.loss import get_losses

def has_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


def train_step(model, x, concepts, n_samples=1, train_enc=True, train_dec=True,  w_concept=1, w_orth=1):

    """
    Train step.
        
        Parameters
        ----------
        model : CodedDVAE instance
            Model to be trained.
        x : torch.tensor
            Batch of data.
        n_sampes : int, optional
            Number of samples used for computing the ELBO. The number of samples is 1 by default.
        train_enc : boolean, optional
            Flag to indicate if the parameters of the encoder need to be updated. True by default.
        train_enc : boolean, optional
            Flag to indicate if the parameters of the decoder need to be updated. True by default.
        w_concept: float, optional
            Weight for the concept loss term. Default 1.
        w_orth: float, optional
            Weight for the orthogonality loss term. Default 1.   
        
        Returns
        -------
        elbo : torch.tensor
            Value of the ELBO.
        kl_div_concepts : torch.tensor
            Value of the Kullback-Leibler divergence term in the ELBO.
        reconstruction: torch.tensor
            Value of the reconstruction term in the ELBO.
        concept_loss: torch.tensor
            Value of the concept_loss (BCE).
        orth_loss: orth_tensor
            Value of the orthogonality loss between the concepts and the side channel.
    """

    x = x.to(model.device)
    concepts= concepts.to(model.device)
    
    model.optimizer_encoder.zero_grad()
    model.optimizer_decoder.zero_grad()

    # Compute losses
    elbo, concept_loss, orth_loss, kl_div_concepts, kl_div_sc, reconstruction = get_losses(
        x, 
        concepts, 
        model.encoder, 
        model.decoder, 
        beta=model.beta, 
        concept_inf=model.concept_inf, 
        sc_type=model.sc_type, 
        sc_inf=model.sc_inf, 
        G_concept=model.G_concept, 
        H_concept=model.H_concept, 
        G_sc=model.G_sc, 
        H_sc=model.H_sc,
        likelihood=model.likelihood, 
        n_samples=n_samples,
        whitening=model.whitening,
        )

    # Sanity check
    assert torch.any(torch.isinf(elbo))==False, "Invalid ELBO value (inf)."
    assert torch.any(torch.isnan(elbo))==False, "Invalid ELBO value (nan)."

    # Gradients
    loss = - elbo + w_concept*concept_loss + w_orth*orth_loss
    loss.backward()

    # Optimizer step
    if train_dec:
        #torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1)
        model.optimizer_decoder.step()
    if train_enc:
        #torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1)
        model.optimizer_encoder.step()

    return elbo, concept_loss, orth_loss, kl_div_concepts, kl_div_sc, reconstruction

    
    
def trainloop(model, train_dataloader, n_epochs, n_samples=1, train_enc=True, train_dec=True, w_concept=1, w_orth=1,verbose=True, wb=False, start_epoch=0):

    """
    Trainloop to train the model for a given number of epochs.
        
        Parameters
        ----------
        model : CodedDVAE instance
            Model to be trained.
        train_dataloader : torch Dataloader
            Dataloader with the training set.
        n_epochs: int
            Number of epochs.
        n_sampes : int, optional
            Number of samples used for computing the ELBO. The number of samples is 1 by default.
        train_enc : boolean, optional
            Flag to indicate if the parameters of the encoder need to be updated. True by default.
        train_enc : boolean, optional
            Flag to indicate if the parameters of the decoder need to be updated. True by default.
        w_concept: float, optional
            Weight for the concept loss term. Default 1.
        w_orth: float, optional
            Weight for the orthogonality loss term. Default 1.   
        verbose: boolean, optional
            Flag to print the ELBO during training. True by default.
        wb: boolean, optional
            Flag to log the ELBO, KL term and reconstruction term to Weights&Biases.
        start_epoch: int, optional
            Epoch where the trainloop starts. This is useful to obtain coherent logs in weights and biases when we finetune a model.
        
        Returns
        -------
        elbo_evolution : list
            List containing the ELBO values obtained during training (1 value per epoch).
        kl_concepts_evolution : list
            List containing the Kullback-Leibler divergence values for the concept distribution obtained during training (1 value per epoch).
        kl_sc_evolution : list
            List containing the Kullback-Leibler divergence values for the side channel distribution obtained during training (1 value per epoch).
        rec_evolution : list
            List containing reconstruction term values obtained during training (1 value per epoch).
        concept_loss_evolution : list
            List containing BCE values obtained during training (1 value per epoch).
        orth_loss_evolution : list
            List containing orthogobality loss values obtained during training (1 value per epoch).
    """

    elbo_evolution = []
    kl_concepts_evolution =[]
    kl_sc_evolution = []
    rec_evolution = []
    concept_loss_evolution = []
    orth_loss_evolution = []

    for e in range(start_epoch, n_epochs):

        elbo_epoch = 0
        kl_concepts_epoch = 0
        kl_sc_epoch = 0
        reconstruction_epoch = 0
        concept_loss_epoch = 0
        orth_loss_epoch = 0

        for x, concepts in train_dataloader:    # Batches
            
            elbo, concept_loss, orth_loss, kl_div_concepts, kl_div_sc, reconstruction = train_step(model, x, concepts, n_samples=n_samples, train_enc=train_enc, train_dec=train_dec, w_concept=w_concept, w_orth=w_orth)

            elbo_epoch += elbo.item()
            reconstruction_epoch += reconstruction.item()
            kl_concepts_epoch += kl_div_concepts.item()
            kl_sc_epoch += kl_div_sc.item()
            concept_loss_epoch += concept_loss.item()
            orth_loss_epoch += orth_loss.item()

        elbo_evolution.append(elbo_epoch/len(train_dataloader))     
        rec_evolution.append(reconstruction_epoch/len(train_dataloader)) 
        kl_concepts_evolution.append(kl_concepts_epoch/len(train_dataloader))
        kl_sc_evolution.append(kl_sc_epoch/len(train_dataloader))
        concept_loss_evolution.append(concept_loss_epoch/len(train_dataloader))
        orth_loss_evolution.append(orth_loss_epoch/len(train_dataloader))

        # Empty cache
        torch.cuda.empty_cache()   
        
        if wb:
            wandb.log({"elbo/epoch": elbo_epoch/len(train_dataloader),
                        "kl_concept/epoch":kl_concepts_epoch/len(train_dataloader),
                        "kl_sc/epoch":kl_sc_epoch/len(train_dataloader),
                        "reconstruction/epoch": reconstruction_epoch/len(train_dataloader),
                        "concept_loss/epoch": concept_loss_epoch/len(train_dataloader),
                        "orth_loss/epoch": orth_loss_epoch/len(train_dataloader),
                        "epoch:": e })

        
        if verbose:
            print("ELBO after %d epochs: %f" %(e+1, elbo_evolution[-1]))
        

    return elbo_evolution, concept_loss_evolution, orth_loss_evolution, kl_concepts_evolution, kl_sc_evolution, rec_evolution