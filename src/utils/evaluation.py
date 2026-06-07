import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy
from src.utils.sampling import sample_from_qc_given_x, sample_from_qz_given_x, modulate_words
from src.train.loss import compute_word_logprobs
from src.nn.modules import dclamp
from src.train.loss import log_gaussian
import pickle
import wandb
from pprint import pprint
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from sklearn.linear_model  import LogisticRegression

def compute_binary_entropy(bit_probs, words):

    """
    Compute entropy in base 2 of the distribution over words.

    Parameters
    ----------
    bit_probs : torch.tensor
        Bit probabilities.
    words : torch.tensor
        Codebook.

    Returns
    -------
    Entropy in base 2 of the distribution over words.
    """

    # Compute the entropy in base 2 of the distribution over words.  
    _, logq_norm = compute_word_logprobs(bit_probs, words)

    # Transform to probs to compute entropy in base 2
    probsq = torch.exp(logq_norm)
    entropy_qm = entropy(probsq.cpu().data.numpy(), base=2, axis=1)

    return torch.mean(torch.tensor(entropy_qm))


def get_latent_projections(model, dataset):

    """
    Obtain latent projections of data points.

    Parameters
    ----------
    model : CodedVAE instance
        Model.
    dataset : torch Dataset
        Dataset for which the projections will be computed.

    Returns
    -------
    projections : numpy.ndarray
        Latent projections of the observations in dataset.
    labels_projections:  numpy.ndarray
        Labels of the projected observations.
    """

    # Generate an auxiliar dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)

    # Encode training data
    first=True
    for images, labels in dataloader:

        if model.inference == 'uncoded':
            # Encode and sample
            bit_probs = model.encoder.forward(images.to(model.device))
            z_sample = sample_from_qz_given_x(bit_probs, beta=model.beta)

        if model.inference == 'rep':
            # Encode and sample
            encoder_out = model.encoder.forward(images.to(model.device))
            logpm1 = torch.matmul(torch.log(encoder_out), model.H.to(model.device))
            logpm0 = torch.matmul(torch.log(1-encoder_out), model.H.to(model.device))

            log_marginals = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

            bit_probs = torch.exp(log_marginals_norm[:,:,1])
            z_sample = sample_from_qz_given_x(bit_probs, beta=model.beta, n_samples=1)

        # Concatenate the projections
        if first:
            labels_projections = labels.data.numpy()
            projections = z_sample.cpu().data.numpy()
            first=False
        else:
            projections = np.concatenate((projections, z_sample.cpu().data.numpy()), axis=0)
            labels_projections = np.concatenate((labels_projections, labels.data.numpy()), axis=0)

    return projections, labels_projections


def get_confident_latent_projections(model, dataset, words, threshold=0.8):

    """
    Obtain latent projections of data points. We only consider projections with a probability larger than an established threshold.

    Parameters
    ----------
    model : CodedVAE instance
        Model.
    dataset : torch Dataset
        Dataset for which the projections will be computed.
    words : torch.tensor
        Codebook.
    threshold : float
        Threshold to consider a projection confident.

    Returns
    -------
    projections : numpy.ndarray
        Latent projections of the observations in dataset.
    labels_projections:  numpy.ndarray
        Labels of the projected observations.
    """

    # Generate an auxiliar dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)

    # Encode training data
    first=True
    for images, labels in dataloader:

        if model.inference == 'uncoded':
            # Encode and sample
            bit_probs = model.encoder.forward(images.to(model.device))
            _, logq_norm = compute_word_logprobs(bit_probs, words.to(model.device))
            row_indices = (torch.exp(logq_norm).max(dim=1).values > threshold).nonzero(as_tuple=True)[0]
            z_sample = sample_from_qz_given_x(bit_probs[row_indices,:], beta=model.beta)
            labels = labels[row_indices.cpu()]
            
        if model.inference == 'rep':
            # Encode and sample
            encoder_out = model.encoder.forward(images.to(model.device))
            logpm1 = torch.matmul(torch.log(encoder_out), model.H.to(model.device))
            logpm0 = torch.matmul(torch.log(1-encoder_out), model.H.to(model.device))

            log_marginals = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

            bit_probs = torch.exp(log_marginals_norm[:,:,1])

            _, logq_norm = compute_word_logprobs(bit_probs, words.to(model.device))
            row_indices = (torch.exp(logq_norm).max(dim=1).values > threshold).nonzero(as_tuple=True)[0]
            z_sample = sample_from_qz_given_x(bit_probs[row_indices,:], beta=model.beta)
            labels = labels[row_indices.cpu()]

        # Concatenate the projections
        if first:
            labels_projections = labels.data.numpy()
            projections = z_sample.cpu().data.numpy()
            first=False
        else:
            projections = np.concatenate((projections, z_sample.cpu().data.numpy()), axis=0)
            labels_projections = np.concatenate((labels_projections, labels.data.numpy()), axis=0)

    return projections, labels_projections


def get_latent_words(model, dataset):

    """
    Obtain latent words of data points. 

    Parameters
    ----------
    model : CodedVAE instance
        Model.
    dataset : torch Dataset
        Dataset for which the projections will be computed.

    Returns
    -------
    projections : numpy.ndarray
        Latent words of the observations in dataset.
    labels_projections:  numpy.ndarray
        Labels of the projected observations.
    """

    # Generate an auxiliar dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)

    # Encode training data
    first=True
    for images, labels in dataloader:

        if model.inference == 'rep':
            # Encode and sample
            encoder_out = model.encoder.forward(images.to(model.device))
            logpm1 = torch.matmul(torch.log(encoder_out), model.H.to(model.device))
            logpm0 = torch.matmul(torch.log(1-encoder_out), model.H.to(model.device))

            log_marginals = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

            words = torch.bernoulli(torch.exp(log_marginals_norm[:,:,1]))

        # Concatenate the projections
        if first:
            labels_projections = labels.data.numpy()
            projections = words.cpu().data.numpy()
            first=False
        else:
            projections = np.concatenate((projections, words.cpu().data.numpy()), axis=0)
            labels_projections = np.concatenate((labels_projections, labels.data.numpy()), axis=0)
        
    return projections, labels_projections


# ---- Class error in reconstruction ---- #

class ClassifierNetwork(nn.Module):
    """
    Class implementing a CNN-based classifier.
    """

    def __init__(self):
        super(ClassifierNetwork, self).__init__()

        """
        Initialize an instance of the class.
        """

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=1024)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

        self.logsoftmax = nn.LogSoftmax(dim=1)  

    def forward(self, x):

        """
        Forward pass.

        Parameters
        ----------
        x: torch.tensor
            Batch of data.
        """

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.logsoftmax(out)
        
        return out


def eval_reconstruction(classifier_network, model, dataloader, code_words=None, threshold=None):

    """
    Evaluate the reconstruction accuracy.

    Parameters
    ----------
    classifier_network : ClassifierNetwork instance
        Classifier.
    model : CodedVAE instance
        Model we are evaluating.
    dataloader : torch Dataloader
        Dataloader of a given dataset.
    code_words: torch.tensor, optional
        Codebook.
    threshold: float
        Threshold to consider a projection confident.

    Returns
    -------
    Reconstruction accuracy.

    """

    # Evaluation mode
    classifier_network.eval()

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():

        same_class = 0
        num_data_points = 0
        for images, labels in dataloader:
                
            if (not (threshold is None)) and (not (code_words is None)):

                bit_probs = model.encoder.forward(images.to(model.device))
                if model.inference == 'rep':
                    logpm1 = torch.matmul(torch.log(bit_probs), model.H.to(model.device))
                    logpm0 = torch.matmul(torch.log(1-bit_probs), model.H.to(model.device))
                    log_marginals = torch.stack((logpm0, logpm1), dim=2)
                    log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)
                    bit_probs = torch.exp(log_marginals_norm[:,:,1])
                _, logq_norm = compute_word_logprobs(bit_probs, code_words.to(model.device))
                row_indices = (torch.exp(logq_norm).max(dim=1).values > threshold).nonzero(as_tuple=True)[0]
                if len(row_indices) == 0:
                    continue
                images = images[row_indices.cpu()]
                labels = labels[row_indices.cpu()]
        
            _, reconstructed = model.forward(images)
            
            probs = classifier_network(reconstructed)
            pred = np.argmax(probs.cpu().detach().numpy(), axis=1)

            same_class += (labels.data.numpy()==pred).sum()
            num_data_points += images.shape[0]

    return same_class/num_data_points


