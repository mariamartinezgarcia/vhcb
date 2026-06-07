import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import itertools
import matplotlib.pyplot as plt

import random
from collections import Counter

from torch import nn

import wandb
import pickle
import argparse
import yaml
import os

import src.nn.modules as modules
from src.utils.sampling import modulate_words
from src.vhcb_stygan import VHCB_StyGAN2

from datasets.celeba import get_celeba_dataloader
from models import clf_pseudolabeler
from utils.datasets import CelebAHQ_dataset, ImageFolderDataset
from utils.evaluation import evaluate_single_concept_intervention, evaluate_concept_inference, evaluate_side_channel, evaluate_hamming_concept_intervention, evaluation_generation, load_labels

from torch.utils.data import Subset
from pprint import pprint


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config-file", type=str, default='vhcb_stygan2_celebahq_train.yml', help='config file to load from /configs')
    args = parser.parse_args()

    # --- Load configuration file --- #
    with open('./configs/'+args.config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {args.config_file}")

    #--- Select device ---#
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda =  config["model"]["use_cuda"] and  torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device= torch.device("cpu")

    #--- Dataset ---#
    dataset = config["dataset"]["name"]

    # --- Weights and Biases --- # 
    if config["log"]["wandb"]:
    
        if config['model']["sc_type"] is None:
            wandb.init(
                project = config["log"]["wandb_project"],
                entity = config["log"]["wandb_user"],
                name = 'eval concept '+config["model"]["concept_inf"]+', sc None ,'+dataset.lower(),
                group = dataset,
                job_type = 'eval',
                config=config
            )
        else:
            wandb.init(
                project = config["log"]["wandb_project"],
                entity = config["log"]["wandb_user"],
                name = 'eval concept '+config["model"]["concept_inf"]+', sc '+config["model"]["concept_inf"]+', '+dataset.lower(),
                group = dataset,
                job_type = 'eval',
                config=config
            )

    print(f"Using device: {device}")
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())

    # --- Build VHCB model from configuration file --- #
    model = VHCB_StyGAN2(config)
    model.to(device)
    print(f'loading pretrained VHCB checkpoint from {config["model"]["checkpoint"]}')
    model.vhcb.load_state_dict(torch.load(f'{config["model"]["checkpoint"]}', map_location=device))
    model.eval()

    # --- Pseudo-label source for evaluation --- #
    # We consider the same concept set as used for training the VHCB
    concept_set = config["model"]["concepts"]["concept_names"]
    if config['evaluation']['clf_type'] == 'clip':
        print('using CLIP zero-shot for pseudo-labels')
        set_of_classes = config["model"]["concepts"]["set_of_classes"]
        clf = clf_pseudolabeler.CLIP_PseudoLabeler(set_of_classes, device)
    elif config['evaluation']['clf_type'] == 'supervised':
        print('using supervised model for pseudo-labels')
        if dataset == 'cub' or dataset=='cub256' or dataset=='cub64':
            concept_set_checkpoint = [concept.replace(', ', '_').replace(' ', '_') for concept in concept_set]
            clf = clf_pseudolabeler.Sup_PseudoLabeler(concept_set_checkpoint, device, dataset=dataset, model_type=config['evaluation']['clf_architecture'])
        else:
            clf = clf_pseudolabeler.Sup_PseudoLabeler(concept_set, device, dataset=dataset, model_type=config['evaluation']['clf_architecture'])
    clf.eval()

    if 'unknown' in concept_set:
        concept_set.remove('unknown')

    # --- Number of steps to generate images and infer concepts --- #
    batch_size = config["evaluation"]["batch_size"] 
    n_steps = config["evaluation"]["n_steps"] 

    ######################################
    # --- BLOCK 1: CONCEPT INFERENCE --- #
    ######################################

    '''
    Assess how effectively the Concept Bottleneck captures the concepts.
        1. Generate a set of images with the pre-trained generative model.
        2. Project the latent into the concept space using the VHCB.
        3. Obtain pseudo-labels for the generated images using the specified classfier.
        4. Evaluate the concept inference performance by comparing the pseudo-labels with the ground truth labels.
    '''
    print('\nEvaluating concept inference...')
    evaluate_concept_inference(model, clf, concept_set, n_steps, batch_size, device, dataset, wb=config["log"]["wandb"])
    print('\nConcept inference evaluation completed!\n')
    
    
    ######################################################
    # --- BLOCK 2: SIDE CHANNEL(UNSUPERVISED LATENT) --- #
    ######################################################

    '''
    Evaluate whether concept information leaks through the side channel.
    To make this general (since in the baseline model the unsupervised latent cannot be sampled directly), proceed as follows:
        1. Sample two generative latent vectors.
        2. Pass them through the concept bottleneck to obtain: the latent concept vectors, and the latent side channel vectors.
        3. Pass the concept vector and side channel through the decoder to obtain the reconstructed latent.
        4. Pass the reconstructed latent throgh the pre-trained generative model to obtain the generated images.
        5. Obtain pseudo-labels for the generated images using the specified classifier.
        6. Swap the side channel vectors between the two samples. Forward the modified vectors to get the new pseudo-labels. 
           If the side channel does not carry concept information, these pseudo-labels should remain unchanged.
        7. Compare the original pseudo-labels from step 3 with the new pseudo-labels from step 5 to assess whether concept information leaks through the side channel.
    '''
    
    
    print('\nEvaluating whether concept information leaks through the side channel...')
    
    if config['model']['sc_type'] == 'binary' or config['model']['sc_type'] == 'continuous' or args.baseline:
        evaluate_side_channel(model, clf, concept_set, n_steps, batch_size, device, dataset, wb=config["log"]["wandb"])
        print('\nSide channel evaluation completed!\n')
    else:
        print('This model does not have side channel!\n')
    

    ##########################################
    # --- BLOCK 3: CONCEPT INTERVENTIONS --- #
    ##########################################

    '''
    Evaluate if we can do test-time interventions and control the generation of the pre-trained model. We evaluate if the target concept is observed 
    in the generated image and if the rest of non-target concepts have been affected by the intervention.
    To do this, we use latent vectors from the pre-trained generative model that are known in advance to produce images where the target concept is 
    either active or inactive.
    1. Single concept interventions.
        - Inactive -> Active 
        - Active -> Inactive
    '''
    
    # --- Single concept interventions --- #

    # Inactive -> Active
    print('\nEvaluating single concept interventions inactive -> active ...')
    evaluate_single_concept_intervention(model, clf, config['evaluation']['concepts_to_intervene'], concept_set, device, dataset, direction='i2a', wb=config["log"]["wandb"])
    print('\nSingle concept intervention inactive -> active completed!\n')

    # Active -> Inactive
    print('\nEvaluating single concept interventions active -> inactive ...')
    evaluate_single_concept_intervention(model, clf, config['evaluation']['concepts_to_intervene'], concept_set, device,  dataset, direction='a2i', wb=config["log"]["wandb"])
    print('\nSingle concept intervention active -> inactive completed!\n')
    
    '''
    2. Interventions based on minimum Hamming distance.
        Instead of intervening single concepts arbitrarilly, which can yield concept vectors that are very unlikely in the training data distribution 
        (for example, a man with make up or a woman with beard), we can intervene the set of concepts that would yield a concept vector that is in 
        distribution. We can find the target pattern at minimum Hamming to intervene the minimum number of concepts. As we are moving from a probable 
        concept set to another probable concept set, we expect the metrics to improve.
    '''
    
    # --- Interventions based on minimum Hamming distance --- #
    print('\nEvaluating concept interventions based on minimum Hamming distance...')
    evaluate_hamming_concept_intervention(model, clf, concept_set, n_steps, batch_size, device, dataset, wb=config["log"]["wandb"], most_probable=100)
    print('Concept interventions based on minimum Hamming distance evaluation completed!\n')
    

    #######################################
    # --- BLOCK 4: CONCEPT GENERATION --- #
    #######################################

    '''
    In our setup we can naturally generate images with specific distributions of concepts. This is different from interventions since, instead of modifying the output of 
    the encoder before forwarding it through the decoder, we can directly sample the side channel from the prior and fix the distribution we want for the concepts and 
    generate a latent accordingly.
    '''

    print('\nEvaluating concept generation (this evaluation is specific for the VHCB)...')
    evaluation_generation(model, clf, concept_set, n_steps, batch_size, device, dataset, wb=config["log"]["wandb"])
    print('\nGeneration evaluation completed!\n')

    print('\nEvaluating concept generation (this evaluation is specific for the VHCB) with the most probable concept patterns ...')
    evaluation_generation(model, clf, concept_set, n_steps, batch_size, device, dataset, wb=config["log"]["wandb"], most_probable=100)
    print('\nGeneration evaluation with the most probable concept patterns completed!\n')
    

if __name__ == '__main__':
    main() 

    

