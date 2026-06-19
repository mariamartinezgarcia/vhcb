import os
import sys
sys.path.append('.')

import argparse
import numpy as np
import yaml
import torch
from ast import literal_eval
from torchvision.utils import save_image
from torch import nn
from src.vhcb_stygan import VHCB_StyGAN2
from models import clf_pseudolabeler
from src.utils.sampling import sample_from_qz_given_x
from src.train.loss import kl_div_bernoulli, kl_div_gaussian, log_gaussian
from torchvision import transforms, models
import pickle
import matplotlib.pyplot as plt

def load_labels(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    label_lines = lines[1:]  # skip header
    labels = np.array([
        [int(x) for x in line.strip().split()[1:]]  # skip image ID
        for line in label_lines
    ])
    # Convert -1 -> 0
    labels = np.where(labels == -1, 0, labels)
    return header, labels

def generate_images(model, save_image_loc="images_eval/", num_samples=100, dataset='celebahq', device='cuda', set_of_classes=[], most_probable=None):
    
    # Disable gradient tracking since we're only doing inference, not training
    with torch.no_grad():

        # Sample noise and labels as generator input
        if most_probable is not None:
            # Generate latents conditioned on a specific concept configuration
            latent, concept_sample = model.vhcb.generate(n_samples=num_samples, device=device, m=most_probable)
        else:
            # Generate latents by sampling freely from the prior (unconditional / random sampling)
            latent, concept_sample = model.vhcb.generate(n_samples=num_samples, device=device)

        # Decode the sampled latents into images using the generator's synthesis network
        # noise_mode='const' disables per-pixel stochastic noise injection for deterministic output
        gen_imgs = model.gen.synthesis(latent, noise_mode='const')

        # Rescale pixel values from [-1, 1] (typical generator output range) to [0, 1]
        gen_imgs = gen_imgs.mul(0.5).add_(0.5)

        # Convert to a plottable format:
        # scale to [0, 255], clamp to valid range, move channels last (C,H,W -> H,W,C),
        # move off GPU, convert to numpy, and cast to int for imshow
        gen_imgs_plot = gen_imgs.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).detach().cpu().numpy().astype(int)

        # Set up a 10 x (num_samples/10) grid of subplots to display all generated images
        fig, axes = plt.subplots(10, int(num_samples/10), figsize=(25, 25))
        for ax, img in zip(axes.flat, gen_imgs_plot):
            # Plot each image; use grayscale colormap only if the image has no channel dimension
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.axis('off')  # hide axis ticks/labels for a cleaner grid

        # Tighten spacing between subplots
        plt.subplots_adjust(wspace=0.02, hspace=0.02)

        # Save with a different filename depending on whether this was random sampling
        # or sampling around the most probable concept
        if most_probable is None:
            plt.savefig(save_image_loc+'generated_sampling.png', dpi=150, bbox_inches='tight')
        else:
            plt.savefig(save_image_loc+'generated_most_probable.png', dpi=150, bbox_inches='tight')

        plt.close()  
    return


def images_single_concept_intervention(model, concepts_to_intervene, concept_set, device, direction='i2a', save_image_loc="images_eval/", dataset='celebahq'):
   
    with torch.no_grad():

        print('\n\nConcepts to intervene: ', concepts_to_intervene)

        for target_concept in concepts_to_intervene:

            if direction =='i2a':

                original_state = 'inactive'
                target_state = 'active'

                # --- Inactive -> Active intervention --- #
                print(f'\n\tIntervening {target_concept} from inactive to active...')

                if dataset == 'celebahq':
                    # Load generated latents with the target concept inactive
                    if os.path.exists('eval_latents_rn50_celebahq/'+target_concept+'/inactive.pkl'):
                        try:
                            with open('eval_latents_rn50_celebahq/'+target_concept+'/inactive.pkl', 'rb') as f:
                                dict_latent = pickle.load(f)
                        except EOFError:
                            print('[EOFError] Cannot open file eval_latents_rn50/'+target_concept+'/inactive.pkl')
                            continue
                    else:
                        continue
                if dataset == 'cub':
                    # Load generated latents with the target concept inactive
                    print('eval_latents_rn50_cub/'+target_concept+'/inactive.pkl')
                    if os.path.exists('eval_latents_rn50_cub/'+target_concept+'/inactive.pkl'):
                        try:
                            with open('eval_latents_rn50_cub/'+target_concept+'/inactive.pkl', 'rb') as f:
                                dict_latent = pickle.load(f)
                        except EOFError:
                            print('[EOFError] Cannot open file eval_latents_rn50_cub/'+target_concept+'/inactive.pkl')
                            continue
                    else:
                        continue
            
            if direction == 'a2i':

                original_state = 'active'
                target_state = 'inactive'

                # --- Active -> Inactive intervention --- #
                print(f'\n\tIntervening {target_concept} from active to inactive...')
                if dataset == 'celebahq':
                    # Load generated latents with the target concept inactive
                    if os.path.exists('eval_latents_rn50_celebahq/'+target_concept+'/active.pkl'):

                        try:
                            with open('eval_latents_rn50_celebahq/'+target_concept+'/active.pkl', 'rb') as f:
                                dict_latent = pickle.load(f)
                        except EOFError:
                            print('[EOFError] Cannot open file eval_latents_rn50/'+target_concept+'/active.pkl')
                            continue
                    else:
                        continue
                if dataset == 'cub':
                    # Load generated latents with the target concept inactive
                    print('eval_latents_rn50_cub/'+target_concept+'/active.pkl')
                    if os.path.exists('eval_latents_rn50_cub/'+target_concept+'/active.pkl'):
                        try:
                            with open('eval_latents_rn50_cub/'+target_concept+'/active.pkl', 'rb') as f:
                                dict_latent = pickle.load(f)
                        except EOFError:
                            print('[EOFError] Cannot open file eval_latents_rn50_cub/'+target_concept+'/active.pkl')
                            continue
                    else:
                        continue

            original = []
            intervened = []

            if dataset == 'cub':
                target_concept = target_concept.replace(', ', '_').replace(' ', '_')

            for j in range(10):
                # Select concept we want to intervene (recall that the concept set is the set of concepts used for training)
                idx_concept = concept_set.index(target_concept)

                # Get latent from the dict 
                latent = dict_latent[str(j)]['latent']

                # Get the indices to map the concepts used for training to the whole concept set
                all_concepts = dict_latent[str(j)]['concept_names']
                if dataset == 'cub':
                    all_concepts = [concept.replace(', ', '_').replace(' ', '_') for concept in all_concepts]
                indices = [all_concepts.index(item) for item in concept_set]

                n_repetitions = int(model.vhcb.bits_code_concept/model.vhcb.n_concepts)
                probs_concepts, out_sc = model.vhcb.encode(latent.to(device))

                # Uncoded case
                if model.vhcb.concept_inf == 'uncoded':
                    # Sample z 
                    z_concept = sample_from_qz_given_x(probs_concepts, beta=model.vhcb.beta, n_samples=1)

                # Coded case
                if model.vhcb.concept_inf == 'rep':
                    # Introduce code structure
                    qc = torch.matmul(probs_concepts, model.vhcb.G_concept.to(device))
                    # Sample z
                    z_concept = sample_from_qz_given_x(qc, beta=model.vhcb.beta, n_samples=1)

                # Infer side channel posterior, sample z_sc
                z_sc = None

                if model.vhcb.sc_inf == 'uncoded':
                    #Sample z
                    z_sc = sample_from_qz_given_x(out_sc, beta=model.vhcb.beta, n_samples=1)

                if model.vhcb.sc_inf == 'rep':
                    # Introduce code structure
                    qc = torch.matmul(out_sc, model.vhcb.G_sc.to(device))
                    # Sample z
                    z_sc = sample_from_qz_given_x(qc, beta=model.vhcb.beta, n_samples=1)
                
                if z_sc is None:
                    latent_sample = z_concept[:,:,0]
                else:
                    latent_sample = torch.cat((z_concept[:,:,0], z_sc[:,:,0]), dim=1)
    
                out_decoder = model.vhcb.decode(latent_sample)
                gen_imgs_recon_latent = model.gen.synthesis(out_decoder, noise_mode='const')

                # to make it from -1 to 1 range to 0 to 1
                gen_imgs_recon_latent = gen_imgs_recon_latent.mul(0.5).add_(0.5)

                original.append(gen_imgs_recon_latent.cpu())
                
                # Select concept we want to intervene
                idx_z_concept = int(idx_concept*n_repetitions)

                if direction =='i2a':
                    # Concept active
                    concept_intervened_probs = probs_concepts.clone()
                    concept_intervened_probs[:, idx_concept] = 1.0
    
                if direction == 'a2i':
                    # Concept inactive
                    concept_intervened_probs = probs_concepts.clone()
                    concept_intervened_probs[:, idx_concept] = 0.0
                    
                # Only change the z corresponding to the intervened concept!
                qc_concept_intervened = torch.matmul(concept_intervened_probs, model.vhcb.G_concept.to(device))
                z_concept_intervened_sampled = sample_from_qz_given_x(qc_concept_intervened, beta=model.vhcb.beta, n_samples=1)
                z_concept_intervened = z_concept.clone()
                z_concept_intervened[:,idx_z_concept:idx_z_concept+n_repetitions] = z_concept_intervened_sampled[:,idx_z_concept:idx_z_concept+n_repetitions]

                # Incorporate the side channel
                if z_sc is None:
                    latent_sample_intervened = z_concept_intervened[:,:,0]
                else:
                    latent_sample_intervened = torch.cat((z_concept_intervened[:,:,0], z_sc[:,:,0]), dim=1)

                # Forward decoder
                out_decoder_intervened = model.vhcb.decode(latent_sample_intervened)

                # Generate image
                gen_imgs_recon_latent_intervened = model.gen.synthesis(out_decoder_intervened, noise_mode='const')
                # to make it from -1 to 1 range to 0 to 1
                gen_imgs_recon_latent_intervened = gen_imgs_recon_latent_intervened.mul(0.5).add_(0.5)

                intervened.append(gen_imgs_recon_latent_intervened.cpu())

            # Set up a 2x10 grid of subplots to display all generated images
            fig, axes = plt.subplots(2, 10, figsize=(20, 4))
            # Fill the first row with original images
            for col, img in enumerate(original):
                img = img.squeeze(0).mul(255).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(int)
                ax = axes[0, col]
                ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                ax.axis('off')

            # Fill the second row with intervened images
            for col, img in enumerate(intervened):
                #img = img.squeeze(0).permute(1, 2, 0)
                img = img.squeeze(0).mul(255).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(int)
                ax = axes[1, col]
                ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                ax.axis('off')

            # Tighten spacing between subplots
            plt.subplots_adjust(wspace=0.02, hspace=0.02)

            # Save 
            plt.savefig(save_image_loc+"intervention_"+target_concept+"_"+original_state+"_to_"+target_state+".png", dpi=150, bbox_inches='tight')
            plt.close()  
        
    return

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config-file", type=str, default='vhcb_stygan2_celebahq_eval.yml', help='config file to load from /configs')
    parser.add_argument("-f", "--folder-name", type=str, default='vhcb_stygan2_celebahq', help='name of the folder to save the results')
    args = parser.parse_args()

    # --- Load configuration file --- #
    with open('./configs/'+args.config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {args.config_file}")

    # --- Select device --- #
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda =  config["model"]["use_cuda"] and  torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device= torch.device("cpu")

    print(f"Using device: {device}")
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())

    # --- Dataset --- #
    dataset = config["dataset"]["name"]

    # --- Concepts --- #
    concept_set = config["model"]["concepts"]["concept_names"]
    if dataset == 'cub':
        concept_set = [concept.replace(', ', '_').replace(' ', '_') for concept in concept_set]
 
    # --- Build VHCB model from configuration file --- #
    model = VHCB_StyGAN2(config)
    model.to(device)
    print(f'loading pretrained VHCB checkpoint from {config["model"]["checkpoint"]}')
    model.vhcb.load_state_dict(torch.load(f'{config["model"]["checkpoint"]}', map_location=device))
    model.eval()

    # --- Create folder for results --- #
    save_image_loc = "images_eval/"+args.folder_name+"/"
    os.makedirs(save_image_loc, exist_ok=True)
    concepts_to_intervene = config['evaluation']['concepts_to_intervene']

    # --- Random generation --- #
    generate_images(model, save_image_loc, device='cuda', dataset=dataset, set_of_classes=concept_set)
    torch.cuda.empty_cache()

    # --- Pattern-based generation --- #
    if dataset == 'celebahq':
        header, labels_train = load_labels("./datasets/CelebAMask-HQ/train.txt")
        patterns = labels_train
        # Select the concepts used for training the VHCB
        indices = [header.index(item) for item in concept_set]
    if dataset == 'cub':
        header, patterns = load_labels("./datasets/CUB_200_2011/attributes/train_patterns.txt")
        indices = [219, 236, 55, 290, 152, 21, 245, 7, 36, 52]

    # Select the concepts used for training the VHCB
    #indices = [header.index(item) for item in concept_set]
    patterns = patterns[:, indices]
    # Obtain counts
    unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
    # Get indices that would sort counts in descending order
    sorted_indices = np.argsort(-counts)
    # Reorder both arrays
    sorted_patterns = unique_patterns[sorted_indices]
    indices = np.random.choice(20, 100, replace=True)
    # Select the patterns
    most_probable_patterns = torch.FloatTensor(sorted_patterns[indices])
    generate_images(model, save_image_loc, device='cuda', dataset=dataset, set_of_classes=concept_set, most_probable=most_probable_patterns)
    torch.cuda.empty_cache()

    # --- Single concept intervention (inactive -> active) --- #
    images_single_concept_intervention(model, concepts_to_intervene, concept_set, device, direction='i2a', save_image_loc=save_image_loc, dataset=dataset)
    torch.cuda.empty_cache()

    # --- Single concept intervention (active -> inactive) --- #
    images_single_concept_intervention(model, concepts_to_intervene, concept_set, device, direction='a2i', save_image_loc=save_image_loc, dataset=dataset)
    torch.cuda.empty_cache()
    
if __name__ == '__main__':
    main()