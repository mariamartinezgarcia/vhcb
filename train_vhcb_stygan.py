"Based on https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm"

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
from models import clf_pseudolabeler
from src.utils.sampling import sample_from_qz_given_x
from src.train.loss import kl_div_bernoulli, kl_div_gaussian, log_gaussian
from src.vhcb_stygan import VHCB_StyGAN2
import time
import wandb

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--pseudo-label", type=str, default='supervised', help='choice of pseudo-label source: clip zero shot or supervised')
    parser.add_argument("-c", "--config-file", type=str, default='vhcb_stygan2_celebahq_train.yml', help='name of the config to load from /configs')
    parser.add_argument("-cp_name", "--checkpoint-name", default="vhcb_stygan2_celebahq", help="name for saving checkpoint")
    parser.add_argument("--load-pretrained", action='store_true', default=False, help='whether to load checkpoint from models/checkpoints/.')
    parser.add_argument("--pretrained-load-name", type=str, default='', help='filename to load from models/checkpoints/')
    args = parser.parse_args()
    
    # Load configuration file 
    with open('./configs/'+args.config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {args.config_file}")

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if (torch.cuda.is_available() and config["train_config"]["use_cuda"]):
        use_cuda=True
        device = torch.device("cuda")
    else:
        use_cuda=False
        device = torch.device("cpu")
    print('Using device: ', device)

    # Directory to save checkpoint
    if config["train_config"]["save_model"]:
        save_model_name =  f"{config['dataset']['name']}_{args.checkpoint_name}"

    # Model congifig
    model_type = config["model"]["type"]
    dataset = config["dataset"]["name"]

    if (torch.cuda.is_available() and config["train_config"]["use_cuda"]):
        use_cuda=True
        device = torch.device("cuda")
    else:
        use_cuda=False
        device = torch.device("cpu")


    # --- Weights and Biases --- # 

    if config["log"]["wandb"]:
    
        if config['model']["sc_type"] is None:
            wandb.init(
                project = config["log"]["wandb_project"],
                entity = config["log"]["wandb_user"],
                name = 'concept '+config["model"]["concept_inf"]+', sc None ,'+dataset.lower(),
                group = dataset,
                job_type = 'train',
                config=config
            )
        else:
            wandb.init(
                project = config["log"]["wandb_project"],
                entity = config["log"]["wandb_user"],
                name = 'concept '+config["model"]["concept_inf"]+', sc '+config["model"]["sc_type"]+', '+dataset.lower(),
                group = dataset,
                job_type = 'train',
                config=config
            )

    # Pretrained weights are already loaded for StyleGAN2 through config file
    model = VHCB_StyGAN2(config)
    model.to(device)

    # If a checkpoint is loaded
    if args.load_pretrained:
        print(f'loading checkpoint from models/checkpoints/{args.pretrained_load_name}')
        model.vhcb.load_state_dict(torch.load(f'{config["model"]["checkpoint"]}'))

    # Load pseudo-label model
    if args.pseudo_label == 'clip':
        print('using CLIP zero-shot for pseudo-labels')
        concept_set = config['model']['concepts']['concept_names']
        set_of_classes = config['model']['concepts']['set_of_classes']
        clf = clf_pseudolabeler.CLIP_PseudoLabeler(set_of_classes, device)

    elif args.pseudo_label == 'supervised':
        print('using supervised model for pseudo-labels')
        concept_set = config['model']['concepts']['concept_names']
        clsf_model_type = 'rn18'
        if dataset == 'cub' or dataset=='cub256' or dataset=='cub64':
            concept_set_checkpoint = [concept.replace(', ', '_').replace(' ', '_') for concept in concept_set]
            clf = clf_pseudolabeler.Sup_PseudoLabeler(concept_set_checkpoint, device, dataset=dataset, model_type=clsf_model_type)
        else:
            clf = clf_pseudolabeler.Sup_PseudoLabeler(concept_set, device, dataset=dataset, model_type=clsf_model_type)

    # freezing base generator parameters
    for param in model.gen.parameters():
        param.requires_grad = False

    # optimizer
    opt = torch.optim.Adam(model.vhcb.parameters(), lr=config["train_config"]["recon_lr"], betas=literal_eval(config["train_config"]["betas"]))

    # loss 
    reconstr_loss = torch.nn.MSELoss()
    beta_concepts = config['train_config']['beta_concepts']
    beta_sc = config['train_config']['beta_sc']

    # Training loop
    # In the post-hoc setting, batches of samples are generated using the base generative model 
    # instead of relying on the original dataset.
    # 1. Generate a batch of images using the base generative model
    # 2. Obtain pseudo-labels for the desired concepts from the generated images
    # 3. Pass the samples through the VHCB layer and compute the loss using the generated data 
    #    and their corresponding pseudo-labels
    steps_per_epoch = config["train_config"]["steps_per_epoch"]
    batch_size = config["dataset"]["batch_size"]
    for epoch in range(config["train_config"]["epochs"]):
        model.train()
        start = time.time()

        for i in range(steps_per_epoch):

            opt.zero_grad()

            ### 1. Sample latent noises
            z = torch.randn((batch_size, model.gen.z_dim), device=device)
            latent = model.gen.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)
            # generate images with the sampled latent
            gen_imgs_latent = model.gen.synthesis(latent, noise_mode='const')
            # to make it from -1 to 1 range to 0 to 1
            gen_imgs_latent = gen_imgs_latent.mul(0.5).add_(0.5)

            with torch.no_grad():
                # get probabilities and predicted labels from pseudolabeler
                # pseudo_probs are the probabilities of each concept being ACTIVE
                pseudo_prob, pseudo_labels = clf.get_pseudo_labels(gen_imgs_latent, return_prob=True)
                pseudo_prob = [pm.detach() for pm in pseudo_prob]
                pseudo_labels = [pl.detach() for pl in pseudo_labels]

            stacked_labels = torch.stack(pseudo_labels).T.float()
            stacked_probs = torch.stack(pseudo_prob).T
            
            elbo, loss_concepts, kl_div_sc, reconstruction_term, concept_probs, out_sc, out_decoder, z_sc, z_concept = model.vhcb.get_elbo(latent, stacked_probs, beta_concepts=beta_concepts, beta_sc = beta_sc)
            assert torch.any(torch.isinf(elbo))==False, "Invalid ELBO value (inf)."
            assert torch.any(torch.isnan(elbo))==False, "Invalid ELBO value (nan)."

            # generate images with the reconstructed latent
            gen_imgs_recon_latent = model.gen.synthesis(out_decoder, noise_mode='const')
            # to make it from -1 to 1 range to 0 to 1
            gen_imgs_recon_latent = gen_imgs_recon_latent.mul(0.5).add_(0.5)

            # include the MSE term in the loss
            img_recon_loss = reconstr_loss(gen_imgs_latent, gen_imgs_recon_latent)

            loss = -elbo + img_recon_loss

            loss.backward()
            opt.step()

            # Compute acc to keep track of the training process
            with torch.no_grad():

                labels = (concept_probs >= 0.5).int()
                acc = torch.sum(stacked_labels == labels, dim=1) / labels.shape[1]
                acc = torch.mean(acc)
        
            batches_done = epoch * steps_per_epoch + i
            if batches_done % config["train_config"]["log_interval"] == 0:
                print(
                    "Model %s Dataset %s [Epoch %d/%d] [Batch %d/%d] [total loss: %.4f] [kl concepts: %.4f] [avg lat rec: %.4f] [avg img rec: %.4f]"
                    % (model_type,dataset,epoch, config["train_config"]["epochs"], i, steps_per_epoch, loss.item(), torch.mean(loss_concepts).item(), elbo.item(), img_recon_loss.item())
                    )
                
            if config["log"]["wandb"]:
                wandb.log({
                    "batches_done": epoch * steps_per_epoch + i,
                    "img_recon_loss": img_recon_loss.item(),
                    "elbo": elbo.item(),
                    "reconstruction_term": torch.mean(reconstruction_term).item(),
                    "kl_concepts": torch.mean(loss_concepts).item(),
                    "kl_sc": torch.mean(kl_div_sc).item(),
                    "concept_acc": acc.item(),
                })

        if config["train_config"]["save_model"]:
            torch.save(model.vhcb.state_dict(), "models/checkpoints/"+save_model_name+"_vhcb.pt")
            print('Model saved in "models/checkpoints/'+save_model_name+'_vhcb.pt"')

        end = time.time()
        print("epoch time", end - start)
        print()

if __name__ == '__main__':
    main()