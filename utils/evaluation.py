from torch import nn
import torch
import numpy as np
import pickle
import wandb
import os
from pprint import pprint
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from sklearn.linear_model  import LogisticRegression

def kl_bern(a, b):
    return a * torch.log(a / b) + (1 - a) * torch.log((1 - a) / (1 - b))


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()

    def forward(self, p: torch.tensor, q: torch.tensor):

        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))

        p = torch.clamp(p,1e-3, 1-1e-3)
        q = torch.clamp(q,1e-3, 1-1e-3)

        m = 0.5 * (p + q)

        kl_pm = kl_bern(p, m)
        kl_qm = kl_bern(q, m)

        per_coord_jsd = 0.5 * (kl_pm + kl_qm)
        total_jsd = torch.sum(per_coord_jsd, dim=1)

        return torch.mean(total_jsd)

def total_variation_dist(p,q):
    """
    Computes the Total Variation Distance between two probability distributions p and q.
    p and q should be [batch_size, n_dim] tensors of the same length.
    """
    if p.shape[1] != q.shape[1]:
        raise ValueError("p and q must have the same length")

    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))

    return torch.mean(torch.mean(torch.abs(p - q), dim=1), dim=0)


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

def find_closest_patterns(concepts, pattern_ranking):
    # Expand a and b to shape [batch_size, 200, n_concepts]
    a_exp = concepts.unsqueeze(1)                  # [batch_size, 1, n_concepts]
    b_exp = pattern_ranking.unsqueeze(0)                  # [1, 200, n_concepts]

    # Compute Hamming distance: count where bits differ
    hamming = torch.sum(a_exp != b_exp, dim=2) # [batch_size, 200]

    # Mask out exact matches (distance 0)
    hamming_masked = hamming.clone()
    hamming_masked[hamming_masked == 0] = concepts.shape[1]

    # Get indices of the minimum Hamming distances
    min_indices = torch.argmin(hamming_masked, dim=1)   # [batch_size]

    # Gather the corresponding patterns from b
    c = pattern_ranking[min_indices]  
    # [batch_size, n_concepts]
    return c, hamming[torch.arange(hamming.shape[0]),min_indices]


def intervene_single_concept(model, cb_concepts, cb_side_channel, target_idx, target_value=1.):

    '''
    Intervene one single concept.
    '''

    probs_concept_interv = cb_concepts.clone()
    probs_concept_interv[:, target_idx] = target_value

    # Sample latent concept vector 
    _, z_concepts_interv_sample, sc_sample = model.vhcb.sample_from_latent_dist(probs_concept_interv, cb_side_channel, test=True, map=True)

    # Obtain latent with the target concept intervened
    if sc_sample is None:
        recon_latent_interv = model.vhcb.decode(z_concepts_interv_sample)
    else:
        recon_latent_interv = model.vhcb.decode(torch.cat((z_concepts_interv_sample, sc_sample), dim=1))

    return recon_latent_interv


def intervene_multiple_concepts(model, hard_pseudo_labels, cb_concepts, cb_side_channel, target_concepts):

    # Obtain a mask indicating which concepts we need to intervene for each image
    target_concepts = target_concepts.to(cb_concepts.device)
    hard_pseudo_labels = hard_pseudo_labels.to(cb_concepts.device)
    mask = hard_pseudo_labels != target_concepts

    # Intervene the concept posterior probabilities assigning the target value
    probs_concepts_interv = cb_concepts.clone()
    probs_concepts_interv[mask] = target_concepts[mask].float() # we can set directly to {0,1} because we are sampling hard bits

    # Sample latent concept vector 
    _, z_concepts_interv_sample, sc_sample = model.vhcb.sample_from_latent_dist(probs_concepts_interv, cb_side_channel, test=True, map=True)


    # Obtain latent with the target concepts intervened
    if sc_sample is None:
        recon_latent_interv = model.vhcb.decode(z_concepts_interv_sample)
    else:
        recon_latent_interv = model.vhcb.decode(torch.cat((z_concepts_interv_sample, sc_sample), dim=1))

    return mask, recon_latent_interv

def evaluate_concept_inference(model, clf, concept_set, n_steps, batch_size, device, dataset, wb=True):

    '''
    Assess how effectively the Concept Bottleneck captures the concepts.
        1. Generate a set of images with the pre-trained generative model.
        2. Project the latent into the concept space using the VHCB.
        3. Obtain pseudo-labels for the generated images using the specified classfier.
        4. Evaluate the concept inference performance by comparing the pseudo-labels with the ground truth labels.
    '''

    concept_probs = []
    concept_labels = []
    soft_pseudo_labels = []
    hard_pseudo_labels = []

    with torch.no_grad():
        # Loop over the number of steps to generate images and infer concepts
        for step in range(n_steps):

            # Sample a noise vector and obtain the latent
            z = torch.randn((batch_size, model.gen.z_dim), device=device)
            latent = model.gen.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)

            # Forward the latent through the pre-trained generative model to get the generated images
            gen_imgs = model.gen.synthesis(latent, noise_mode='const')
            gen_imgs = gen_imgs.mul(0.5).add_(0.5) # Rescale it from -1 to 1 range to 0 to 1

            # Obtain pseudo-labels for the generated images
            probs_pseudo_labels, pseudo_labels = clf.get_pseudo_labels(gen_imgs, return_prob=True)

            # Forward the latent through encoder to get the predicted concepts
            probs, _ = model.vhcb.encode(latent)
            labels = (probs.clone() >= 0.5).int()

            # Store the results
            concept_probs.append(probs.cpu())
            concept_labels.append(labels.cpu())
            soft_pseudo_labels.append([t.cpu() for t in probs_pseudo_labels])
            hard_pseudo_labels.append([t.cpu() for t in pseudo_labels])

        concept_probs = torch.cat(concept_probs, dim=0)
        concept_labels = torch.cat(concept_labels, dim=0)
        soft_pseudo_labels = torch.stack([torch.stack(lst, dim=1) for lst in soft_pseudo_labels], dim=0)
        soft_pseudo_labels = soft_pseudo_labels.view(-1, len(concept_set))  # Reshape to [n_samples, n_concepts]
        hard_pseudo_labels = torch.stack([torch.stack(lst, dim=1) for lst in hard_pseudo_labels], dim=0)
        hard_pseudo_labels = hard_pseudo_labels.view(-1, len(concept_set))  # Reshape to [n_samples, n_concepts]

        # -- Compare hard pseudo-labels with hard concepts -- #
        # Accuracy per concept
        concept_acc = (hard_pseudo_labels == concept_labels).float().mean(dim=0)
        mean_concept_acc = concept_acc.mean().item()
        dict_concept_acc = {concept_set[i]: concept_acc[i].item() for i in range(len(concept_set))}
        print(f'\n\tConcept inference accuracy: {dict_concept_acc}')
        print(f'\n\tMean concept inference accuracy: {mean_concept_acc:.4f}')
        if wb:
            wandb.log({"concept_inference/mean_concept_acc": mean_concept_acc})
            for concept, acc in dict_concept_acc.items():
                wandb.log({f"concept_inference/{concept}_acc": acc})
        
        # -- Compare soft pseudo-labels with posterior probs -- #
        # Cosine similarity
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        concept_cos_sim = torch.mean(cos_sim(concept_probs, soft_pseudo_labels))
        print(f'\tCosine similarity between concept probabilities and pseudo-labels: {concept_cos_sim.item():.4f}')
        if wb:
            wandb.log({"concept_inference/cosine_similarity": torch.mean(concept_cos_sim).item()})
        
        # Total Variation Distance
        tv_dist = total_variation_dist(concept_probs, soft_pseudo_labels)
        print(f'\tTotal Variation Distance between concept probabilities and pseudo-labels: {tv_dist.item():.4f}')
        if wb:
            wandb.log({"concept_inference/total_variation_distance": tv_dist.item()})

        # Jensen-Shannon Divergence
        js_divergence = JSD()
        js_div = js_divergence(concept_probs, soft_pseudo_labels).item()
        print(f'\tJensen-Shannon Divergence between concept probabilities and pseudo-labels: {js_div:.4f}')
        if wb:
            wandb.log({"concept_inference/jensen_shannon_divergence": js_div})

    return


def evaluate_side_channel(model, clf, concept_set, n_steps, batch_size, device, dataset, wb=True):

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

    ground_truth_labels = []
    ground_truth_probs = []
    swap_labels = []
    swap_probs = []

    n_imgs = n_steps*batch_size

    with torch.no_grad():
  
        for step in range(0, n_imgs, 2):
            
            # Sample two noise vectors and obtain the latents
            z1 = torch.randn((1, model.gen.z_dim), device=device)
            z2 = torch.randn((1, model.gen.z_dim), device=device)
            latent1 = model.gen.mapping(z1, None, truncation_psi=1.0, truncation_cutoff=None)
            latent2 = model.gen.mapping(z2, None, truncation_psi=1.0, truncation_cutoff=None)

            # Forward the latents through VHCB encoder to get the predicted concepts and side channels
            probs1, side_channel1 = model.vhcb.encode(latent1)
            probs2, side_channel2 = model.vhcb.encode(latent2)
            _, z_concept_sample1, sc_sample1 = model.vhcb.sample_from_latent_dist(probs1, side_channel1, test=True, map=True)
            _, z_concept_sample2, sc_sample2 = model.vhcb.sample_from_latent_dist(probs2, side_channel2, test=True, map=True)
    
            # Mix the concept and the side channel vectors and btain the reconstructed latents
            recon_latent1 = model.vhcb.decode(torch.cat((z_concept_sample1, sc_sample1), dim=1))
            recon_latent2 = model.vhcb.decode(torch.cat((z_concept_sample2, sc_sample2), dim=1))
            recon_latent12 = model.vhcb.decode(torch.cat((z_concept_sample1, sc_sample2), dim=1))
            recon_latent21 = model.vhcb.decode(torch.cat((z_concept_sample2, sc_sample1), dim=1))

            # Forward the latents through the pre-trained generative model to get the generated images
            gen_imgs1 = model.gen.synthesis(recon_latent1, noise_mode='const')
            gen_imgs2 = model.gen.synthesis(recon_latent2, noise_mode='const')
            gen_imgs12 = model.gen.synthesis(recon_latent12, noise_mode='const')
            gen_imgs21 = model.gen.synthesis(recon_latent21, noise_mode='const')
            gen_imgs1 = gen_imgs1.mul(0.5).add_(0.5)
            gen_imgs2 = gen_imgs2.mul(0.5).add_(0.5)
            gen_imgs12 = gen_imgs12.mul(0.5).add_(0.5)
            gen_imgs21 = gen_imgs21.mul(0.5).add_(0.5)

            # Obtain pseudo-labels for the generated images
            probs_pseudo_labels1, pseudo_labels1 = clf.get_pseudo_labels(gen_imgs1, return_prob=True)
            probs_pseudo_labels2, pseudo_labels2 = clf.get_pseudo_labels(gen_imgs2, return_prob=True)
            probs_pseudo_labels12, pseudo_labels12 = clf.get_pseudo_labels(gen_imgs12, return_prob=True)
            probs_pseudo_labels21, pseudo_labels21 = clf.get_pseudo_labels(gen_imgs21, return_prob=True)

            # Store the results
            ground_truth_labels.append([t.cpu() for t in pseudo_labels1])
            ground_truth_labels.append([t.cpu() for t in pseudo_labels2])
            ground_truth_probs.append([t.cpu() for t in probs_pseudo_labels1])
            ground_truth_probs.append([t.cpu() for t in probs_pseudo_labels2])
            swap_labels.append([t.cpu() for t in pseudo_labels12])
            swap_labels.append([t.cpu() for t in pseudo_labels21])
            swap_probs.append([t.cpu() for t in probs_pseudo_labels12])
            swap_probs.append([t.cpu() for t in probs_pseudo_labels21])

        ground_truth_labels = torch.stack([torch.stack(lst, dim=1) for lst in ground_truth_labels], dim=0)
        ground_truth_labels = ground_truth_labels.view(-1, len(concept_set))  # Reshape to [n_samples, n_concepts]
        ground_truth_probs = torch.stack([torch.stack(lst, dim=1) for lst in ground_truth_probs], dim=0)
        ground_truth_probs = ground_truth_probs.view(-1, len(concept_set))  # Reshape to [n_samples, n_concepts]
        swap_labels = torch.stack([torch.stack(lst, dim=1) for lst in swap_labels], dim=0)
        swap_labels = swap_labels.view(-1, len(concept_set))  # Reshape to [n_samples, n_concepts]
        swap_probs = torch.stack([torch.stack(lst, dim=1) for lst in swap_probs], dim=0)
        swap_probs = swap_probs.view(-1, len(concept_set))  # Reshape to [n_samples, n_concepts]

    # -- Compare hard pseudo-labels -- #
    # Accuracy per concept
    concept_acc = (ground_truth_labels == swap_labels).float().mean(dim=0)
    mean_concept_acc = concept_acc.mean().item()
    dict_concept_acc = {concept_set[i]: concept_acc[i].item() for i in range(len(concept_set))}
    print(f'\n\tConcept accuracy modifying the side channel: {dict_concept_acc}')
    print(f'\n\tMean concept accuracy modifying the side channel: {mean_concept_acc:.4f}')
    if wb:
        wandb.log({"side_channel/mean_concept_acc": mean_concept_acc})
        for concept, acc in dict_concept_acc.items():
            wandb.log({f"side_channel/{concept}_acc": acc})
    
    # -- Compare soft pseudo-labels -- #
    # Cosine similarity
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    concept_cos_sim = torch.mean(cos_sim(ground_truth_probs, swap_probs))
    print(f'\tCosine similarity between soft pseudo labels after modifying the side channel: {concept_cos_sim.item():.4f}')
    if wb:
        wandb.log({"side_channel/cosine_similarity": torch.mean(concept_cos_sim).item()})
    
    # Total Variation Distance
    tv_dist = total_variation_dist(ground_truth_probs, swap_probs)
    print(f'\tTotal Variation Distance between soft pseudo labels after modifying the side channel: {tv_dist.item():.4f}')
    if wb:
        wandb.log({"side_channel/total_variation_distance": tv_dist.item()})

    # Jensen-Shannon Divergence
    js_divergence = JSD()
    js_div = js_divergence(ground_truth_probs, swap_probs).item()
    print(f'\tJensen-Shannon Divergence between soft pseudo labels after modifying the side channel: {js_div:.4f}')
    if wb:
        wandb.log({"side_channel/jensen_shannon_divergence": js_div})

    return
    

def evaluate_single_concept_intervention(model, clf, concepts_to_intervene, concept_set, device, dataset, direction='i2a', wb=True):

    '''
    Evaluate if we can do test-time interventions and control the generation of the pre-trained model. We evaluate if the target concept is observed 
    in the generated image and if the rest of non-target concepts have been affected by the intervention.
    To do this, we use latent vectors from the pre-trained generative model that are known in advance to produce images where the target concept is 
    either active or inactive.
    * Single concept interventions.
        - Inactive -> Active 
        - Active -> Inactive
    '''

    with torch.no_grad():
    
        metrics_single_intervention = {}

        target_acc = []
        nontarget_acc = []
        cos_sim_nontarget = []
        tv_dist_nontarget = []
        js_div_nontarget = []

        for target_concept in concepts_to_intervene:

            if direction =='i2a':

                # --- Inactive -> Active intervention --- #
                print(f'\n\tIntervening {target_concept} from inactive to active...')

                # Load generated latents with the target concept inactive
                if os.path.exists('eval_latents_rn50_'+dataset+'/'+target_concept+'/inactive.pkl'):

                    try:
                        with open('eval_latents_rn50_'+dataset+'/'+target_concept+'/inactive.pkl', 'rb') as f:
                            dict_inactive = pickle.load(f)
                    except EOFError:
                        print('[EOFError] Cannot open file eval_latents_rn50_'+dataset+'/'+target_concept+'/inactive.pkl')
                        continue
                else:
                    continue
            
            if direction == 'a2i':

                # --- Active -> Inactive intervention --- #
                print(f'\n\tIntervening {target_concept} from active to inactive...')

                if os.path.exists('eval_latents_rn50_'+dataset+'/'+target_concept+'/active.pkl'):
                    # Load generated latents with the target concept active
                    try:
                        with open('eval_latents_rn50_'+dataset+'/'+target_concept+'/active.pkl', 'rb') as f:
                            dict_inactive = pickle.load(f)
                    except EOFError:
                        print('[EOFError] Cannot open file eval_latents_rn50_'+dataset+'/'+target_concept+'/active.pkl')
                        continue
                else:
                    continue

            hard_pseudo_labels_before = []
            hard_pseudo_labels_after = []
            probs_pseudo_labels_before = []
            probs_pseudo_labels_after = []

            for j in range(int(len(dict_inactive))):
                # Select concept we want to intervene (recall that the concept set is the set of concepts used for training)
                idx_concept = concept_set.index(target_concept)

                # Get latent from the dict (we know it produces an image with the concept inactive)
                latent = dict_inactive[str(j)]['latent']
                labels = dict_inactive[str(j)]['labels']
                probs = dict_inactive[str(j)]['probs']

                # Get the indices to map the concepts used for training to the whole concept set
                all_concepts = dict_inactive[str(j)]['concept_names']
                indices = [all_concepts.index(item) for item in concept_set]

                # Select the labels and probabilities for the concepts used for training
                hard_pseudo_labels_before.append(labels[:, indices].cpu())
                probs_pseudo_labels_before.append(probs[:, indices].cpu())

                # Forward encoder
                probs_concepts, out_sc = model.vhcb.encode(latent.to(device))

                if direction == 'i2a':
                    # Activate concept
                    recon_latent_interv = intervene_single_concept(model, probs_concepts, out_sc, idx_concept, target_value=1.)
                if direction == 'a2i':
                    # Deactivate concept
                    recon_latent_interv = intervene_single_concept(model, probs_concepts, out_sc, idx_concept, target_value=0.)

                # Forward the latent through the pre-trained generative model to get the generated image
                gen_imgs_interv = model.gen.synthesis(recon_latent_interv, noise_mode='const')
                gen_imgs_interv = gen_imgs_interv.mul(0.5).add_(0.5)

                # Obtain pseudo-labels for the generated images
                probs_pseudo_labels, pseudo_labels = clf.get_pseudo_labels(gen_imgs_interv, return_prob=True)

                probs_pseudo_labels = torch.stack(probs_pseudo_labels, dim=1)
                pseudo_labels = torch.stack(pseudo_labels, dim=1)

                hard_pseudo_labels_after.append(pseudo_labels.cpu())
                probs_pseudo_labels_after.append(probs_pseudo_labels.cpu())

            hard_pseudo_labels_before = torch.stack(hard_pseudo_labels_before, dim=0).view(-1, len(concept_set))
            probs_pseudo_labels_before = torch.stack(probs_pseudo_labels_before, dim=0).view(-1, len(concept_set))

            hard_pseudo_labels_after = torch.stack(hard_pseudo_labels_after, dim=0).view(-1, len(concept_set))
            probs_pseudo_labels_after = torch.stack(probs_pseudo_labels_after, dim=0).view(-1, len(concept_set))

            # -- Compare hard pseudo-labels -- #
            hard_pseudo_labels_interv = hard_pseudo_labels_before.clone()
            if direction == 'i2a':
                hard_pseudo_labels_interv[:, idx_concept] = 1.0
            if direction == 'a2i':
                hard_pseudo_labels_interv[:, idx_concept] = 0.0
            # Accuracy per concept
            concept_acc = (hard_pseudo_labels_interv == hard_pseudo_labels_after).float().mean(dim=0)
            dict_concept_acc = {concept_set[i]: concept_acc[i].item() for i in range(len(concept_set))}
            # Remove the target concept to check if the non-target concepts changed
            mask = torch.ones(len(concept_acc), dtype=torch.bool)
            mask[idx_concept] = False
            mean_nontar_concept_acc = concept_acc[mask].mean().item()
            mean_tar_concept_acc = concept_acc[idx_concept].mean().item()
            print(f'\tAccuracy target concept {target_concept} {direction} (successful interventions): {mean_tar_concept_acc}')
            print(f'\tMean concept accuracy non-target concepts: {mean_nontar_concept_acc:.4f}')
            pprint(f'\tConcept accuracy after intervention of concept {target_concept} {direction}: {dict_concept_acc}')
            if wb:
                wandb.log({f"single_concept_interv_{direction}/mean_target_concept_acc_{target_concept}": mean_tar_concept_acc})
                wandb.log({f"single_concept_interv_{direction}/mean_nontarget_concept_acc_{target_concept}": mean_nontar_concept_acc})
            
            # -- Compare soft pseudo-labels -- #
            # Remove the target concept to check if the non-target concepts changed
            mask = torch.ones(probs_pseudo_labels_before.size(1), dtype=torch.bool)
            mask[idx_concept] = False
            # Cosine similarity
            cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
            concept_cos_sim = torch.mean(cos_sim(probs_pseudo_labels_before[:,mask], probs_pseudo_labels_after[:,mask]))
            print(f'\tCosine similarity between soft pseudo labels after intervention of concept {target_concept} {direction}: {concept_cos_sim.item():.4f}')
            if wb:
                wandb.log({f"single_concept_interv_{direction}/cosine_similarity_nontarget_{target_concept}": torch.mean(concept_cos_sim).item()})
            
            # Total Variation Distance
            tv_dist = total_variation_dist(probs_pseudo_labels_before[:,mask], probs_pseudo_labels_after[:,mask])
            print(f'\tTotal Variation Distance between soft pseudo labels after intervention of concept {target_concept} {direction}: {tv_dist.item():.4f}')
            if wb:
                wandb.log({f"single_concept_interv_{direction}/total_variation_nontarget_{target_concept}": tv_dist.item()})

            # Jensen-Shannon Divergence
            js_divergence = JSD()
            js_div = js_divergence(probs_pseudo_labels_before[:,mask], probs_pseudo_labels_after[:,mask]).item()
            print(f'\tJensen-Shannon Divergence between soft pseudo labels after intervention of concept {target_concept} {direction}: {js_div:.4f}')
            if wb:
                wandb.log({f"single_concept_interv_{direction}/jensen_shannon_divergence_nontarget_{target_concept}": js_div})

            # Results
            target_acc.append(mean_tar_concept_acc)
            nontarget_acc.append(mean_nontar_concept_acc)
            cos_sim_nontarget.append(concept_cos_sim.item())
            tv_dist_nontarget.append(tv_dist.item())
            js_div_nontarget.append(js_div)


        print('\n\n\tAggregated results:')
        mean_target_acc = np.mean(target_acc)
        mean_nontarget_acc = np.mean(nontarget_acc)
        mean_cos_sim_nontarget = np.mean(cos_sim_nontarget)
        mean_tv_dist_nontarget = np.mean(tv_dist_nontarget)
        mean_js_div_nontarget = np.mean(js_div_nontarget)
        print(f'\tMean target concept accuracy after {direction} interventions: {mean_target_acc:.4f}')
        print(f'\tMean non-target concept accuracy after {direction} interventions: {mean_nontarget_acc:.4f}')
        print(f'\tMean cosine similarity between soft pseudo labels after {direction} interventions: {mean_cos_sim_nontarget:.4f}')
        print(f'\tMean total variation distance between soft pseudo labels after {direction} interventions: {mean_tv_dist_nontarget:.4f}')
        print(f'\tMean Jensen-Shannon Divergence between soft pseudo labels after i{direction} interventions: {mean_js_div_nontarget:.4f}')
        if wb:
            wandb.log({f"single_concept_interv_{direction}/mean_target_acc": mean_target_acc})
            wandb.log({f"single_concept_interv_{direction}/mean_nontarget_acc": mean_nontarget_acc})
            wandb.log({f"single_concept_interv_{direction}/mean_cos_sim_nontarget": mean_cos_sim_nontarget})
            wandb.log({f"single_concept_interv_{direction}/mean_tv_dist_nontarget": mean_tv_dist_nontarget})
            wandb.log({f"single_concept_interv_{direction}/mean_js_div_nontarget": mean_js_div_nontarget})
   
    return


def evaluate_hamming_concept_intervention(model, clf, concept_set, n_steps, batch_size, device, dataset, wb=True, most_probable=100):
    '''
    Evaluate if we can do test-time interventions and control the generation of the pre-trained model. We evaluate if the target concept is observed 
    in the generated image and if the rest of non-target concepts have been affected by the intervention.
    To do this, we use latent vectors from the pre-trained generative model that are known in advance to produce images where the target concept is 
    either active or inactive.
    * Interventions based on minimum Hamming distance.
        Instead of intervening single concepts arbitrarilly, which can yield concept vectors that are very unlikely in the training data distribution 
        (for example, a man with make up or a woman with beard), we can intervene the set of concepts that would yield a concept vector that is in 
        distribution. We can find the target pattern at minimum Hamming to intervene the minimum number of concepts. As we are moving from a probable 
        concept set to another probable concept set, we expect the metrics to improve.
    '''

    # Obtain the ranking of 'patterns' from the dataset
    if dataset == 'celebahq':
        header, patterns = load_labels("./datasets/CelebAMask-HQ/train.txt")
        # Select the concepts used for training the VHCB
        indices = [header.index(item) for item in concept_set]
    if dataset == 'cub':
        header, patterns = load_labels("./datasets/CUB_200_2011/attributes/train_patterns.txt")
        indices = [219, 236, 55, 290, 152, 21, 245, 7, 36, 52]

    patterns = patterns[:, indices]
    # Obtain counts
    unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
    # Get indices that would sort counts in descending order
    sorted_indices = np.argsort(-counts)
    # Reorder both arrays
    sorted_patterns = unique_patterns[sorted_indices]

    with torch.no_grad():

        hard_pseudo_labels_before = []
        hard_pseudo_labels_after = []
        probs_pseudo_labels_before = []
        probs_pseudo_labels_after = []
        mask_intervention = []
        targets = []

        for step in range(n_steps):

            # Sample a noise vector and obtain the latent
            z = torch.randn((batch_size, model.gen.z_dim), device=device)
            latent = model.gen.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)

            # Forward the latent through the pre-trained generative model to get the generated images
            gen_imgs = model.gen.synthesis(latent, noise_mode='const')
            gen_imgs = gen_imgs.mul(0.5).add_(0.5) # Rescale it from -1 to 1 range to 0 to 1

            # Obtain pseudo-labels for the generated images
            probs_pseudo_labels, hard_pseudo_labels = clf.get_pseudo_labels(gen_imgs, return_prob=True)

            hard_pseudo_labels = [t.cpu() for t in hard_pseudo_labels]
            hard_pseudo_labels = torch.stack(hard_pseudo_labels, dim=1)
            probs_pseudo_labels = [t.cpu() for t in probs_pseudo_labels]
            probs_pseudo_labels = torch.stack(probs_pseudo_labels, dim=1)

            hard_pseudo_labels_before.append(hard_pseudo_labels)
            probs_pseudo_labels_before.append(probs_pseudo_labels)

            # Obtain the pattern at minimum Hamming distance for intervention
            target_patterns, _ = find_closest_patterns(hard_pseudo_labels, torch.tensor(sorted_patterns[:most_probable]))
            targets.append(target_patterns)

            # Forward the latent through VHCB encoder to get the predicted concepts
            probs_concepts, out_sc = model.vhcb.encode(latent)
            mask, recon_latent_interv = intervene_multiple_concepts(model, hard_pseudo_labels, probs_concepts, out_sc, target_patterns)

            mask_intervention.append(mask)

            # Forward the latent through the pre-trained generative model to get the generated image
            gen_imgs_interv = model.gen.synthesis(recon_latent_interv, noise_mode='const')
            gen_imgs_interv = gen_imgs_interv.mul(0.5).add_(0.5)

            # Obtain pseudo-labels for the generated images
            probs_pseudo_labels, pseudo_labels = clf.get_pseudo_labels(gen_imgs_interv, return_prob=True)

            probs_pseudo_labels = [t.cpu() for t in probs_pseudo_labels]
            probs_pseudo_labels = torch.stack(probs_pseudo_labels, dim=1)
            pseudo_labels = [t.cpu() for t in pseudo_labels]
            pseudo_labels = torch.stack(pseudo_labels, dim=1)

            hard_pseudo_labels_after.append(pseudo_labels.cpu())
            probs_pseudo_labels_after.append(probs_pseudo_labels.cpu())


        hard_pseudo_labels_before = torch.stack(hard_pseudo_labels_before, dim=0).view(-1, len(concept_set))
        probs_pseudo_labels_before = torch.stack(probs_pseudo_labels_before, dim=0).view(-1, len(concept_set))
        hard_pseudo_labels_after = torch.stack(hard_pseudo_labels_after, dim=0).view(-1, len(concept_set))
        probs_pseudo_labels_after = torch.stack(probs_pseudo_labels_after, dim=0).view(-1, len(concept_set))

        mask_intervention = torch.stack(mask_intervention, dim=0).view(-1, len(concept_set)).cpu()
        targets = torch.stack(targets, dim=0).view(-1, len(concept_set))
        
        # -- Compare hard pseudo-labels -- #
        # Since in this case we are intervening arbitraty concepts for each latent, it does not make sense to do a per-concept analysis in this scenario
        # Accuracy in target concepts
        acc_target_concepts = (targets[mask_intervention] == hard_pseudo_labels_after[mask_intervention]).float().mean()
        # Accuracy in non-target concepts
        acc_nontarget_concepts = (targets[~mask_intervention] == hard_pseudo_labels_after[~mask_intervention]).float().mean()
        print(f'\tMean accuracy target concepts (successful interventions): {acc_target_concepts:.4f}')
        print(f'\tMean accuracy non-target concepts: {acc_nontarget_concepts:.4f}')
        if wb:
            wandb.log({f"hamming_dist_interv/mean_accuracy_target_concepts": acc_target_concepts})
            wandb.log({f"hamming_dist_interv/mean_accuracy_nontarget_concepts": acc_nontarget_concepts})
        
        # -- Compare soft pseudo-labels -- #
        concept_cos_sim_mean = []
        tv_dist_mean = []
        jsd_mean = []

        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        js_divergence = JSD()
        # We do not have the same number of concepts intervened for each sample, so we need to compute the metrics row by row
        for i in range(targets.shape[0]):
            # Cosine similarity
            concept_cos_sim = torch.mean(cos_sim(probs_pseudo_labels_before[i,~mask_intervention[i,:]].unsqueeze(0), probs_pseudo_labels_after[i,~mask_intervention[i,:]].unsqueeze(0)))
            concept_cos_sim_mean.append(concept_cos_sim.item())
            # Total Variation Distance
            tv_dist = total_variation_dist(probs_pseudo_labels_before[i,~mask_intervention[i,:]].unsqueeze(0), probs_pseudo_labels_after[i,~mask_intervention[i,:]].unsqueeze(0))
            tv_dist_mean.append(tv_dist.item())
            # Jensen-Shannon Divergence
            js_div = js_divergence(probs_pseudo_labels_before[i,~mask_intervention[i,:]].unsqueeze(0), probs_pseudo_labels_after[i,~mask_intervention[i,:]].unsqueeze(0))
            jsd_mean.append(js_div.item())

        concept_cos_sim_mean = np.mean(concept_cos_sim_mean)
        print(f'\tMean Cosine Similarity between non-target soft pseudo labels after intervention: {concept_cos_sim_mean:.4f}')
        if wb:
            wandb.log({f"hamming_dist_interv/mean_cosine_similarity_nontarget": concept_cos_sim_mean})

        tv_dist_mean = np.mean(tv_dist_mean)
        print(f'\tMean Total Variation Distance between non-target soft pseudo labels after intervention: {tv_dist_mean:.4f}')
        if wb:
            wandb.log({f"hamming_dist_interv/mean_tv_dist_nontarget": tv_dist_mean})

        jsd_mean = np.mean(jsd_mean)
        print(f'\tMean Jensen Shannon Divergence between non-target soft pseudo labels after intervention: {jsd_mean:.4f}')
        if wb:
            wandb.log({f"hamming_dist_interv/mean_js_div_nontarget": tv_dist_mean})

    return

def evaluation_generation(model, clf, concept_set, n_steps, batch_size, device, dataset, wb=True, most_probable=0):

    '''
    In the VHCB setup we can naturally generate images with specific distributions of concepts. This is different from interventions since, instead of modifying the output of 
    the encoder before forwarding it through the decoder, we can directly sample the side channel from the prior and fix the distribution we want for the concepts and 
    generate a latent accordingly.
    '''

    concepts_sampled = []
    hard_pseudo_labels_img = []
    soft_pseudo_labels_img = []
    hard_concepts_inference = []
    soft_concepts_inference = []
    
    if most_probable > 0:
        # Obtain the ranking of 'patterns' present in the dataset 
        
        if dataset == 'celebahq':
            header, patterns = load_labels("./datasets/CelebAMask-HQ/train.txt")
            # Select the concepts used for training the VHCB
            indices = [header.index(item) for item in concept_set]
        if dataset == 'cub':
            header, patterns = load_labels("./datasets/CUB_200_2011/attributes/train_patterns.txt")
            indices = [219, 236, 55, 290, 152, 21, 245, 7, 36, 52]
        
        patterns = patterns[:, indices]
        # Obtain counts
        unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
        # Get indices that would sort counts in descending order
        sorted_indices = np.argsort(-counts)
        # Reorder both arrays
        sorted_patterns = unique_patterns[sorted_indices]

        # Select the patterns
        most_probable_patterns = sorted_patterns[:most_probable]

    with torch.no_grad():

        # Loop over the number of steps to generate images and infer concepts
        for step in range(n_steps):

            #if(coverage <= 1) and (coverage > 0):
            if most_probable > 0:
                if len(most_probable_patterns) < most_probable:
                    most_probable = len(most_probable_patterns)
                indices = np.random.choice(most_probable, batch_size, replace=True)
                m_sample = torch.FloatTensor(most_probable_patterns[indices,:])
                generated_latent, _ = model.vhcb.generate(n_samples=batch_size, m=m_sample)
                concepts_sampled.append(m_sample.cpu())

            else:
                # Sample random concept vectos and generate the corresponding latents
                generated_latent, m_sample = model.vhcb.generate(n_samples=batch_size)
                concepts_sampled.append(m_sample.cpu())

            # Forward the generated latent through the pre-trained generative model to produce images
            gen_imgs_interv = model.gen.synthesis(generated_latent, noise_mode='const')
            gen_imgs_interv = gen_imgs_interv.mul(0.5).add_(0.5)

            # Obtain pseudo-labels for the generated images
            soft_pseudo_labels, hard_pseudo_labels = clf.get_pseudo_labels(gen_imgs_interv, return_prob=True)
            hard_pseudo_labels = [t.cpu() for t in hard_pseudo_labels]
            hard_pseudo_labels = torch.stack(hard_pseudo_labels, dim=1)
            hard_pseudo_labels_img.append(hard_pseudo_labels.cpu())
            soft_pseudo_labels = [t.cpu() for t in soft_pseudo_labels]
            soft_pseudo_labels = torch.stack(soft_pseudo_labels, dim=1)
            soft_pseudo_labels_img.append(soft_pseudo_labels.cpu())

            # Map the generated latent back to the concept bottleneck
            concept_posterior, _ = model.vhcb.encode(generated_latent)
            labels_inference = (concept_posterior.clone() >= 0.5).int()
            hard_concepts_inference.append(labels_inference.cpu())
            soft_concepts_inference.append(concept_posterior.cpu())

    concepts_sampled = torch.stack(concepts_sampled, dim=0).view(-1, len(concept_set))
    hard_pseudo_labels_img = torch.stack(hard_pseudo_labels_img, dim=0).view(-1, len(concept_set))
    soft_pseudo_labels_img = torch.stack(soft_pseudo_labels_img, dim=0).view(-1, len(concept_set))
    hard_concepts_inference = torch.stack(hard_concepts_inference, dim=0).view(-1, len(concept_set))
    soft_concepts_inference = torch.stack(soft_concepts_inference, dim=0).view(-1, len(concept_set))

    # -- Compare hard pseudo-labels with hard concepts -- #
    # Accuracy in generation
    concept_acc_generation = (concepts_sampled == hard_pseudo_labels_img).float().mean(dim=0)
    mean_concept_acc_generation = concept_acc_generation.mean().item()
    dict_concept_acc_generation = {concept_set[i]: concept_acc_generation[i].item() for i in range(len(concept_set))}
    pprint(f'\n\tConcept accuracy in generation: {dict_concept_acc_generation}')
    print(f'\tMean concept accuracy in generation: {mean_concept_acc_generation:.4f}')
    if wb:
        wandb.log({"concept_generation/mean_concept_acc_generation": mean_concept_acc_generation})
        for concept, acc in dict_concept_acc_generation.items():
            wandb.log({f"concept_generation/{concept}_acc_generation": acc})

    # Accuracy in inference after generation
    concept_acc_inf_gen = (concepts_sampled == hard_concepts_inference).float().mean(dim=0)
    mean_concept_acc_inf_gen = concept_acc_inf_gen.mean().item()
    dict_concept_acc_inf_gen = {concept_set[i]: concept_acc_inf_gen[i].item() for i in range(len(concept_set))}
    pprint(f'\n\tConcept accuracy in inference after generation: {dict_concept_acc_inf_gen}')
    print(f'\tMean concept accuracy in inference after generation: {mean_concept_acc_inf_gen:.4f}')
    if wb:
        wandb.log({"concept_generation/mean_concept_acc_inf_gen": mean_concept_acc_inf_gen})
        for concept, acc in dict_concept_acc_inf_gen.items():
            wandb.log({f"concept_generation/{concept}_acc_inf_gen": acc})

    return
