import os
import torch
from torch import nn
import pickle
from src.vhcb_layer import VHCB_layer

class VHCB_StyGAN2(nn.Module):

    def __init__(self, config: dict):
        super(VHCB_StyGAN2, self).__init__()
        
        self.config = config
        self.noise_dim =  config["model"]["latent_noise_dim"]
        self.num_channels =  config["dataset"]["num_channels"]
        self.has_concepts= config["model"]["has_concepts"]
        self.model_type = config["model"]["type"]
        self.num_ws = config["model"].get("num_ws", None)

        if(self.has_concepts):
            self.concept_name =config["model"]["concepts"]["concept_names"]
            self.input_latent_dim  = config["model"]["input_latent_dim"]
            self.n_concepts =len(self.concept_name)
            self._build_model()

    def _build_model(self):
        pretrained_model_path = self.config['model']['pretrained']
        print(f'loading stylegan2 from {pretrained_model_path}')
        with open(pretrained_model_path, 'rb') as f:
            self.gen = pickle.load(f)['G_ema']
        
        assert self.num_ws is not None

        # --- Read Configuration File --- #
        # Model
        beta = self.config['model']['beta']
        concept_inf = self.config['model']['concept_inf']
        sc_type = self.config['model']['sc_type']
        sc_inf = self.config['model']['sc_inf']
        sc_dim = self.config['model']['sc_dim']
        # Codes
        concept_code_root = self.config['model']['concept_code']['root']
        concept_code_file = self.config['model']['concept_code']['file']
        concept_bits_info = self.config['model']['concept_code']['bits_info']
        concept_bits_code = self.config['model']['concept_code']['bits_code']
        sc_code_root = self.config['model']['sc_code']['root']
        sc_code_file = self.config['model']['sc_code']['file']
        sc_bits_info = self.config['model']['sc_code']['bits_info']
        sc_bits_code = self.config['model']['sc_code']['bits_code']

        # --- Repetition Codes --- #
        # Concepts
        G_concept=None
        if concept_inf == 'rep':

            assert self.n_concepts == concept_bits_info, "The number of 'concept' information bits must be equal to the number of concepts."

            # Load matrices
            if concept_code_file == 'default':
                concept_code_path = os.path.join(concept_code_root, 'rep_matrices_'+str(self.n_concepts)+'_'+str(concept_bits_code)+'.pkl')
            else:
                concept_code_path = os.path.join(concept_code_root, concept_code_file)
            
            with open(concept_code_path, 'rb') as file:
                rep_matrices = pickle.load(file)

            G_concept = rep_matrices['G']

        # Side Channel
        G_sc=None
        if sc_type=='binary' and sc_inf == 'rep':
            # Load matrices
            if sc_code_file == 'default':
                sc_code_path = os.path.join(sc_code_root, 'rep_matrices_'+str(sc_bits_info)+'_'+str(sc_bits_code)+'.pkl')
            else:
                sc_code_path = os.path.join(sc_code_root, sc_code_file)
            
            with open(sc_code_path, 'rb') as file:
                rep_matrices = pickle.load(file)

            G_sc = rep_matrices['G']

        self.vhcb = VHCB_layer(self.noise_dim, self.n_concepts, concept_inf=concept_inf, sc_type=sc_type, sc_inf=sc_inf, sc_dim=sc_dim, G_concept=G_concept, G_sc=G_sc, beta=beta, num_ws=self.num_ws)




