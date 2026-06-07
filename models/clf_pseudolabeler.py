"Extracted from https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from utils.datasets import CelebAHQ_dataset, CUB_dataset_multiconc

from torchvision import datasets, models, transforms
from torchvision import models

def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in range(math.ceil(len(text)/batch_size)):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features


class CustomNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            # t.sub_(m).div_(s)
            t = t - m
            t = torch.div(t, s)
        return tensor


class CLIP_PseudoLabeler(nn.Module):
    def __init__(self, set_of_classes, device='cuda:0', clip_model_name='ViT-B/16', clip_model_type='clip'):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.clip_model_type = clip_model_type
        if self.clip_model_type == 'clip':
            self.clip_model, self.clip_transform = clip.load(self.clip_model_name, device)
            self.text_features = []
            self.set_of_classes = set_of_classes
            for cls_list in self.set_of_classes:
                self.text = clip.tokenize(["{}".format(word) for word in cls_list]).to(device)
                self.text_features.append(get_clip_text_features(self.clip_model, self.text))
                self.text_features[-1] /= self.text_features[-1].norm(dim=-1, keepdim=True)

        self.clip_mean = (0.48145466, 0.4578275, 0.40821073)
        self.clip_std = (0.26862954, 0.26130258, 0.27577711)

        self.clip_norm = CustomNormalize(self.clip_mean, self.clip_std)

    # return_prob=True returns top-1 probabilities instead of margin (which is top-1st minus top-2nd)
    def get_pseudo_labels(self, image, return_prob=False):
        tf_input = F.interpolate(image, size=224, mode='bicubic', align_corners=False)
        tf_input = self.clip_norm(tf_input)
        pred_cls_list = []
        pred_prob_list = []
        with torch.no_grad():
            image_features = self.clip_model.encode_image(tf_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for cls_set_idx in range(len(self.set_of_classes)):
                if self.clip_model_type == 'clip':
                    similarity = (100.0 * image_features @ self.text_features[cls_set_idx].T).softmax(dim=-1)
                elif self.clip_model_type == 'siglip':
                    similarity = torch.sigmoid(image_features @ self.text_features[cls_set_idx].T * self.clip_model.logit_scale.exp() + self.clip_model.logit_bias)
                if return_prob:
                    pred_prob, pred_cls = similarity.topk(1)
                    prob_label_1 = torch.where(pred_cls == 0, 1 - pred_prob, pred_prob)
                    pred_cls = pred_cls.squeeze(1)
                    margin = prob_label_1[:,0]
                else:
                    pred_prob, _ = similarity.topk(2)
                    _, pred_cls = similarity.topk(1)
                    pred_cls = pred_cls.squeeze(1)
                    margin = pred_prob[:, 0] - pred_prob[:, 1]
                pred_cls_list.append(pred_cls)
                pred_prob_list.append(margin)
        return pred_prob_list, pred_cls_list

    def get_soft_pseudo_labels(self, image):
        tf_input = F.interpolate(image, size=224, mode='bicubic', align_corners=False)
        tf_input = self.clip_norm(tf_input)
        pred_logits_list = []
        image_features = self.clip_model.encode_image(tf_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        for cls_set_idx in range(len(self.set_of_classes)):
            # not taking softmax to return the (kind of) logits
            similarity = (100.0 * image_features @ self.text_features[cls_set_idx].T)
            pred_logits_list.append(similarity)
        return pred_logits_list


class Sup_PseudoLabeler(nn.Module):
    def __init__(self, set_of_classes, device='cuda:0', dataset='celebahq', model_type='rn18'):
        super().__init__()

        self.set_of_classes = set_of_classes
        self.dataset = dataset
        self.model_type = model_type

        self.models = nn.ModuleList()
        for idx, cls_list in enumerate(self.set_of_classes):
            if model_type == 'rn18':
                curr_model = models.resnet18(weights='DEFAULT')
                num_features = curr_model.fc.in_features
                curr_model.fc = nn.Linear(num_features, 2) # binary classification (num_of_class == 2)
            elif model_type == 'rn50':
                curr_model = models.resnet50(weights='DEFAULT')
                num_features = curr_model.fc.in_features
                curr_model.fc = nn.Linear(num_features, 2) # binary classification (num_of_class == 2)
            elif model_type == 'vit_l_16':
                curr_model = models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1')
                num_features = curr_model.heads.head.in_features
                curr_model.heads = nn.Linear(num_features, len(cls_list))

            if len(cls_list) == 2:
                conc_save_name = cls_list[-1].replace(' ', '_')
            else:
                conc_save_name = cls_list.replace(' ', '_')

            curr_model.load_state_dict(torch.load(f'models/clf_checkpoints/{self.dataset}_{conc_save_name}_{self.model_type}_conclsf.pth'))
            print(f'Loaded model for {conc_save_name} concept from models/clf_checkpoints/{self.dataset}_{conc_save_name}_{self.model_type}_conclsf.pth')

            if device is not None:
                curr_model = curr_model.to(device)

            for param in curr_model.parameters():
                param.requires_grad = False

            self.models.append(curr_model)

        # we will resize manually with F interpolate since it's already in tensor form
        self.sup_mean = (0.485, 0.456, 0.406)
        self.sup_std = (0.229, 0.224, 0.225)

        self.sup_norm = CustomNormalize(self.sup_mean, self.sup_std)

    # return_prob=True returns top-1 probabilities instead of margin (which is top-1st minus top-2nd)
    def get_pseudo_labels(self, image, return_prob=False):
        if self.model_type == 'vit_l_16':
            tf_input = F.interpolate(image, size=512, mode='bicubic', align_corners=False)
        elif self.dataset == 'celebahq' or self.dataset == 'cub':
            tf_input = F.interpolate(image, size=256, mode='bicubic', align_corners=False)
        elif self.dataset == 'celeba64' or self.dataset == 'cub64':
            tf_input = F.interpolate(image, size=64, mode='bicubic', align_corners=False)
        
        tf_input = self.sup_norm(tf_input)
        pred_cls_list = []
        pred_prob_list = []
        with torch.no_grad():#, torch.cuda.amp.autocast():
            for cls_set_idx in range(len(self.set_of_classes)):
                probs = self.models[cls_set_idx](tf_input).softmax(dim=-1)
                if return_prob:
                    pred_prob, pred_cls = probs.topk(1)
                    pred_cls = pred_cls.squeeze(1)
                    prob_cls_1 = probs[:, 1]
                    margin = prob_cls_1
                else:
                    pred_prob, _ = probs.topk(2)
                    _, pred_cls = probs.topk(1)
                    pred_cls = pred_cls.squeeze(1)
                    margin = pred_prob[:, 0] - pred_prob[:, 1]
                pred_cls_list.append(pred_cls)
                pred_prob_list.append(margin)
        return pred_prob_list, pred_cls_list

    def get_soft_pseudo_labels(self, image):
        tf_input = F.interpolate(image, size=256, mode='bicubic', align_corners=False)
        tf_input = self.sup_norm(tf_input)
        pred_logits_list = []
        for cls_set_idx in range(len(self.set_of_classes)):
            # not taking softmax to return the (kind of) logits
            logits = self.models[cls_set_idx](tf_input)
            pred_logits_list.append(logits)
        return pred_logits_list

