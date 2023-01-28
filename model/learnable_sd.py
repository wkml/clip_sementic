import torch
import torch.nn as nn
import numpy as np
from utils.checkpoint import load_clip_to_cpu
from .semantic import semantic
from clip import clip
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DenseCLIP(nn.Module):
    def __init__(self, args, classnames,
                image_feature_dim=2048, num_classes=80, 
                word_feature_dim=512):
        super(DenseCLIP, self).__init__()
        args.use_attn = False
        self.clip_model = load_clip_to_cpu(args).float()
        self.num_classes = num_classes
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.dtype = self.clip_model.dtype
        self.text_features = self.get_text_features(classnames)
        self.word_semantic = semantic(num_classes= self.num_classes,
                                      image_feature_dim = self.image_feature_dim,
                                      word_feature_dim=self.word_feature_dim)

        self.logit_scale = self.clip_model.logit_scale
        self.fc = nn.Linear(self.image_feature_dim, word_feature_dim)

    def forward(self, image):
        image_features = self.clip_model.encode_image(image.type(self.dtype))       #[bs, 2048, H, W]
        text_features = self.text_features
        # SD
        sd_features, coefficient = self.word_semantic(image_features, text_features)    # [bs, 80, 512]
        sd_features = torch.relu(self.fc(sd_features))
        
        sd_features = sd_features / sd_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * sd_features @ text_features.t()          # [bs, 80, 80]

        output = torch.diagonal(logits, dim1=-2, dim2=-1)
        
        return output, coefficient

    def get_text_features(self, classnames):
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        text_features = self.clip_model.encode_text(prompts)
        return nn.Parameter(text_features, requires_grad=False)
