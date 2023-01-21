import torch
import torch.nn as nn
from model.prompt import PromptLearner, TextEncoder
from utils.checkpoint import load_clip_to_cpu
import torch.nn.functional as F
from clip import clip


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DenseCLIP(nn.Module):
    def __init__(self, args, classnames):
        super().__init__()
        args.use_attn = True
        clip_model = load_clip_to_cpu(args).float()
        self.prompt_learner = PromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.model = clip_model
        self.dtype = clip_model.dtype
        self.logit_scale = clip_model.logit_scale.exp()

    def forward(self, image):
        image_features = self.model.encode_image(image.type(self.dtype))                                    # [bs, 196, 512]
        image_features = image_features[:, 0, :]                                                            # [bs, 512]

        prompts, temperature, spatial_T = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)                            # [bs, 80, 512]

        # 求相似度
        logits = self.logit_scale * image_features @ text_features.t()
        return logits
    
    def get_text_features(self, classnames):
        temp = "In the scene there is a {}"
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        text_features = self.model.encode_text(prompts)
        return nn.Parameter(text_features, requires_grad=False)