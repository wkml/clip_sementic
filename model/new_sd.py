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
        self.text_features = self.get_text_features(classnames)
    
        # self.v_linear_weight = self.model.visual.attnpool.v_proj.weight
        # self.v_linear_bias = self.model.visual.attnpool.v_proj.bias
        # self.c_linear_weight = self.model.visual.attnpool.c_proj.weight
        # self.c_linear_bias = self.model.visual.attnpool.c_proj.bias


    def forward(self, image):
        # image_features = self.model.encode_image(image.type(self.dtype))                                   # [b, c, h, w]
        # b, c, h, w = image_features.shape
        # x = image_features.reshape(b, c, h * w).permute(0, 2, 1)
        # x = F.linear(x, self.v_linear_weight, self.v_linear_bias)
        # image_features = F.linear(x, self.c_linear_weight, self.c_linear_bias)
        image_features, _ = self.model.encode_image(image.type(self.dtype))                              # [bs, 196, 512]

        prompts_sd, prompts_global, temperature, spatial_T = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features_sd = self.text_encoder(prompts_sd, tokenized_prompts)
        text_features = self.text_features
        # text_features_global = self.text_encoder(prompts_global, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_sd = text_features_sd / text_features_sd.norm(dim=-1, keepdim=True)
        

        coefficient = text_features_sd @ image_features.permute(0, 2, 1)                                   # [bs, 80, 196]
        scale = spatial_T.exp()
        coefficient = F.softmax(scale * coefficient, dim=-1)                                               # [bs, 80, 196]
        sd_features = coefficient @ image_features                                                         # [bs, 80, 512]
        
        sd_features = sd_features / sd_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)

        # 求相似度
        logit_scale = temperature.exp()
        logits = (logit_scale * sd_features * text_features).sum(dim=-1)
        return logits, coefficient
    
    def get_text_features(self, classnames):
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        text_features = self.model.encode_text(prompts)
        return nn.Parameter(text_features, requires_grad=False)