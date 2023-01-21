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
        self.local_head = nn.Linear(512, 1024)
        self.local_head2 = nn.Linear(1024, 512)
        self.logit_scale = clip_model.logit_scale.exp()

    def forward(self, image):
        image_features = self.model.encode_image(image.type(self.dtype))                                    # [bs, 196, 512]
        global_image_features = image_features[:, 0, :]                                                     # [bs, 512]
        local_image_features = image_features[:, 1:, :]                                                     # [bs, 512, 196]
        local_image_features = self.local_head2(F.relu(self.local_head(local_image_features)))
        # print(global_image_features.shape, local_image_features.shape)

        # image_features = torch.cat([global_image_features.unsqueeze(1), local_image_features], dim=1)        # [bs, 197, 512]
        prompts, temperature, spatial_T = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        # text_features = self.text_features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        # text_features_global = self.text_encoder(prompts_global, tokenized_prompts)
        
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        global_image_features = global_image_features / global_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features_sd = text_features_sd / text_features.norm(dim=-1, keepdim=True)

        coefficient = text_features @ local_image_features.permute(0, 2, 1)                                   # [bs, 80, 196]
        # 对类别取，一个块应该只属于一个类
        coefficient = F.softmax(spatial_T.exp() * coefficient, dim=-1)                                                   # [bs, 80, 196]
        sd_features = coefficient @ local_image_features                                                      # [bs, 80, 512]
        
        sd_features = sd_features / sd_features.norm(dim=-1, keepdim=True)
        # text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)

        # 求相似度
        logits_local = (temperature.exp() * sd_features * text_features).sum(dim=-1)
        logits_global = self.logit_scale * global_image_features @ text_features.t()
        return (logits_local + logits_global) / 2
    
    def get_text_features(self, classnames):
        temp = "In the scene there is a {}"
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        text_features = self.model.encode_text(prompts)
        return nn.Parameter(text_features, requires_grad=False)