import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib
from visualize import visualize_grid_attention_v2

from model.new_sd import DenseCLIP
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cocoapi/PythonAPI'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def getCategoryList(item):
    categories = set()
    for t in item:
        categories.add(t['category_id'])
    return list(categories)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_data_transform = transforms.Compose([transforms.Resize((512, 512)),
                                            transforms.CenterCrop(448),
                                            transforms.ToTensor(),
                                            normalize])
Reshape = transforms.Resize((448, 448))


def main(): 
    with open("data/coco/category_name.json", 'r') as f:
        name = np.array(eval(f.read()))
    args = argparse.ArgumentParser()
    args.backbone_name = 'RN101'
    args.crop_size = 448
    args.n_ctx = 16
    args.ctx_init = 'In the scene there is a'
    args.csc = False
    args.class_token_position = 'end'

    model = DenseCLIP(args=args, classnames=name)
    checkpoint = torch.load("exp/checkpoint/new_SD-exp3.2-cat_global_local_prompt/Checkpoint_Best.pth", map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    root = '/data/public/coco2014/val2014'
    test_list = '/data/public/coco2014/annotations/instances_val2014.json'
    coco = COCO(test_list)
    ids = list(coco.imgs.keys())
    for i in range(len(ids)):
        img_id = ids[i]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids) 
        categories = getCategoryList(target)
        categories = [i - 1 for i in categories]
        path = coco.loadImgs(img_id)[0]['file_name']
        input = Image.open(os.path.join(root, path)).convert('RGB')
        input = test_data_transform(input).unsqueeze(0)
        output, attention_map = model(input)
        print(categories)
        print(output)
        # print(attention_map[0][0].reshape(14, 14).detach().numpy())
        for i, attn in enumerate(attention_map[0][categories]):
            visualize_grid_attention_v2(os.path.join(root, path),
                           save_path="test_{}".format(name[i]),
                           attention_mask = attn.reshape(14, 14).detach().numpy(),
                           save_image=True,
                           save_original_image=True,
                           quality=100)
        break

if __name__ == '__main__':
    main()