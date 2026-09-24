import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm  # 进度条
import torch
import torch.nn.functional as F

from seem.modeling.BaseModel import BaseModel
from seem.modeling import build_model
from seem.modeling.language.loss import vl_similarity
from seem.utils.arguments import load_opt_from_config_files
from seem.utils.constants import COCO_PANOPTIC_CLASSES
from seem.utils.distributed import init_distributed
from seem.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import mobileclip

def load_texts(text_folder):
    """加载所有language.txt文件内容"""
    text_files = sorted([f for f in os.listdir(text_folder) if os.path.isdir(os.path.join(text_folder, f))])
    texts = []
    text_names = []
    for folder in text_files:
        txt_path = os.path.join(text_folder, folder, "language.txt")
        text_names.append(folder)
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                texts.append(f.read().strip())
        except:
            print(f"Error reading {txt_path}. Using folder name as text.")
            texts.append(folder[2:])
    return texts, text_names

import os
from PIL import Image

def load_folder_images(text_folder):
    """
    加载所有子文件夹中与子文件夹同名的PNG图片
    :param text_folder: 包含子文件夹的根目录
    :return: (images, image_names) 
             images: 加载的PIL.Image对象列表
             image_names: 对应的子文件夹名称列表
    """
    subfolders = sorted([f for f in os.listdir(text_folder) 
                        if os.path.isdir(os.path.join(text_folder, f))])
    images = []
    image_names = []
    
    for folder in subfolders:
        # 构建图片路径：子文件夹名称 + '.png'
        img_path = os.path.join(text_folder, folder, f"{folder}.png")
        
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            image_names.append(folder)
        except FileNotFoundError:
            print(f"Warning: {img_path} not found. Skipping.")
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}. Skipping.")
    
    return images, image_names

def process_images(image_folder):
    """加载所有图片并预处理"""
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    images = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path).convert('RGB')
        images.append(img)
    return images, image_files

def compute_similarity_table(image_folder, text_folder, cal_score, output_csv="similarity_scores.csv"):
    """
    计算图片和文本的相似度矩阵
    :param image_folder: 图片文件夹路径
    :param text_folder: 文本文件夹路径
    :param cal_score: 计算相似度的函数
    :param output_csv: 输出CSV文件名
    :return: 相似度DataFrame
    """
    # 加载数据
    # texts, text_names = load_texts(text_folder)
    texts, text_names = load_folder_images(text_folder)
    images, img_names = process_images(image_folder)
    
    # 初始化结果矩阵
    scores = np.zeros((len(images), len(texts)))
    
    # 计算所有组合的分数
    for i, img in enumerate(tqdm(images, desc="Processing images")):
        for j, text in enumerate(texts):
            scores[i, j] = cal_score(img, text, model=seem_model)
            
    
    # 创建DataFrame
    df = pd.DataFrame(scores, 
                     index=[os.path.splitext(name)[0] for name in img_names],
                     columns=text_names)
    
    # 保存结果
    df.to_csv(output_csv)
    print(f"Similarity scores saved to {output_csv}")
    return df

def build_seem_model(conf_path, ckpt_path):
    opt = load_opt_from_config_files([conf_path])
    opt = init_distributed(opt)
    model = BaseModel(opt, build_model(opt)).from_pretrained(ckpt_path).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
    return model

@torch.no_grad()
def seem_seg(model, image, reference, text_seg=True):
    image_ori = image
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
    data = {"image": images, "height": height, "width": width}
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False

    if text_seg:
        model.model.task_switch['grounding'] = True
        data['text'] = [reference]
    batch_inputs = [data]
    results,image_size,extra = model.model.evaluate_demo(batch_inputs)
    pred_masks = results['pred_masks'][0] # [Q, H, W]
    v_emb = results['pred_captions'][0] # [Q, d]
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7) # [Q, d]
    t_emb = extra['grounding_class'] # [1, d]
    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7) # [1, d]
    temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    out_prob = torch.matmul(v_emb, t_emb.T) # [Q, 1]
    # matched_id = out_prob.max(0)[1] # which instance
    matched_score, matched_id = out_prob.max(0) # 相似度分数

    pred_masks_pos = pred_masks[matched_id,:,:] # [1, H, W]， 实例的mask
    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
    # for idx, mask in enumerate(pred_masks_pos):
    #     # color = random_color(rgb=True, maximum=1).astype(np.int32).tolist()
    #     color = [1.0, 0.0, 0.0]
    #     text = str(matched_score.item())[:5]
    #     visual = Visualizer(image_ori, metadata=MetadataCatalog.get('coco_2017_train_panoptic'))
    #     demo = visual.draw_binary_mask(mask, color=color, text=text, alpha=0.5)
    return matched_score.item(), pred_masks_pos[0]


# ------------------------------------------------------------------
# Inference methods to interact with vectorized simulation
# environments
# ------------------------------------------------------------------

# 利用clip模型计算相似度分数
@torch.no_grad()
@torch.cuda.amp.autocast()
def mobileclip_score(clip_model, clip_tokenizer, clip_preprocess, image, text, score_type = 'text'):
    
    if score_type == 'text':
        text = clip_tokenizer([text])
        image = clip_preprocess(Image.fromarray(image.astype(np.uint8), mode='RGB')).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        probs = (image_features @ text_features.T)#.softmax(dim=-1)
    elif score_type == 'image':
        image1 = clip_preprocess(image).unsqueeze(0)
        image2 = clip_preprocess(text).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image1_features = clip_model.encode_image(image1)
            image2_features = clip_model.encode_image(image2)
            image1_features /= image1_features.norm(dim=-1, keepdim=True)
            image2_features /= image2_features.norm(dim=-1, keepdim=True)

        probs = (image1_features @ image2_features.T)#.softmax(dim=-1)
    
    return probs[0,0].item()



seem_cfg = {
    'conf_path': '/home/wxl/lagmemo/Segment-Everything-Everywhere-All-At-Once/configs/seem/focall_unicl_lang_demo.yaml',
    'ckpt_path': '/home/wxl/lagmemo/Segment-Everything-Everywhere-All-At-Once/checkpoints/seem_focall_v0.pt',
}

clip_cfg = '/home/wxl/lagmemo/ml-mobileclip/checkpoints/mobileclip_s0.pt'

clip_model, _, clip_preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=clip_cfg)
clip_tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

seem_model = build_seem_model(**seem_cfg)

# 示例cal_score函数（需要根据实际需求实现）
def cal_score(image, text, model= None):
    """示例相似度计算函数"""
    # 这里应该是你的实际相似度计算逻辑
    # 返回一个0-1之间的分数
    
    # seem, text2img
    # score,_ = seem_seg(model, image, text)
    
    # clip, img2img
    score = mobileclip_score(clip_model, clip_tokenizer, clip_preprocess, image, text, score_type='image')

    return score

# 使用示例
if __name__ == "__main__":
    similarity_df = compute_similarity_table(
        image_folder='/home/wxl/lagmemo/lagmemo/result/0/rgb',
        text_folder="/home/wxl/lagmemo/gs_data/groundtruth_data",
        cal_score=cal_score
    )