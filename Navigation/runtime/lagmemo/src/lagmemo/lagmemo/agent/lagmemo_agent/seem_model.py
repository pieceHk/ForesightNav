import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from seem.modeling.BaseModel import BaseModel
from seem.modeling import build_model
from seem.modeling.language.loss import vl_similarity
from seem.utils.arguments import load_opt_from_config_files
from seem.utils.constants import COCO_PANOPTIC_CLASSES
from seem.utils.distributed import init_distributed
from seem.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# # 设置随机种子
# seed = 0
# torch.manual_seed(seed)
# np.random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
# # os.environ['PYTHONHASHSEED'] = str(seed)
# import random
# random.seed(seed)

# 画半径为5的圆mask
def create_center_point_mask(image: Image.Image, radius=5):
    """
    Create a binary mask with a small filled circle at the center of the image.
    """
    width, height = image.size
    mask = np.zeros((height, width), dtype=np.uint8)
    cx, cy = width // 2, height // 2
    for y in range(-radius, radius+1):
        for x in range(-radius, radius+1):
            if 0 <= cy + y < height and 0 <= cx + x < width:
                if x**2 + y**2 <= radius**2:
                    mask[cy + y, cx + x] = 1
    return mask

# 从npy文件加载mask
def load_mask_from_npy(mask_path: str):
    """
    Load a mask from a .npy file.
    """
    mask = np.load(mask_path).astype("uint8") # (1,353,353)
    if mask.ndim == 3:
        mask = mask[0]   
    return mask

class SEEMModel:
    def __init__(self, conf_path, ckpt_path):
        seem_cfg = {
            'conf_path': conf_path,
            'ckpt_path': ckpt_path,
        }
        
        self.seem_model = self.build_seem_model(**seem_cfg)
        # Resize transform: keep aspect ratio, min edge = 512
        self.resize_transform = transforms.Resize(512, interpolation=Image.BICUBIC)
            
    def build_seem_model(self, conf_path, ckpt_path):
        opt = load_opt_from_config_files([conf_path])
        opt = init_distributed(opt)
        model = BaseModel(opt, build_model(opt)).from_pretrained(ckpt_path).eval().cuda()
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
        return model

    @torch.no_grad()
    def img2txt_seem_seg(self, image, reference, text_seg=True):
        image_ori = image
        width = image_ori.shape[1]
        height = image_ori.shape[0]
        # image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
        data = {"image": images, "height": height, "width": width}
        self.seem_model.model.task_switch['spatial'] = False
        self.seem_model.model.task_switch['visual'] = False
        self.seem_model.model.task_switch['grounding'] = False
        self.seem_model.model.task_switch['audio'] = False

        if text_seg:
            self.seem_model.model.task_switch['grounding'] = True
            data['text'] = [reference]
        batch_inputs = [data]
        results,image_size,extra = self.seem_model.model.evaluate_demo(batch_inputs)
        pred_masks = results['pred_masks'][0] # [Q, H, W]
        v_emb = results['pred_captions'][0] # [Q, d]
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7) # [Q, d]
        t_emb = extra['grounding_class'] # [1, d]
        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7) # [1, d]
        temperature = self.seem_model.model.sem_seg_head.predictor.lang_encoder.logit_scale
        out_prob = torch.matmul(v_emb, t_emb.T) * temperature # [Q, 1]
        # matched_id = out_prob.max(0)[1] # which instance
        matched_score, matched_id = out_prob.max(0) # 相似度分数

        pred_masks_pos = pred_masks[matched_id,:,:] # [1, H, W]， 实例的mask
        pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
        goal_mask = cv2.resize(pred_masks_pos[0], (image_ori.shape[1], image_ori.shape[0]), interpolation=cv2.INTER_LINEAR)
        for idx, mask in enumerate(pred_masks_pos):
            # color = random_color(rgb=True, maximum=1).astype(np.int32).tolist()
            color = [1.0, 0.0, 0.0]
            text = str(matched_score.item())[:5]
            visual = Visualizer(image_ori, metadata=MetadataCatalog.get('coco_2017_train_panoptic'))
            demo = visual.draw_binary_mask(mask, color=color, text=text, alpha=0.5)
        demo = cv2.cvtColor(demo.get_image(), cv2.COLOR_BGR2RGB)
        demo = cv2.resize(demo, (image_ori.shape[1], image_ori.shape[0]), interpolation=cv2.INTER_LINEAR)
        return matched_score.item(), goal_mask, np.array(demo)

    # @torch.no_grad()
    # def img2img_seem_seg(self, image, text_seg=True):
    #     '''
    #     return:
    #         - pred_masks: [Q, H, W]， Q个实例的mask
    #     '''
        
    #     image_ori = image
    #     reference = 'goal'
    #     width = image_ori.shape[1]
    #     height = image_ori.shape[0]
    #     # image_ori = np.asarray(image_ori)
    #     images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
    #     data = {"image": images, "height": height, "width": width}
    #     self.seem_model.model.task_switch['spatial'] = False
    #     self.seem_model.model.task_switch['visual'] = False
    #     self.seem_model.model.task_switch['grounding'] = False
    #     self.seem_model.model.task_switch['audio'] = False

    #     if text_seg:
    #         self.seem_model.model.task_switch['grounding'] = True
    #         data['text'] = [reference]
    #     batch_inputs = [data]
    #     results,image_size,extra = self.seem_model.model.evaluate_demo(batch_inputs)
    #     pred_masks = results['pred_masks'][0] # [Q, H, W]
    #     goal_masks = np.zeros((pred_masks.shape[0], image_ori.shape[0], image_ori.shape[1]))
    #     goal_imgs = []
    #     goal_masks = []

    #     for matched_id in range(pred_masks.shape[0]):
    #         matched_id_t = torch.tensor([matched_id]).cuda()
    #         pred_masks_pos = pred_masks[matched_id_t,:,:] # [1, H, W]， 实例的mask
    #         pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
    #         cur_mask = cv2.resize(pred_masks_pos[0], (image_ori.shape[1], image_ori.shape[0]), interpolation=cv2.INTER_LINEAR)
    #         # 如果这个mask80%以上的区域是1，则不保存
    #         if np.sum(cur_mask) / (cur_mask.shape[0] * cur_mask.shape[1]) > 1:
    #             continue
    #         goal_masks.append(cur_mask)
    #         # goal_masks[matched_id] = cur_mask
    #         demo = (image_ori*goal_masks[-1][:,:,np.newaxis]).astype(np.uint8)
    #         demo = cv2.cvtColor(demo, cv2.COLOR_BGR2RGB)
    #         goal_imgs.append(demo)
    #         # cv2.imwrite(f"result/output_{matched_id}.png", demo)
    #     goal_masks = np.array(goal_masks) # [Q, H, W]
    #     return goal_masks, goal_imgs
    
    def get_seem_semantic_image(self, image, mask, score):
        color = [1.0, 0.0, 0.0]
        text = str(score)[:5]
        visual = Visualizer(image, metadata=MetadataCatalog.get('coco_2017_train_panoptic'))
        demo = visual.draw_binary_mask(mask, color=color, text=text, alpha=0.5)
        demo = cv2.cvtColor(demo.get_image(), cv2.COLOR_BGR2RGB)
        demo = cv2.resize(demo, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        return demo
    
    # metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    def preprocess_image(self, image: Image.Image):
        image_resized = self.resize_transform(image)
        np_image_hw3 = np.asarray(image_resized)
        torch_image = torch.from_numpy(np_image_hw3.copy()).permute(2, 0, 1).cuda()  # [3, H, W]
        return torch_image, np_image_hw3

    @torch.no_grad()
    def img2img_seem_seg(self, query_image: np.array, ref_image: np.array, goal_mask = None):
        query_image = Image.fromarray(query_image).convert("RGB")
        ref_image = Image.fromarray(ref_image).convert("RGB")
        query_image_ori = np.asarray(query_image) # (405, 544, 3)
        height_ori = query_image_ori.shape[0]
        width_ori = query_image_ori.shape[1]
        query_tensor, query_img_np = self.preprocess_image(query_image)  # (405, 544) -> (512, 687)
        ref_tensor, _ = self.preprocess_image(ref_image) # (353,353) -> (512,512)
        height_query = query_tensor.shape[1]
        width_query = query_tensor.shape[2]
        height_ref = ref_tensor.shape[1]
        width_ref = ref_tensor.shape[2]

        self.seem_model.model.task_switch['spatial'] = False
        self.seem_model.model.task_switch['visual'] = False
        self.seem_model.model.task_switch['grounding'] = False
        self.seem_model.model.task_switch['audio'] = False

        # 准备参考图像和mask（中心点）
        if goal_mask is None:
            ref_mask = create_center_point_mask(ref_image)  # (353,353)
        else:
            ref_mask = goal_mask
        # 或者加载已有mask（用seem_seg生成）
        # ref_mask = load_mask_from_npy("/home/zht/github_play/Segment-Everything-Everywhere-All-At-Once/exp_demo/refer_mask/mask.npy")  # (353,353)
        ref_mask_tensor = torch.from_numpy(np.array(ref_mask)[:, :, None]).permute(2, 0, 1)[None,]
        ref_mask_tensor = (F.interpolate(ref_mask_tensor, size=(height_ref, width_ref), mode='bilinear') > 0)

        # 推理参考图像，提取视觉embedding
        self.seem_model.model.task_switch['spatial'] = True
        self.seem_model.model.task_switch['visual'] = True
        batched_inputs_ref = [{
            'image': ref_tensor,
            'height': height_ref,
            'width': width_ref,
            'spatial_query': {
                'rand_shape': ref_mask_tensor
            }
        }]
        with torch.no_grad():
            ref_output, _ = self.seem_model.model.evaluate_referring_image(batched_inputs_ref)

        # 关闭ref模式
        self.seem_model.model.task_switch['spatial'] = False

        # 设置query数据，注入reference视觉向量
        query_data = {
            'image': query_tensor,
            'height': height_query,
            'width': width_query,
            'visual': ref_output
        }
        batch_inputs = [query_data]
        results, image_size, _ = self.seem_model.model.evaluate_demo(batch_inputs)

        # 获取 mask 匹配结果
        v_emb = results['pred_maskembs']
        s_emb = results['pred_pvisuals']
        pred_masks = results['pred_masks']

        pred_logits = v_emb @ s_emb.transpose(1, 2)
        logit_max, logits_idx_y = pred_logits[:, :, 0].max(dim=1)
        softmax_score = torch.softmax(pred_logits[:, :, 0], dim=1)
        softmax_score_max = softmax_score.max(dim=1)[0]
        # print(f"logits: {pred_logits[:, :, 0]}")
        # print(f"max_id: {logits_idx_y.item()}")
        # print(f"Logits max score: {logit_max.item()}")
        # print(f"Softmax max score: {softmax_score_max.item()}")
        logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)
        logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
        pred_masks_pos = pred_masks[logits_idx]
        pred_class = results['pred_logits'][logits_idx].max(dim=-1)[1]

        # Resize mask 回原图尺寸 image_size[-2:] (512,704) -> (512,687)
        pred_masks_pos = (F.interpolate(pred_masks_pos[None], image_size[-2:], mode='bilinear')[0, :, :height_query, :width_query] > 0.0).float().cpu().numpy()
        # 再压缩到原图尺寸 (512, 687) -> (405, 544)
        pred_masks_pos = F.interpolate(torch.from_numpy(pred_masks_pos).unsqueeze(0), size=(height_ori, width_ori), mode='bilinear')[0, :, :, :].numpy()
        # print(f"mask shape back to: {pred_masks_pos.shape}")

        # save mask to npy
        # np.save("mask.npy", pred_masks_pos)
        # 可视化 text为保留2位小数的logit_max
        visual = Visualizer(query_image_ori, metadata=MetadataCatalog.get('coco_2017_train_panoptic'))
        for idx, mask in enumerate(pred_masks_pos):
            color = [0.0, 1.0, 0.0]
            visual = visual.draw_binary_mask(mask, color=color, text=str(logit_max.item())[:4], alpha=0.5)
        demo = cv2.cvtColor(visual.get_image(), cv2.COLOR_BGR2RGB)
        demo = cv2.resize(demo, (query_image_ori.shape[1], query_image_ori.shape[0]), interpolation=cv2.INTER_LINEAR)
        return softmax_score_max.item(), pred_masks_pos[0], demo



# seem_model = SEEMModel(
#     conf_path='/home/wxl/lagmemo/Segment-Everything-Everywhere-All-At-Once/configs/seem/focall_unicl_lang_demo.yaml',
#     ckpt_path='/home/wxl/lagmemo/Segment-Everything-Everywhere-All-At-Once/checkpoints/seem_focall_v0.pt'
# )
# image_input = Image.open(f"/home/wxl/lagmemo/lagmemo/output.png").convert("RGB")
# img_ref = Image.open(f"/home/wxl/lagmemo/lagmemo/output1.png").convert("RGB")
# image_input = np.asarray(image_input)
# img_ref = np.asarray(img_ref)

# reference = 'display cabinet'
# _, goal_mask,image_output = seem_model.img2txt_seem_seg(img_ref, reference)
# cv2.imwrite("output_txt.png", image_output)

# _, mask, img = seem_model.img2img_seem_seg(image_input, img_ref, goal_mask)

# cv2.imwrite("output.png", img)

# seem_model = SEEMModel(
#     conf_path='/home/wxl/lagmemo/Segment-Everything-Everywhere-All-At-Once/configs/seem/focall_unicl_lang_demo.yaml',
#     ckpt_path='/home/wxl/lagmemo/Segment-Everything-Everywhere-All-At-Once/checkpoints/seem_focall_v0.pt'
# )
# image_input = Image.open(f"/home/wxl/lagmemo/Segment-Everything-Everywhere-All-At-Once/demo/seem/examples/river1.png").convert("RGB")
# img_ref = Image.open(f"/home/wxl/lagmemo/Segment-Everything-Everywhere-All-At-Once/demo/seem/examples/river2.png").convert("RGB")
# image_input = np.asarray(image_input)
# reference = 'river'
# # _,_,image_output = seem_model.seem_seg(image_input, reference)
# # cv2.imwrite("output.png", image_output)

# masks, imgs = seem_model.img2img_seem_seg(image_input, False)

# import torch
# from PIL import Image
# import mobileclip
# import os
# import pandas as pd
# from itertools import combinations

# # 初始化模型
# model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='/home/wxl/lagmemo/ml-mobileclip/checkpoints/mobileclip_s0.pt')
# tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

# def encode_txt(text):
#     """编码图片和文本为特征向量"""
#     text = tokenizer([text])
    
#     with torch.no_grad(), torch.cuda.amp.autocast():
#         text_features = model.encode_text(text)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
    
#     return text_features


# def encode_img(imgs):
#     """编码图片和文本为特征向量"""
#     features = []
    
#     for img in imgs:
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             image_features = model.encode_image(img)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#         features.append(image_features)

#     return features

# def calculate_similarity(feat1, feat2):
#     """计算两个特征向量的相似度"""
#     feat_matrix = torch.stack(feat1, dim=0)  # [N, d]
#     feat2 = feat2.unsqueeze(0) if feat2.dim() == 1 else feat2  # [1, d]
    
#     # 计算相似度 (矩阵乘法 + 广播)
#     similarity_scores = feat_matrix @ feat2.T  # [N, 1]
#     print(f"Similarity scores: {similarity_scores.squeeze(-1)}")
    
#     return similarity_scores.squeeze(-1).argmax(), similarity_scores.max().item()  # 返回最大相似度的索引

# text = "river"
# images = [preprocess(Image.fromarray(img)).unsqueeze(0) for img in imgs]
# f1 = encode_img(images)
# f2 = encode_img([preprocess(img_ref).unsqueeze(0)])

# best_id, score =  calculate_similarity(f1, f2[0])
# print(f"Best ID: {best_id}, Score: {score}")
# best_img = cv2.cvtColor((image_input*masks[best_id][:,:,np.newaxis]).astype(np.uint8), cv2.COLOR_BGR2RGB)
# cv2.imwrite("best_img.png", best_img)

# color = [1.0, 0.0, 0.0]
# text = str(score)[:5]
# visual = Visualizer(image_input, metadata=MetadataCatalog.get('coco_2017_train_panoptic'))
# demo = visual.draw_binary_mask(masks[best_id], color=color, text=text, alpha=0.5)
# demo = cv2.cvtColor(demo.get_image(), cv2.COLOR_BGR2RGB)
# demo = cv2.resize(demo, (image_input.shape[1], image_input.shape[0]), interpolation=cv2.INTER_LINEAR)

# cv2.imwrite("demo.png", demo)