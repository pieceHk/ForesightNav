# from PIL import Image
# import numpy as np
# import torch
# import torch.nn.functional as F

# from seem.modeling.BaseModel import BaseModel
# from seem.modeling import build_model
# from seem.modeling.language.loss import vl_similarity
# from seem.utils.arguments import load_opt_from_config_files
# from seem.utils.constants import COCO_PANOPTIC_CLASSES
# from seem.utils.distributed import init_distributed
# from seem.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog

# def build_seem_model(conf_path, ckpt_path):
#     opt = load_opt_from_config_files([conf_path])
#     opt = init_distributed(opt)
#     model = BaseModel(opt, build_model(opt)).from_pretrained(ckpt_path).eval().cuda()
#     with torch.no_grad():
#         model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
#     return model

# def seem_seg(model, image, reference, text_seg=True):
#     image_ori = image
#     width = image_ori.size[0]
#     height = image_ori.size[1]
#     image_ori = np.asarray(image_ori)
#     images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
#     data = {"image": images, "height": height, "width": width}
#     model.model.task_switch['spatial'] = False
#     model.model.task_switch['visual'] = False
#     model.model.task_switch['grounding'] = False
#     model.model.task_switch['audio'] = False

#     if text_seg:
#         model.model.task_switch['grounding'] = True
#         data['text'] = [reference]
#     batch_inputs = [data]
#     results,image_size,extra = model.model.evaluate_demo(batch_inputs)
#     pred_masks = results['pred_masks'][0] # [Q, H, W]
#     v_emb = results['pred_captions'][0] # [Q, d]
#     v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7) # [Q, d]
#     t_emb = extra['grounding_class'] # [1, d]
#     t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7) # [1, d]
#     temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
#     out_prob = torch.matmul(v_emb, t_emb.T) # [Q, 1]
#     # matched_id = out_prob.max(0)[1] # which instance
#     matched_score, matched_id = out_prob.max(0) # 相似度分数
#     print("score:", matched_score.item())
#     # import ipdb; ipdb.set_trace()

#     pred_masks_pos = pred_masks[matched_id,:,:] # [1, H, W]， 实例的mask
#     pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
#     for idx, mask in enumerate(pred_masks_pos):
#         # color = random_color(rgb=True, maximum=1).astype(np.int32).tolist()
#         color = [1.0, 0.0, 0.0]
#         text = str(matched_score.item())[:5]
#         visual = Visualizer(image_ori, metadata=MetadataCatalog.get('coco_2017_train_panoptic'))
#         demo = visual.draw_binary_mask(mask, color=color, text=text, alpha=0.5)
#         # mask.astype(np.uint8) * 255
#         # mask为numpy array，值为0或1，保存为mask.npy文件
#         np.save(f"demo/seem/examples/mask_{idx}.npy", mask)
#         # import ipdb; ipdb.set_trace()

#     return Image.fromarray(demo.get_image())

# if __name__ == "__main__":
#     model = build_seem_model("configs/seem/focall_unicl_lang_demo.yaml",
#                      "checkpoints/seem_focall_v0.pt")
#     image_input = Image.open(f"/home/wxl/lagmemo/Segment-Everything-Everywhere-All-At-Once/cabinet.png").convert("RGB")
#     reference = 'display cabinet' #"flat screen tv that is located to the left of the cabinet"
#     image_output = seem_seg(model, image_input, reference)
#     image_output.save("output.png")


# segmentation for image with reference image
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
from utils.distributed import init_distributed
from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def build_seem_model(conf_path, ckpt_path):
    opt = load_opt_from_config_files([conf_path])
    opt = init_distributed(opt)
    model = BaseModel(opt, build_model(opt)).from_pretrained(ckpt_path).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
    return model

# Resize transform: keep aspect ratio, min edge = 512
resize_transform = transforms.Resize(512, interpolation=Image.BICUBIC)
# metadata = MetadataCatalog.get('coco_2017_train_panoptic')
def preprocess_image(image: Image.Image):
    image_resized = resize_transform(image)
    np_image_hw3 = np.asarray(image_resized)
    torch_image = torch.from_numpy(np_image_hw3.copy()).permute(2, 0, 1).cuda()  # [3, H, W]
    return torch_image, np_image_hw3

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
def seem_seg(model, query_image: Image.Image, ref_image: Image.Image):
    query_image_ori = np.asarray(query_image) # (405, 544, 3)
    print(f"query image shape: {query_image_ori.shape}")
    height_ori = query_image_ori.shape[0]
    width_ori = query_image_ori.shape[1]
    query_tensor, query_img_np = preprocess_image(query_image)  # (405, 544) -> (512, 687)
    ref_tensor, _ = preprocess_image(ref_image) # (353,353) -> (512,512)
    height_query = query_tensor.shape[1]
    width_query = query_tensor.shape[2]
    height_ref = ref_tensor.shape[1]
    width_ref = ref_tensor.shape[2]

    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False

    # 准备参考图像和mask（中心点）
    ref_mask = create_center_point_mask(ref_image)  # (353,353)
    
    # 或者加载已有mask（用seem_seg生成）
    # ref_mask = load_mask_from_npy("/home/zht/github_play/Segment-Everything-Everywhere-All-At-Once/exp_demo/refer_mask/mask.npy")  # (353,353)
    ref_mask_tensor = torch.from_numpy(np.array(ref_mask)[:, :, None]).permute(2, 0, 1)[None,]
    ref_mask_tensor = (F.interpolate(ref_mask_tensor, size=(height_ref, width_ref), mode='bilinear') > 0)

    # 推理参考图像，提取视觉embedding
    model.model.task_switch['spatial'] = True
    model.model.task_switch['visual'] = True
    batched_inputs_ref = [{
        'image': ref_tensor,
        'height': height_ref,
        'width': width_ref,
        'spatial_query': {
            'rand_shape': ref_mask_tensor
        }
    }]
    with torch.no_grad():
        ref_output, _ = model.model.evaluate_referring_image(batched_inputs_ref)

    # 关闭ref模式
    model.model.task_switch['spatial'] = False

    # 设置query数据，注入reference视觉向量
    query_data = {
        'image': query_tensor,
        'height': height_query,
        'width': width_query,
        'visual': ref_output
    }
    batch_inputs = [query_data]
    results, image_size, _ = model.model.evaluate_demo(batch_inputs)

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
    np.save("mask.npy", pred_masks_pos)
    # 可视化 text为保留2位小数的logit_max
    visual = Visualizer(query_image_ori, metadata=MetadataCatalog.get('coco_2017_train_panoptic'))
    for idx, mask in enumerate(pred_masks_pos):
        color = [0.0, 1.0, 0.0]
        visual = visual.draw_binary_mask(mask, color=color, text=str(logit_max.item())[:4], alpha=0.5)
    return Image.fromarray(visual.get_image())

if __name__ == "__main__":
    model = build_seem_model("configs/seem/focall_unicl_lang_demo.yaml", "checkpoints/seem_focall_v0.pt")

    # 输入图像路径
    image_input = Image.open("/home/wxl/lagmemo/Segment-Everything-Everywhere-All-At-Once/cabinet.png").convert("RGB")
    reference_image = Image.open("/home/wxl/lagmemo/Segment-Everything-Everywhere-All-At-Once/cabinet_goal.png").convert("RGB")
    output = seem_seg(model, image_input, reference_image)
    output.save("output.png")