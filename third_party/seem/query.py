import os
from PIL import Image
import numpy as np
import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.arguments import load_opt_from_config_files
from utils.distributed import init_distributed

def query_with_text(model:BaseModel, feats:torch.tensor, ref_txt:str):
    """
    Query the features with text and get the corresponding scores.
    Args:
        model: The SEEM model.
        feats: Primitive features in [N, d].
        ref_txt: The text to query with.
    Returns:
        Scores in [N,].
    """
    v_emb = feats @ model.model.sem_seg_head.predictor.class_embed
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7) # [N, d]

    t_emb = model.model.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings([ref_txt], name='grounding', token=False, norm=False)['class_emb']
    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7) # [1, d]

    scores = (v_emb @ t_emb.T).squeeze().detach().cpu().numpy() # [N,]
    return scores

def query_with_image(model:BaseModel, feats:torch.tensor, ref_img:Image):
    """
    Query the features with image and get the corresponding scores.
    Args:
        model: The SEEM model.
        feats: Primitive features in [N, d].
        ref_img: The image to query with.
    Returns:
        Scores in [N,].
    """
    v_emb = model.model.sem_seg_head.predictor.mask_embed(feats)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7) # [N, d]

    H = ref_img.size[1]
    W = ref_img.size[0]
    ref_img = np.asarray(ref_img)
    ref_img = torch.from_numpy(ref_img.copy()).permute(2,0,1).to(model.model.device)
    batched_inputs = [{'image': ref_img, 'height': H, 'width': W, 'spatial_query':{}}]
    batched_inputs[0]['spatial_query']['rand_shape'] = torch.ones((1, 1, H, W), dtype=torch.bool, device=model.model.device) # [1, 1, H, W]
    model.model.task_switch['spatial'] = True
    s_emb, _ = model.model.evaluate_referring_image(batched_inputs)
    s_emb = s_emb['visual_query_pos'][0] # [1, d]
    s_emb = s_emb / (s_emb.norm(dim=-1, keepdim=True) + 1e-7) # [1, d]
    model.model.task_switch['spatial'] = False
    
    scores = (v_emb @ s_emb.T).squeeze().detach().cpu().numpy() # [N,]
    return scores

def visualize_scores(image_ori, scores, seg):
    top3_indices = np.argsort(scores)[-3:][::-1]
    top3_scores = scores[top3_indices]
    from utils.visualizer import Visualizer
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori)
    for i in range(3):
        id = top3_indices[i]
        score = top3_scores[i]
        mask = seg == id
        mask = mask.astype(np.float32)
        color = [0.0, 0.0, 0.0]
        color[i] = 1.0
        demo = visual.draw_binary_mask(mask, color=color, text=str(score)[:5], alpha=0.5)
    return Image.fromarray(demo.get_image())

if __name__ == "__main__":
    path_s = "seem/s.npy"
    path_f = "seem/f.npy"
    image_ori = Image.open("demo/seem/examples/river1.png").convert("RGB").resize((640, 480))
    image_ref = Image.open("demo/seem/examples/river2.png").convert("RGB").resize((640, 480))
    seg = np.load(path_s)[0] # [H, W]
    feats = np.load(path_f) # [Q, d] 
    feats = torch.from_numpy(feats).cuda() # [Q, d]
    """
    [Q, d]的feats是从SEEM模型中得到的Q个实例的d维特征
    这里我们只是借助它来验证query方法的正确性
    实际使用中，可以将feats换成N个高斯球的特征（必须是从元根特征pred_primitives中得来的）
    """    

    # Load SEEM model
    opt = load_opt_from_config_files(["configs/seem/focall_unicl_lang_demo.yaml"])
    opt = init_distributed(opt)
    pretrained_pth = os.path.join("checkpoints/seem_focall_v0.pt")
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    # Query with image
    query_with_image(model, feats, image_ref) # [Q, d]
    scores_image = query_with_image(model, feats, image_ref) # [Q,]
    print(scores_image)
    print(seg.shape)
    print(image_ori.size)
    result_image = visualize_scores(image_ori, scores_image, seg)
    result_image.save("sofa_image_query.jpg")

    # Query with text
    scores_text = query_with_text(model, feats, "sofa") # [Q,]
    result_text = visualize_scores(image_ori, scores_text, seg)
    result_text.save("sofa_text_query.jpg")