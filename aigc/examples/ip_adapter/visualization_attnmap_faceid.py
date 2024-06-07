import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import copy

from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus, IPAdapterFaceID
from insightface.app import FaceAnalysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.utils import face_align
from numpy.linalg import norm as l2norm
import cv2
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images


app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root="/root/.insightface")
app.prepare(ctx_id=0, det_size=(640, 640))

v2 = False
# base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
base_model_path = "/data/modelscope_cache/AI-ModelScope/stable-diffusion-v1-5"
vae_model_path = "/data/modelscope_cache/zhuzhukeji/sd-vae-ft-mse"
image_encoder_path = "/data/modelscope_cache/q2792046875/ip-adapter/h94/IP-Adapter/models/image_encoder"
# https://huggingface.co/h94/IP-Adapter-FaceID/tree/main
plus_ip_ckpt = "/data/modelscope_cache/q2792046875/ip-adapter/h94/IP-Adapter/models/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sd15.bin"
ip_ckpt = "/data/modelscope_cache/q2792046875/ip-adapter/h94/IP-Adapter/models/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

pipe.unet = register_cross_attention_hook(pipe.unet)

# generate image
prompt = "photo of a woman in red dress in a garden, white hair, happy"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

import wandb
table = wandb.Table(columns=["prompt", "scale", "face", "gen"])

def rtn_face_get(self, img, face):
    aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
    #print(cv2.imwrite("aimg.png", aimg))
    face.embedding = self.get_feat(aimg).flatten()
    face.crop_face = aimg
    return face.embedding

ArcFaceONNX.get = rtn_face_get
image = cv2.imread("assets/images/woman.png")
faces = app.get(image)
faceid_embeds = faces[0].normed_embedding
faceid_embeds = torch.from_numpy(faceid_embeds).unsqueeze(0)
face_image = faces[0].crop_face

# pip_faceidplus = copy.deepcopy(pipe)
# plus_ip_model = IPAdapterFaceIDPlus(copy.deepcopy(pipe), image_encoder_path, plus_ip_ckpt, device)
plus_ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, plus_ip_ckpt, device)
images = plus_ip_model.generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    face_image=face_image,
    faceid_embeds=faceid_embeds,
    shortcut=v2,
    s_scale=1,
    num_samples=1,
    width=512, height=768,
    num_inference_steps=30, seed=2023
)

# for name, module in pipe.unet.named_modules():
#     if name.split('.')[-1].startswith('attn2'):
#         if hasattr(module.processor, "attn_map"):
#             print(2)

# register_cross_attention_hook(plus_ip_model.pipe.unet)
# print(images[0].size)  # 512, 768, wh
attn_maps = get_net_attn_map((768, 512)) # hw
# print(attn_maps.shape)  # [4, 768, 512]
attn_hot = attnmaps2images(attn_maps)

import matplotlib.pyplot as plt
#axes[0].imshow(attn_hot[0], cmap='gray')
display_images = [cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)] + attn_hot + [images[0]]
fig, axes = plt.subplots(1, len(display_images), figsize=(12, 4))
cmaps = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'jet']
for axe, image in zip(axes, display_images):
    axe.imshow(image, cmap='viridis')
    axe.axis('off')
# plt.show()
plt.savefig("face_attn_1_cp.png")

# pipe_faceid = copy.deepcopy(pipe)
# ip_model = IPAdapterFaceID(copy.deepcopy(pipe), ip_ckpt, device)
ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)
images = ip_model.generate(
    prompt=prompt, negative_prompt=negative_prompt,
    faceid_embeds=faceid_embeds,
    num_samples=1,
    width=512, height=768,
    num_inference_steps=30, seed=2023
)

attn_maps = get_net_attn_map((768, 512))
print(attn_maps.shape)
attn_hot = attnmaps2images(attn_maps)


display_images = [cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)] + attn_hot + [images[0]]
fig, axes = plt.subplots(1, len(display_images), figsize=(12, 4))
for axe, image in zip(axes, display_images):
    axe.imshow(image, cmap='viridis')
    axe.axis('off')
# plt.show()
plt.savefig("face_attn_2_cp.png")

