from ip_adapter.attention_processor import AttnProcessor, IPAttnProcessor
from diffusers import UNet2DConditionModel

ip_model_dir = "/data/modelscope_cache/q2792046875/ip-adapter/h94/IP-Adapter/"
sd_model_dir = "/data/modelscope_cache/AI-ModelScope/stable-diffusion-v1-5/"
sdxl_model_dir = "/data/modelscope_cache/AI-ModelScope/stable-diffusion-xl-base-1.0"
img_encoder_dir = ip_model_dir + "models/image_encoder"

unet = UNet2DConditionModel.from_pretrained(sd_model_dir, subfolder="unet")

attn_procs = {}
unet_sd = unet.state_dict()
for name in unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]
    if cross_attention_dim is None:
        attn_procs[name] = AttnProcessor()
    else:
        layer_name = name.split(".processor")[0]
        weights = {
            "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
            "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
        }
        attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        attn_procs[name].load_state_dict(weights)
unet.set_attn_processor(attn_procs)


# with open('network_structure/unetxl_ip.txt', 'w') as f:
#     print(unet, file=f)