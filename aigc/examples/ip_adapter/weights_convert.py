import torch
ckpt = "checkpoint-50000/pytorch_model.bin"
sd = torch.load(ckpt, map_location="cpu")
image_proj_sd = {}
ip_sd = {}
for k in sd:
    if k.startswith("unet"):
        pass
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]

torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "ip_adapter.bin")