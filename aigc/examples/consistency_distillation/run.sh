export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch train_lcm_distill_lora_sdxl.py \
  --pretrained_teacher_model=${MODEL_NAME}  \
  --pretrained_vae_model_name_or_path=${VAE_PATH} \
  --output_dir="pokemons-lora-lcm-sdxl" \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 \
  --train_batch_size=24 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --lora_rank=64 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --checkpointing_steps=500 \
  --validation_steps=50 \
  --seed="0" \
  --report_to="wandb" 