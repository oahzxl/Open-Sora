# support 3 types of ckpt
# for pretrained model: --ckpt-path OpenSora-v1-HQ-16x512x512.pth
# for main model: --ckpt-path /.../outputs/XXX/epochX-global_stepX/model
# for ema model: --ckpt-path /.../outputs/XXX/epochX-global_stepX/ema.pth

deepspeed --include=localhost:0 scripts/inference.py configs/opensora/inference/16x512x512.py --ckpt-path OpenSora-v1-HQ-16x512x512.pth --prompt-path ./assets/texts/t2v_samples.txt
