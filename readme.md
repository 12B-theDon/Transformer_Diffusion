# Transformer Diffusion

Simple workspace for Transformer + Diffusion study code.

## Structure

- `chapter03/`: Autoencoders, variational autoencoders (VAEs), and CLIP.
- `chapter04/`: Diffusion fundamentals, noise schedules, and denoising models.
- `chapter05/`: Class-conditioned Fashion MNIST generation, Stable Diffusion, classifier-free guidance, and Gradio demos.

## Model Weights

Locally saved checkpoints from the study notebooks (not currently tracked in Git):

- `chapter03/checkpoints/ae_latent2.pth`: MNIST autoencoder weights with a two-dimensional latent space for reconstruction and latent-space exploration.
- `chapter03/checkpoints/vae.pth`: MNIST VAE weights with a two-dimensional latent space for reconstruction and sampling.
- `chapter05/fashion_ddpm/model/diffusion_pytorch_model.safetensors`: Class-conditioned DDPM U-Net weights for generating 32 × 32 grayscale Fashion MNIST images across 10 classes. The adjacent `config.json` stores the model architecture; `fashion_ddpm/scheduler/scheduler_config.json` stores the noise schedule.

The chapter03 checkpoints include model weights and channel/latent dimensions, and can be loaded with `chapter03/checkpoint_loaders.py`. The Fashion DDPM model and scheduler can be restored with Diffusers' `from_pretrained()` methods. Stable Diffusion v1.5 weights are downloaded from Hugging Face when running the chapter05 notebook.
