# --- Required Libraries ---
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import os
import json 


from ldm.models.diffusion.uni_pc.sampler import UniPCSampler 
from ldm.util import instantiate_from_config

# Import the core optimization engine from your other script.
from ldm.models.diffusion.stages_step_optim import StepOptim, NoiseScheduleVP

# =======================================================================
# --- Configuration Block ---
# All user-configurable settings are grouped here for easy access.
# =======================================================================
# 1. Paths
# Folder where your input .txt prompt files are located.
prompts_folder = "prompts/"
# Folder where the generated images will be saved.
output_folder = "outputs/" 
# Path to the model's structural configuration file (e.g., v2-inference.yaml).
config_path = "configs/stable-diffusion/v2-inference.yaml"
# Path to your downloaded Stable Diffusion .ckpt model weights file.
ckpt_path = "./models/v2-1_512-ema-pruned.ckpt"

# 2. Generation Parameters
# The number of inference steps (NFE) to use for sampling.
n_steps = 4
# Classifier-Free Guidance scale. Controls how strongly the prompt influences the output.
guidance_scale = 7.5
# Desired output image dimensions.
height = 512
width = 512
# Number of images to generate per prompt.
batch_size = 1

# 3. Schedule Cache
# Path to the file where the optimized timestep schedule is saved.
# This acts as a cache to avoid re-calculating the schedule on every run.
# NOTE: You should change this to a suitable path on your system.
optimized_schedule_file = 'path/to/your/optimized schedule'
# =======================================================================




# Ensure the folder for saving images exists.
os.makedirs(output_folder, exist_ok=True)
print(f"Prompts will be read from: {prompts_folder}") 
print(f"Generated images will be saved to: {output_folder}")

# --- 1. Load the Stable Diffusion Model ---
print("🔄 Loading model...")
# Load the model's architecture from the YAML configuration file.
config = OmegaConf.load(config_path)
# Build the model structure and then load the trained weights from the checkpoint file.
model = instantiate_from_config(config.model)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)

# This is a specific fix for certain models trained in float16 to prevent dtype errors during inference.
if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model') and \
   hasattr(model.model.diffusion_model, 'dtype') and model.model.diffusion_model.dtype == torch.float16:
    print("Overriding UNet's internal dtype from torch.float16 to torch.float32.")
    model.model.diffusion_model.dtype = torch.float32

# Ensure the model is in float32 and move it to the appropriate compute device.
model.float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval() # Set to evaluation mode.
print("✅ Model loaded successfully!")

# --- 2. Check for or Generate the Optimal Schedule ---
custom_timesteps = None
# Check if a pre-calculated schedule file exists at the specified path.
if os.path.exists(optimized_schedule_file):
    print(f"🔄 Loading optimized timesteps from: {optimized_schedule_file}")
    try:
        # Load the timesteps from the text file.
        loaded_timesteps_np = np.loadtxt(optimized_schedule_file, dtype=np.float32)
        custom_timesteps = torch.from_numpy(loaded_timesteps_np).to(device) 
        
        # IMPORTANT: Validate that the loaded schedule matches the desired number of steps.
        # The schedule should have N+1 points for N steps.
        if len(custom_timesteps) != n_steps + 1:
            print(f"⚠️ Loaded timesteps length {len(custom_timesteps)} does not match n_steps+1 ({n_steps+1}). Regenerating...")
            custom_timesteps = None # Invalidate if it doesn't match.
        else:
            print(f"✅ Loaded {len(custom_timesteps)} timesteps.")
    except Exception as e:
        print(f"❌ Error loading timesteps from {optimized_schedule_file}: {e}. Will regenerate.")
        custom_timesteps = None

# If no valid schedule was loaded from the cache, we must generate one.
if custom_timesteps is None:
    print(f"🔄 Optimized schedule not found or invalid. Generating new schedule for {n_steps} steps...")
    # The optimizer needs the model's original noise schedule (alphas_cumprod) to work.
    if not hasattr(model, 'alphas_cumprod'):
        raise ValueError("The main diffusion model does not have 'alphas_cumprod'.")
    
    # Initialize the NoiseScheduleVP class from the optimization engine.
    alphas_for_ns = model.alphas_cumprod.clone().detach().cpu().to(torch.float32)
    ns_instance = NoiseScheduleVP('discrete', alphas_cumprod=alphas_for_ns, dtype=torch.float32)
    
    # Initialize the optimizer itself.
    step_optimizer = StepOptim(ns=ns_instance)
    # Define the lower bound for time `t` for the optimization.
    eps_t_0_for_optim = (1. / ns_instance.total_N) if ns_instance.schedule_name == 'discrete' else 1e-3
    
    # Call the optimization function to calculate the best timesteps.
    optimized_t_steps_tensor, _ = step_optimizer.get_ts_lambdas(
       n_steps,
       eps=eps_t_0_for_optim,
       initType='edm',
    )
    custom_timesteps = optimized_t_steps_tensor.to(device) 
    
    # Save the newly generated schedule to the cache file for future use.
    try:
        np.savetxt(optimized_schedule_file, custom_timesteps.cpu().numpy(), fmt='%.8f')
        print(f"✅ Optimized timesteps saved to {optimized_schedule_file}")
    except Exception as e:
        print(f"❌ Error saving timesteps to {optimized_schedule_file}: {e}")

# --- 3. Set Up Sampler and Run Inference Loop ---
# Initialize the UniPC sampler with the loaded model.
sampler = UniPCSampler(model)

# Get the unconditional conditioning (empty prompt), used for classifier-free guidance.
uc = model.get_learned_conditioning(batch_size * [""])
print("🔄 Processing prompts...")
# Find all .txt files in the specified prompts directory.
prompt_files = [f for f in os.listdir(prompts_folder) if f.endswith(".txt")]

if not prompt_files:
    print(f"⚠️ No .txt files found in the '{prompts_folder}' directory.")
else:
    # Loop through each prompt file found.
    for prompt_filename in prompt_files:
        prompt_filepath = os.path.join(prompts_folder, prompt_filename)
        try:
            # Read the prompt text from the file.
            with open(prompt_filepath, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        except Exception as e:
            print(f"❌ Error reading {prompt_filename}: {e}")
            continue
        if not prompt: print(f"⚠️ Empty prompt in {prompt_filename}. Skipping."); continue

        print(f"\n✨ Generating image for prompt: '{prompt}'")
        # Get the conditional conditioning from the text prompt.
        c = model.get_learned_conditioning([prompt])
        c = c.to(device)
        if uc is not None: uc = uc.to(device)

        # Define the shape of the initial random noise tensor in the latent space.
        shape_latent = [4, height // 8, width // 8]

        # This is the main sampling call.
        samples, _ = sampler.sample(
            S=n_steps,                      # Number of steps.
            conditioning=c,                 # The prompt embedding.
            batch_size=batch_size,
            shape=shape_latent,
            verbose=True,                   # Print progress.
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc,  # The empty prompt embedding.
            custom_timesteps=custom_timesteps, # CRITICAL: Use our optimized schedule here.
            uni_pc_order=3,                 # Order of the UniPC algorithm.
            optimize_steps_on_demand=False  # We provide the steps, so no on-demand optimization needed.
        )

        # Decode the generated latents from the latent space back into a visible image.
        x_samples = model.decode_first_stage(samples)
        # Clamp and normalize the image tensor to a displayable range [0, 1].
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
        # Convert the tensor to a NumPy array suitable for saving as an image file.
        x_samples_np = x_samples.cpu().permute(0, 2, 3, 1).numpy()
        
        # Save each image in the batch (usually just one).
        for i, img_np_loop in enumerate(x_samples_np):
            img_pil = Image.fromarray((img_np_loop * 255).astype(np.uint8))
            # Create a descriptive filename for the output image.
            base_name = os.path.splitext(prompt_filename)[0]
            output_image_filename = f"{base_name}_gs{guidance_scale}_steps{n_steps}_unipc_optim_ondemand.png"
            output_image_path = os.path.join(output_folder, output_image_filename)
            img_pil.save(output_image_path)
            print(f"🖼️ Image saved to: {output_image_path}")
            
    print("\n✅ All prompts processed!")