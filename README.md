# Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling

> **Note**
> This repository is the official implementation for our AAAI 2026 paper:
> **Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling**
>
> [Official paper link will be added upon publication]

This repository provides an implementation of Stable Diffusion v2.1 focused on accelerated inference. It uses an automated script to discover optimal sampling schedules, enabling faster high-quality image generation from text prompts.

-----

## Getting Started 🚀

To begin, you'll need to install the required packages and download the model checkpoints. Using a virtual environment is strongly recommended.

### Installation

1.  **Clone the Repository**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Set Up a Virtual Environment**

    ```bash
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install PyTorch**
    The specific PyTorch version you need depends on your hardware (CPU or GPU with a specific CUDA version). Therefore, it must be installed separately.

    Search for the **official PyTorch website** to find the correct installation command for your operating system, package manager, and compute platform. For example, a common command for systems with CUDA support is:

    ```bash
    pip3 install torch torchvision torchaudio
    ```

4.  **Install Dependencies**
    Once PyTorch is installed, install the remaining packages using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

### Model Checkpoints

This implementation is designed for Stable Diffusion v2.1. You must download the model checkpoints from the Hugging Face Hub by searching for the official `stabilityai/stable-diffusion-2-1` and `stabilityai/stable-diffusion-2-1-base` repositories.

After downloading, place the `.ckpt` files into a `checkpoints` directory at the root of this project, or note their location for the configuration step.

#### Advanced Note: Using Other Models

This codebase is specifically tuned for the noise schedule of `stabilityai/stable-diffusion-2-1`.

Our optimization method (HSO) is highly dependent on the model's specific noise progression (i.e., the mapping from timestep $t$ to $\alpha_t$ and $\sigma_t$). If you wish to adapt this repository for a different model (e.g., SD 1.5, SDXL, PixArt-$\alpha$), you **must** implement that model's unique noise schedule in the code. Simply loading different weights will not work and will produce poor results.

-----

## How to Use

Image generation is handled by the **`test_unipc_optim.py`** script. All configuration is done by editing the script directly.

### 1. Configure the Script

Open `test_unipc_optim.py` in your editor and modify the configuration variables at the top of the file:

```python
# --- Configuration ---
# Folder containing your .txt prompt files
prompts_folder = "prompts/"

# Folder where generated images will be saved
output_folder = "outputs/"

# Path to the model's configuration file
config_path = "configs/stable-diffusion/v2-inference.yaml"

# Path to your downloaded Stable Diffusion checkpoint
ckpt_path = "checkpoints/v2-1_512-ema-pruned.ckpt"

# Number of inference steps (NFE)
n_steps = 4
````

### 2\. Prepare Your Prompts

Create a folder (e.g., `prompts/`) and place each text prompt in a separate `.txt` file inside it. The script will process every `.txt` file found in the directory specified by `prompts_folder`.

### 3\. Run Inference

Execute the script from your terminal. It will generate an optimized timestep schedule (or load a cached one) and create an image for each prompt.

```bash
python test_unipc_optim.py
```

Generated images will be saved to the directory specified by `output_folder`.

-----

## Acceleration Explained

The core of this project's acceleration is the **`stages_step_optim.py`** script. It implements a meta-learning framework to automatically find an optimized sampling schedule. This process significantly reduces the number of steps needed to generate a quality image.

The `test_unipc_optim.py` script, which you run for inference, automatically uses this optimization engine to generate or load the ideal schedule for the UniPC sampler.

-----

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{zhu2026hso,
  title     = {Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling},
  author    = {Zhu, Aihua and Su, Rui and Zhao, Qinglin and Feng, Li and Shen, Meng and He, Shibo},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

-----

## License

This project is licensed under the MIT License. The Stable Diffusion model weights are licensed under the **CreativeML Open RAIL++-M License**.

## Acknowledgements

This work is built upon the foundational research and models from:

  * **CompVis** and **RunwayML** for the original Stable Diffusion.
  * **Stability AI** for training and releasing the v2.1 models.
  * The authors of the **UniPC sampler** for their contributions to efficient sampling.

<!-- end list -->

```
```