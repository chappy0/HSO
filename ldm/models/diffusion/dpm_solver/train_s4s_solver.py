# S4S_training.py



import torch
import torch.nn as nn
import numpy as np
import os
import logging
from tqdm import tqdm
import random
import lpips # 导入LPIPS库

# 导入您现有的模块
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
# 我们需要一个高质量的教师求解器，例如DPM-Solver++
# from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler
from ldm.util import instantiate_from_config


# S4S_training.py

# ===================================================================
# # --- 终极诊断与修复方案 ---
# import logging
# import sys
# import os

# # --- 诊断部分 ---
# # 打印出Python解释器认为的“当前工作目录”
# # 这能告诉我们脚本是从哪个文件夹启动的
# print("="*50)
# print(f"🐍 [DIAGNOSTIC] Current Working Directory: {os.getcwd()}")

# # 打印出Python解释器当前的完整模块搜索路径
# print("🐍 [DIAGNOSTIC] Current sys.path:")
# for i, path in enumerate(sys.path):
#     print(f"   {i}: {path}")
# print("="*50)

# # --- 硬编码修复部分 ---
# # 请将下面的路径替换为您项目的绝对根目录路径
# # 在Windows上，请使用双反斜杠'\\'或正斜杠'/'
# # 例如: "d:/paper/stablediffusion-main" 或 "d:\\paper\\stablediffusion-main"
# PROJECT_ROOT_PATH = r"d:\paper\stablediffusion-main" # <--- !!! 请确保这个路径是正确的 !!!

# # 检查路径是否存在，如果不存在则报错
# if not os.path.isdir(PROJECT_ROOT_PATH):
#     raise FileNotFoundError(f"Error: The hardcoded project root path does not exist: {PROJECT_ROOT_PATH}")
# if not os.path.isdir(os.path.join(PROJECT_ROOT_PATH, 'ldm')):
#      raise FileNotFoundError(f"Error: The 'ldm' folder was not found inside the project root: {os.path.join(PROJECT_ROOT_PATH, 'ldm')}")

# # 将这个硬编码的路径强制插入到搜索列表的最前面
# if PROJECT_ROOT_PATH not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT_PATH)
#     print(f"✅ [FIX] Hardcoded project root to Python path: {PROJECT_ROOT_PATH}")

# # ===================================================================

# # 现在可以安全地从ldm导入了
# try:
#     from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler
#     from ldm.util import instantiate_from_config
#     print("✅ [SUCCESS] Successfully imported 'ldm' module.")
# except ModuleNotFoundError as e:
#     print(f"❌ [FAILURE] Still failed to import 'ldm' module after hardcoding path.")
#     print("   Please double-check that the PROJECT_ROOT_PATH is set correctly and that the 'ldm' folder exists directly inside it.")
#     raise e

# ... 您脚本中其余所有的import和代码 ...
# import torch
# import torch.nn as nn
# ...
# --- 日志记录设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- 配置区 ---
# 模型和调度路径 (请根据您的实际情况修改)
CONFIG_PATH = r"D:\paper\stablediffusion-main\configs\stable-diffusion\v2-inference.yaml"
CKPT_PATH = r"D:\paper\FCDiffusion_code-main\models\v2-1_512-ema-pruned.ckpt"
SCHEDULE_PATH = r"D:\paper\stablediffusion-main\pre_compute\schedule.txt" # 假设这是我们为10步优化的调度文件

# 训练超参数
N_TEACHER_STEPS = 100        # 教师求解器的步数
STUDENT_ORDER = 3            # 学生求解器的阶数 (使用历史步数)
TRAINING_STEPS = 300         # 训练学生求解器的总迭代次数
LEARNING_RATE = 1e-3         # 学习率
BATCH_SIZE = 1               # 每次只用一张图的轨迹进行训练
GUIDANCE_SCALE = 7.5         # 生成教师轨迹时使用的CFG
LPIPS_WEIGHT = 1.0           # 感知损失的权重
L2_WEIGHT = 0.1              # L2损失的权重 (可选，用于稳定训练)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def generate_teacher_trajectory(model, sampler, n_teacher_steps, student_schedule_t, prompt, guidance_scale):
    """
    使用高NFE的教师求解器，生成在学生调度时间点上的“正确”latent状态。
    """
    logging.info(f"Generating teacher trajectory with {n_teacher_steps} steps...")
    
    # 教师求解器使用自己的高密度调度生成一个高质量的完整轨迹
    teacher_samples, teacher_intermediates = sampler.sample(
        S=n_teacher_steps,
        conditioning=model.get_learned_conditioning(BATCH_SIZE * [prompt]).to(DEVICE),
        batch_size=BATCH_SIZE,
        shape=[4, 64, 64], # For 512x512 image
        verbose=False,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=model.get_learned_conditioning(BATCH_SIZE * [""]).to(DEVICE),
        return_intermediates=True,
    )
    
    # teacher_intermediates['x_inter'] 包含了 n_teacher_steps + 1 个点的轨迹
    teacher_full_trajectory = teacher_intermediates['x_inter'].squeeze(1) # 移除多余的batch维度
    teacher_full_timesteps = sampler.get_timesteps(n_teacher_steps)

    # 为了从教师轨迹中精确采样，我们需要进行插值
    from scipy.interpolate import interp1d
    
    student_t_cpu = student_schedule_t.cpu().numpy()
    teacher_t_cpu = teacher_full_timesteps.cpu().numpy()
    
    # 将轨迹变形以进行插值
    C, H, W = teacher_full_trajectory.shape[1:]
    teacher_flat = teacher_full_trajectory.view(-1, C * H * W).cpu().numpy()

    # 创建插值函数 (确保t是递减的，scipy要求x递增)
    sort_idx = np.argsort(teacher_t_cpu)[::-1]
    interpolator = interp1d(teacher_t_cpu[sort_idx], teacher_flat[sort_idx], axis=0, kind='linear', fill_value="extrapolate")
    
    # 在学生的时间点上进行插值
    student_true_latents_flat = interpolator(student_t_cpu)
    
    student_true_latents = torch.from_numpy(student_true_latents_flat).view(-1, C, H, W).to(DEVICE)
    logging.info("Teacher trajectory generated and interpolated successfully.")
    return student_true_latents

class LearnableSolver(nn.Module):
    """
    一个可学习的线性多步求解器。
    它的参数是历史信息的组合权重。
    """
    def __init__(self, order=3, nfe=10):
        super().__init__()
        self.order = order
        # 为每个采样步骤学习一组独立的系数，能达到更好的效果
        # 总共有 nfe 个更新步骤，每个步骤需要 order-1 个系数
        self.coeffs = nn.Parameter(torch.zeros(nfe, self.order - 1))

    def get_d_i(self, eps_curr, history_eps, step_index):
        """
        根据当前和历史的噪声预测，计算修正后的噪声方向 d_i。
        """
        if not history_eps: # 第一步，没有历史信息
            return eps_curr

        # 提取当前步骤对应的系数
        step_coeffs = self.coeffs[step_index]

        correction = torch.zeros_like(eps_curr)
        num_history_to_use = min(len(history_eps), len(step_coeffs))
        for i in range(num_history_to_use):
            correction += step_coeffs[i] * (eps_curr - history_eps[i])
            
        d_i = eps_curr + correction
        return d_i
    
def main():
    # --- 1. 环境和模型设置 ---
    seed_everything(42)
    
    # 加载预训练的Stable Diffusion模型
    config = OmegaConf.load(CONFIG_PATH)
    model = instantiate_from_config(config.model)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(torch.float32) # 确保模型是float32
    model = model.to(DEVICE).eval()

    # --- 2. 准备调度和求解器 ---
    # 加载我们通过MuSOS找到的最优调度 T*
    student_schedule_t = torch.from_numpy(np.loadtxt(SCHEDULE_PATH, dtype=np.float32)).to(DEVICE)
    NFE = len(student_schedule_t) - 1
    logging.info(f"Loaded student schedule with {NFE} steps from {SCHEDULE_PATH}")

    # 实例化教师求解器
    teacher_sampler = DPMSolverSampler(model)

    # 实例化学生求解器
    student_solver = LearnableSolver(order=STUDENT_ORDER, nfe=NFE).to(DEVICE)

    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(student_solver.parameters(), lr=LEARNING_RATE)
    loss_fn_lpips = lpips.LPIPS(net='alex').to(DEVICE)
    loss_fn_l2 = nn.MSELoss()

    # --- 3. 开始训练循环 ---
    logging.info("Starting S4S Solver Training...")
    pbar = tqdm(range(TRAINING_STEPS))
    for step in pbar:
        optimizer.zero_grad()
        
        # 为了训练稳定，每次迭代使用固定的prompt和seed
        prompt = "a photograph of an astronaut riding a horse"
        seed_everything(42 + step) # 也可以用不同prompt和seed增加泛化性

        # --- A. 生成教师数据 ---
        teacher_latents = generate_teacher_trajectory(model, teacher_sampler, N_TEACHER_STEPS, student_schedule_t, prompt, GUIDANCE_SCALE)
        
        # --- B. 运行学生求解器 ---
        # 使用与教师相同的初始噪声
        student_x_curr = teacher_latents[0:1].clone() 
        eps_history = []
        
        # 模拟NFE步采样
        for i in range(NFE):
            t_curr, t_next = student_schedule_t[i], student_schedule_t[i+1]
            
            with torch.no_grad():
                uncond_eps = model.model.diffusion_model(student_x_curr, t_curr, context=model.get_learned_conditioning([""]).to(DEVICE))
                cond_eps = model.model.diffusion_model(student_x_curr, t_curr, context=model.get_learned_conditioning([prompt]).to(DEVICE))
                eps_curr = uncond_eps + GUIDANCE_SCALE * (cond_eps - uncond_eps)

            # 获取学生求解器预测的噪声方向
            d_i = student_solver.get_d_i(eps_curr, eps_history, step_index=i)
            
            # 使用标准的DDIM更新方程来执行一步（这是S4S论文中的做法）
            # alpha_t, alpha_t_prev = model.alphas_cumprod[t_curr], model.alphas_cumprod[t_next] # 这需要从t值映射到离散索引
            # pred_x0 = (student_x_curr - torch.sqrt(1 - alpha_t) * d_i) / torch.sqrt(alpha_t)
            # dir_xt = torch.sqrt(1 - alpha_t_prev) * d_i
            # student_x_next = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
            
            # 为了简单起见，我们直接用sampler的单步函数，但传入我们自己计算的d_i
            # 注意：这需要修改sampler以接受一个外部的d_i，或者在这里复现更新逻辑
            # 下面是一个简化的、类似DDIM的更新逻辑示例，您需要根据您的UniPC/DPM公式适配
            alpha_t, sigma_t = model.ns.marginal_alpha(t_curr), model.ns.marginal_std(t_curr)
            alpha_next, sigma_next = model.ns.marginal_alpha(t_next), model.ns.marginal_std(t_next)
            
            pred_x0 = (student_x_curr - sigma_t * d_i) / alpha_t
            student_x_curr = alpha_next * pred_x0 + sigma_next * d_i


            # 更新历史
            eps_history.insert(0, eps_curr.detach())
            if len(eps_history) >= student_solver.order - 1:
                eps_history.pop()

        # 学生最终得到的latent
        student_final_latent = student_x_curr

        # --- C. 计算损失 ---
        # 比较学生和教师在最后一步的latent的L2距离
        l2_loss = loss_fn_l2(student_final_latent, teacher_latents[-1:])
        
        # 比较最终生成图像的LPIPS距离
        with torch.no_grad():
             teacher_final_image = model.decode_first_stage(teacher_latents[-1:])
        student_final_image = model.decode_first_stage(student_final_latent)
        lpips_loss = loss_fn_lpips(student_final_image, teacher_final_image).mean()
        
        # 总损失
        total_loss = LPIPS_WEIGHT * lpips_loss + L2_WEIGHT * l2_loss
        
        # --- D. 更新 ---
        total_loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Epoch {step}, Total Loss: {total_loss.item():.4f}, LPIPS: {lpips_loss.item():.4f}, L2: {l2_loss.item():.4f}")

    # --- 4. 训练完成，保存并打印学到的系数 ---
    logging.info("S4S Solver training complete.")
    best_coeffs = student_solver.coeffs.detach().cpu().numpy()
    logging.info(f"Learned Coefficients:\n{best_coeffs}")
    
    # 保存系数，以便在采样时使用
    np.save(f"s4s_coeffs_nfe{NFE}.npy", best_coeffs)
    logging.info(f"Coefficients saved to s4s_coeffs_nfe{NFE}.npy")


if __name__ == '__main__':
    # 确保您的模型、采样器和路径都已正确设置
    main()
