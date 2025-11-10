
# # sampler.py

# import torch
# import numpy as np
# import os
# import json
# import logging
# import time
# from omegaconf import OmegaConf

# try:
#     from .uni_pc import NoiseScheduleVP, model_wrapper, UniPC
# except ImportError:
#     logging.error("Failed to import from .uni_pc. Ensure uni_pc.py is in the same package directory as sampler.py.")
#     from uni_pc import NoiseScheduleVP, model_wrapper, UniPC

# # from ldm.models.diffusion.step_optim_z1 import StepOptim
# from ldm.models.diffusion.step_optim import StepOptim


# class UniPCSampler(object):
#     def __init__(self, model, **kwargs):
#         super().__init__()
#         self.model = model
#         self.sampler_opt = kwargs.get('opt', None)

#         if not hasattr(self.model, 'alphas_cumprod'):
#             raise AttributeError("The provided model requires an 'alphas_cumprod' attribute for UniPCSampler.")
        
#         self.model_device_fallback = torch.device("cpu")
#         try:
#             if hasattr(self.model, 'betas') and self.model.betas is not None:
#                 self.model_device_fallback = self.model.betas.device
#             else:
#                 self.model_device_fallback = next(self.model.parameters()).device
#         except StopIteration:
#             logging.warning("Model has no parameters or 'betas'. Sampler device fallback to CPU. Ensure model is moved to device later.")
#         except AttributeError:
#             logging.warning("Model does not have 'betas' or parameters. Sampler device fallback to CPU.")

#     @torch.no_grad()
#     def sample(self,
#                S,
#                batch_size,
#                shape,
#                conditioning=None,
#                unconditional_guidance_scale=1.,
#                unconditional_conditioning=None,
#                x_T=None,
#                verbose=True,
#                custom_timesteps_tensor=None,
#                optimize_steps_on_demand=True,
#                step_optim_config=None,
#                step_optim_init_type='unif_t',
#                step_optim_eps_t_0_val=None,
#                step_optim_schedule_dir="./optimized_schedules/unipc",
#                uni_pc_order=3,
#                uni_pc_variant='bh1',
#                uni_pc_predict_x0=True,
#                uni_pc_thresholding=False,
#                uni_pc_max_val=1.0,
#                **kwargs
#                ):

#         if not isinstance(S, int) or S <= 0:
#             raise ValueError(f"Number of sampling steps S must be a positive integer, got {S}")

#         if conditioning is not None:
#             cond_shape_0 = conditioning[list(conditioning.keys())[0]].shape[0] if isinstance(conditioning, dict) else conditioning.shape[0]
#             if cond_shape_0 != batch_size:
#                 logging.warning(f"Warning: Conditioning batch size {cond_shape_0} != requested batch_size {batch_size}")

#         C, H_latent, W_latent = shape
#         size = (batch_size, C, H_latent, W_latent)
#         device = self.model_device_fallback

#         if x_T is None:
#             img = torch.randn(size, device=device, dtype=torch.float32)
#         else:
#             img = x_T.to(device=device, dtype=torch.float32)

#         # 1. Initialize NoiseScheduleVP (using the one from uni_pc.py via import)
#         #    NoiseScheduleVP's internal arrays (t_array, log_alpha_array) are created on CPU by default
#         #    in the __init__ of the uni_pc.py version I helped correct.
#         #    Its methods (like marginal_log_mean_coeff) then move these arrays to the device
#         #    of the input time tensor 't' (e.g., self.t_array.to(t.device)).
#         alphas_for_ns = self.model.alphas_cumprod.clone().detach().cpu().to(torch.float32)
#         ns_for_sampler = NoiseScheduleVP(schedule='discrete', alphas_cumprod=alphas_for_ns, dtype=torch.float32)
        
#         timesteps_to_use_by_unipc = None
#         effective_steps = S

#         if custom_timesteps_tensor is not None:
#             if verbose: logging.info("Using explicitly passed custom_timesteps_tensor.")
#             if not isinstance(custom_timesteps_tensor, torch.Tensor):
#                 timesteps_to_use_by_unipc = torch.tensor(custom_timesteps_tensor, dtype=img.dtype, device=device)
#             else:
#                 timesteps_to_use_by_unipc = custom_timesteps_tensor.to(device=device, dtype=img.dtype)
#         elif optimize_steps_on_demand:
#             config_hash_part = ""
#             if step_optim_config:
#                 try:
#                     config_str = json.dumps(dict(sorted(step_optim_config.items())), sort_keys=True)
#                     config_hash_part = f"_cfg{hash(config_str) & 0xFFFFFFFF:08x}"
#                 except TypeError:
#                     config_hash_part = "_customcfg"
            
#             schedule_filename = f"unipc_opt_{S}steps{config_hash_part}.txt"
#             schedule_file_path = os.path.join(step_optim_schedule_dir, schedule_filename)
#             os.makedirs(step_optim_schedule_dir, exist_ok=True)

#             if os.path.exists(schedule_file_path):
#                 if verbose: logging.info(f"Loading optimized timesteps from: {schedule_file_path}")
#                 try:
#                     loaded_timesteps_np = np.loadtxt(schedule_file_path, dtype=np.float32)
#                     timesteps_to_use_by_unipc = torch.from_numpy(loaded_timesteps_np).to(device=device, dtype=img.dtype)
#                     if len(timesteps_to_use_by_unipc) != S + 1:
#                         logging.warning(f"Loaded timesteps length {len(timesteps_to_use_by_unipc)} from {schedule_file_path} "
#                                         f"does not match S+1 ({S+1}). Regenerating...")
#                         timesteps_to_use_by_unipc = None
#                     else:
#                         if verbose: logging.info(f"Successfully loaded {len(timesteps_to_use_by_unipc)} timesteps from file.")
#                 except Exception as e:
#                     logging.warning(f"Error loading timesteps from {schedule_file_path}: {e}. Will regenerate.")
#                     timesteps_to_use_by_unipc = None
            
#             if timesteps_to_use_by_unipc is None:
#                 if verbose: logging.info(f"Optimized schedule not found or invalid at {schedule_file_path}. Generating new schedule for {S} steps...")
                
#                 current_step_optim_config = {
#                     'p_is_dynamic': True, 'p_val_at_T': 2.0, 'p_val_at_eps': 1.5,
#                     'objective_type': 'paper_power_q', 'power_q_val': 1.6,
#                     'use_pf_inspired_error_metric': True
#                 }
#                 if step_optim_config: current_step_optim_config.update(step_optim_config)

#                 # StepOptim is initialized with ns_for_sampler (which is an instance of uni_pc.NoiseScheduleVP)
#                 # ns_for_sampler's internal arrays are on CPU. StepOptim operates on CPU.
#                 step_optimizer = StepOptim(ns=ns_for_sampler, **current_step_optim_config) 
                
#                 _eps_t_0 = step_optim_eps_t_0_val
#                 if _eps_t_0 is None:
#                     _eps_t_0 = (1. / ns_for_sampler.total_N) if ns_for_sampler.schedule == 'discrete' and ns_for_sampler.total_N > 0 else 1e-3

#                 optimized_ts_desc, _ = step_optimizer.get_ts_lambdas(
#                     N_intervals=S, eps_t_0_val=_eps_t_0, initType=step_optim_init_type
#                 )
#                 timesteps_to_use_by_unipc = optimized_ts_desc.to(device=device, dtype=img.dtype)
                
#                 try:
#                     np.savetxt(schedule_file_path, timesteps_to_use_by_unipc.cpu().numpy(), fmt='%.8f')
#                     if verbose: logging.info(f"Optimized timesteps saved to {schedule_file_path}")
#                 except Exception as e:
#                     logging.error(f"Error saving timesteps to {schedule_file_path}: {e}")
#         else:
#             if verbose: logging.info("Using UniPC's default timestep generation.")

#         if timesteps_to_use_by_unipc is not None:
#             if len(timesteps_to_use_by_unipc) > 1 and timesteps_to_use_by_unipc[0].item() < timesteps_to_use_by_unipc[-1].item():
#                 timesteps_to_use_by_unipc = torch.flip(timesteps_to_use_by_unipc, dims=[0])
#             effective_steps = len(timesteps_to_use_by_unipc) - 1
#             if S != effective_steps and verbose:
#                  logging.info(f"UniPCSampler: Effective steps set to {effective_steps} based on custom_timesteps length (original S was {S}).")
        
#         t_start_unipc = timesteps_to_use_by_unipc[0].item() if timesteps_to_use_by_unipc is not None else ns_for_sampler.T
#         t_end_for_unipc = timesteps_to_use_by_unipc[-1].item() if timesteps_to_use_by_unipc is not None \
#             else ((1. / ns_for_sampler.total_N) if ns_for_sampler.schedule == 'discrete' and ns_for_sampler.total_N > 0 else 1e-3)

#         # The ns_for_sampler passed to model_wrapper and UniPC should be the instance we have.
#         # Its internal methods will handle moving its arrays (e.g. t_array) to the device of the input 't' or 'lamb'
#         model_fn_for_unipc = model_wrapper(
#             model=lambda x, t_continuous, c: self.model.apply_model(x, t_continuous, c),
#             noise_schedule=ns_for_sampler, # Pass the ns_for_sampler instance directly
#             model_type=getattr(self.model, 'model_type', "noise"),
#             guidance_type="classifier-free",
#             condition=conditioning,
#             unconditional_condition=unconditional_conditioning,
#             guidance_scale=unconditional_guidance_scale,
#         )

#         uni_pc_solver = UniPC(
#             model_fn=model_fn_for_unipc,
#             noise_schedule=ns_for_sampler, # Pass the ns_for_sampler instance directly
#             predict_x0=uni_pc_predict_x0,
#             thresholding=uni_pc_thresholding,
#             max_val=uni_pc_max_val,
#             variant=uni_pc_variant
#         )
        
#         if verbose:
#             logging.info(f"UniPCSampler: Calling UniPC.sample with effective_steps={effective_steps}, order={uni_pc_order}, "
#                          f"t_start={t_start_unipc:.4f}, t_end={t_end_for_unipc:.4f}, predict_x0={uni_pc_predict_x0}, thresholding={uni_pc_thresholding}")
#             if timesteps_to_use_by_unipc is not None:
#                 logging.info(f"Using custom/optimized timesteps (first 5): {timesteps_to_use_by_unipc[:5].cpu().tolist()}")

#         x_sampled = uni_pc_solver.sample(
#             x=img,
#             steps=effective_steps,
#             t_start=t_start_unipc,
#             t_end=t_end_for_unipc,
#             order=uni_pc_order,
#             skip_type=kwargs.get('uni_pc_skip_type', 'time_uniform'),
#             method="multistep",
#             lower_order_final=kwargs.get('lower_order_final', True),
#             denoise_to_zero=kwargs.get('denoise_to_zero', False),
#             custom_timesteps=timesteps_to_use_by_unipc
#         )

#         return x_sampled.to(device), None

# sampler.py

import torch
import numpy as np
import os
import json
import logging
import time 
from omegaconf import OmegaConf
from tqdm import tqdm # UniPCSampler 内部的 tqdm 可以不加，但如果想看进度可以加

try:
    from .uni_pc import NoiseScheduleVP, model_wrapper, UniPC 
except ImportError:
    logging.error("Failed to import from .uni_pc. Ensure uni_pc.py is in the same package directory as sampler.py.")
    from uni_pc import NoiseScheduleVP, model_wrapper, UniPC

try:
    from ldm.models.diffusion.step_optim_z1 import StepOptim
except ImportError:
    logging.error("Failed to import StepOptim from ldm.models.diffusion.step_optim_z1. Ensure path is correct.")
    try:
        from step_optim_z1 import StepOptim
    except ImportError as e_so:
        logging.error(f"Failed to import StepOptim directly: {e_so}")
        raise


class UniPCSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model 
        self.sampler_opt = kwargs.get('opt', None) 

        if not hasattr(self.model, 'alphas_cumprod'):
            raise AttributeError("The provided model requires an 'alphas_cumprod' attribute for UniPCSampler.")
        
        self.model_device_fallback = torch.device("cpu") 
        try:
            if hasattr(self.model, 'betas') and self.model.betas is not None:
                self.model_device_fallback = self.model.betas.device
            else:
                self.model_device_fallback = next(self.model.parameters()).device
        except StopIteration:
            logging.warning("Model has no parameters or 'betas'. Sampler device fallback to CPU. Ensure model is moved to device later.")
        except AttributeError:
            logging.warning("Model does not have 'betas' or parameters. Sampler device fallback to CPU.")

    @torch.no_grad()
    def sample(self,
               S, 
               batch_size,
               shape, 
               conditioning=None,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               x_T=None,
               verbose=True, # Controls overall verbosity of this method
               custom_timesteps_tensor=None, 
               optimize_steps_on_demand=True, 
               step_optim_config=None, 
               step_optim_init_type='unif_t',
               step_optim_eps_t_0_val=None,
               step_optim_schedule_dir="./optimized_schedules/unipc", 
               uni_pc_order=3,
               uni_pc_variant='bh1',
               uni_pc_predict_x0=True,    
               uni_pc_thresholding=False, 
               uni_pc_max_val=1.0,
               # New argument to control per-step UniPC verbosity (if UniPC supports it)
               uni_pc_verbose_steps=False, 
               **kwargs 
               ):

        if not isinstance(S, int) or S <= 0:
            raise ValueError(f"Number of sampling steps S must be a positive integer, got {S}")

        if conditioning is not None:
            cond_shape_0 = conditioning[list(conditioning.keys())[0]].shape[0] if isinstance(conditioning, dict) else conditioning.shape[0]
            if cond_shape_0 != batch_size:
                logging.warning(f"Warning: Conditioning batch size {cond_shape_0} != requested batch_size {batch_size}")

        C, H_latent, W_latent = shape
        size = (batch_size, C, H_latent, W_latent)
        device = self.model_device_fallback 

        if x_T is None:
            img = torch.randn(size, device=device, dtype=torch.float32)
        else:
            img = x_T.to(device=device, dtype=torch.float32)

        alphas_for_ns = self.model.alphas_cumprod.clone().detach().cpu().to(torch.float32)
        ns_for_sampler = NoiseScheduleVP(schedule='discrete', alphas_cumprod=alphas_for_ns, dtype=torch.float32)
        
        timesteps_to_use_by_unipc = None
        effective_steps = S

        # --- Logic for obtaining or generating custom_timesteps_tensor (same as before) ---
        if custom_timesteps_tensor is not None:
            # ... (same as before)
            if verbose: logging.info("Using explicitly passed custom_timesteps_tensor.")
            if not isinstance(custom_timesteps_tensor, torch.Tensor):
                timesteps_to_use_by_unipc = torch.tensor(custom_timesteps_tensor, dtype=img.dtype, device=device)
            else:
                timesteps_to_use_by_unipc = custom_timesteps_tensor.to(device=device, dtype=img.dtype)

        elif optimize_steps_on_demand:
            # ... (same logic for loading or generating via StepOptim and saving) ...
            config_hash_part = ""
            if step_optim_config:
                try:
                    config_str = json.dumps(dict(sorted(step_optim_config.items())), sort_keys=True)
                    config_hash_part = f"_cfg{hash(config_str) & 0xFFFFFFFF:08x}" 
                except TypeError: 
                    config_hash_part = "_customcfg"
            
            schedule_filename = f"unipc_opt_{S}steps{config_hash_part}.txt"
            schedule_file_path = os.path.join(step_optim_schedule_dir, schedule_filename)
            os.makedirs(step_optim_schedule_dir, exist_ok=True)

            if os.path.exists(schedule_file_path):
                if verbose: logging.info(f"Loading optimized timesteps from: {schedule_file_path}")
                try:
                    loaded_timesteps_np = np.loadtxt(schedule_file_path, dtype=np.float32)
                    timesteps_to_use_by_unipc = torch.from_numpy(loaded_timesteps_np).to(device=device, dtype=img.dtype)
                    if len(timesteps_to_use_by_unipc) != S + 1:
                        logging.warning(f"Loaded timesteps length {len(timesteps_to_use_by_unipc)} from {schedule_file_path} "
                                        f"does not match S+1 ({S+1}). Regenerating...")
                        timesteps_to_use_by_unipc = None
                    else:
                        if verbose: logging.info(f"Successfully loaded {len(timesteps_to_use_by_unipc)} timesteps from file.")
                except Exception as e:
                    logging.warning(f"Error loading timesteps from {schedule_file_path}: {e}. Will regenerate.")
                    timesteps_to_use_by_unipc = None
            
            if timesteps_to_use_by_unipc is None: 
                if verbose: logging.info(f"Optimized schedule not found or invalid at {schedule_file_path}. Generating new schedule for {S} steps...")
                
                current_step_optim_config = {
                    'p_is_dynamic': True, 'p_val_at_T': 2.0, 'p_val_at_eps': 1.5,
                    'objective_type': 'paper_power_q', 'power_q_val': 1.6,
                    'use_pf_inspired_error_metric': True
                }
                if step_optim_config: current_step_optim_config.update(step_optim_config)

                step_optimizer = StepOptim(ns=ns_for_sampler.to(torch.device("cpu")), **current_step_optim_config) 
                
                _eps_t_0 = step_optim_eps_t_0_val
                if _eps_t_0 is None:
                    _eps_t_0 = (1. / ns_for_sampler.total_N) if ns_for_sampler.schedule == 'discrete' and ns_for_sampler.total_N > 0 else 1e-3

                optimized_ts_desc, _ = step_optimizer.get_ts_lambdas(
                    N_intervals=S, eps_t_0_val=_eps_t_0, initType=step_optim_init_type
                )
                timesteps_to_use_by_unipc = optimized_ts_desc.to(device=device, dtype=img.dtype)
                
                try:
                    np.savetxt(schedule_file_path, timesteps_to_use_by_unipc.cpu().numpy(), fmt='%.8f')
                    if verbose: logging.info(f"Optimized timesteps saved to {schedule_file_path}")
                except Exception as e:
                    logging.error(f"Error saving timesteps to {schedule_file_path}: {e}")
        else:
            if verbose: logging.info("Using UniPC's default timestep generation.")
        # --- End of timestep obtaining logic ---

        if timesteps_to_use_by_unipc is not None:
            if len(timesteps_to_use_by_unipc) > 1 and timesteps_to_use_by_unipc[0].item() < timesteps_to_use_by_unipc[-1].item():
                timesteps_to_use_by_unipc = torch.flip(timesteps_to_use_by_unipc, dims=[0])
            effective_steps = len(timesteps_to_use_by_unipc) - 1
            if S != effective_steps and verbose:
                 logging.info(f"UniPCSampler: Effective steps set to {effective_steps} based on custom_timesteps length (original S was {S}).")
        else: # If no custom steps and no on-demand optimization, effective_steps remains S
            effective_steps = S

        
        t_start_unipc = timesteps_to_use_by_unipc[0].item() if timesteps_to_use_by_unipc is not None else ns_for_sampler.T
        t_end_for_unipc = timesteps_to_use_by_unipc[-1].item() if timesteps_to_use_by_unipc is not None \
            else ((1. / ns_for_sampler.total_N) if ns_for_sampler.schedule == 'discrete' and ns_for_sampler.total_N > 0 else 1e-3)

        model_fn_for_unipc = model_wrapper(
            model=lambda x, t_continuous, c: self.model.apply_model(x, t_continuous, c),
            noise_schedule=ns_for_sampler, 
            model_type=getattr(self.model, 'model_type', "noise"),
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        uni_pc_solver = UniPC(
            model_fn=model_fn_for_unipc,
            noise_schedule=ns_for_sampler, 
            predict_x0=uni_pc_predict_x0,
            thresholding=uni_pc_thresholding,
            max_val=uni_pc_max_val, 
            variant=uni_pc_variant
        )
        
        if verbose:
            logging.info(f"UniPCSampler: Preparing to call UniPC.sample with effective_steps={effective_steps}, order={uni_pc_order}, "
                         f"t_start={t_start_unipc:.4f}, t_end={t_end_for_unipc:.4f}, predict_x0={uni_pc_predict_x0}, thresholding={uni_pc_thresholding}")
            if timesteps_to_use_by_unipc is not None:
                logging.info(f"Using custom/optimized timesteps (first 5): {timesteps_to_use_by_unipc[:5].cpu().tolist()}")

        # ***** PRECISE SAMPLING TIME MEASUREMENT *****
        core_sampling_start_time = time.time()

        x_sampled = uni_pc_solver.sample(
            x=img,
            steps=effective_steps,
            t_start=t_start_unipc,
            t_end=t_end_for_unipc,
            order=uni_pc_order,
            skip_type=kwargs.get('uni_pc_skip_type', 'time_uniform'), 
            method="multistep",
            lower_order_final=kwargs.get('lower_order_final', True),
            denoise_to_zero=kwargs.get('denoise_to_zero', False),
            custom_timesteps=timesteps_to_use_by_unipc,
            verbose_steps=uni_pc_verbose_steps # Pass this to UniPC.sample
        )
        
        core_sampling_end_time = time.time()
        core_sampling_duration = core_sampling_end_time - core_sampling_start_time
        if verbose:
            logging.info(f"⏱️ UniPC Core Sampling Process took: {core_sampling_duration:.4f} seconds for {effective_steps} steps.")
        # *********************************************

        return x_sampled.to(device), {"core_sampling_time": core_sampling_duration} # Optionally return timing info



# # sampler.py (简化版，配合外部处理 StepOptim 的 test_unipc.py)
# import time
# import torch
# import numpy as np
# import logging
# from .uni_pc import NoiseScheduleVP, model_wrapper, UniPC # 确保从正确路径导入

# class UniPCSampler(object):
#     def __init__(self, model, **kwargs):
#         super().__init__()
#         self.model = model
#         if not hasattr(self.model, 'alphas_cumprod'):
#             raise AttributeError("Model needs 'alphas_cumprod'.")
#         try:
#             self.model_device = self.model.betas.device
#         except AttributeError:
#             try:
#                 self.model_device = next(self.model.parameters()).device
#             except StopIteration:
#                 self.model_device = torch.device("cpu")
#                 logging.warning("Cannot determine model device, defaulting to CPU for sampler.")


#     @torch.no_grad()
#     def sample(self,
#                S, 
#                batch_size,
#                shape, 
#                conditioning=None,
#                unconditional_guidance_scale=1.,
#                unconditional_conditioning=None,
#                x_T=None,
#                verbose=True,
#                custom_timesteps_tensor=None, # <<< 主要依赖这个参数
#                # UniPC specific parameters
#                uni_pc_order=3,
#                uni_pc_variant='bh1',
#                uni_pc_predict_x0=True,    
#                uni_pc_thresholding=False, 
#                uni_pc_max_val=1.0,
#                **kwargs 
#                ):

#         if conditioning is not None:
#             cond_shape_0 = conditioning[list(conditioning.keys())[0]].shape[0] if isinstance(conditioning, dict) else conditioning.shape[0]
#             if cond_shape_0 != batch_size:
#                 logging.warning(f"Warning: Cond batch size {cond_shape_0} != req batch_size {batch_size}")

#         C, H_latent, W_latent = shape
#         size = (batch_size, C, H_latent, W_latent)
#         device = self.model_device

#         if x_T is None:
#             img = torch.randn(size, device=device, dtype=torch.float32)
#         else:
#             img = x_T.to(device=device, dtype=torch.float32)

#         # NoiseScheduleVP for UniPC and model_wrapper
#         alphas_for_ns = self.model.alphas_cumprod.clone().detach().cpu().to(torch.float32)
#         ns_instance = NoiseScheduleVP(schedule='discrete', alphas_cumprod=alphas_for_ns, dtype=torch.float32)
        
#         model_fn_for_unipc = model_wrapper(
#             model=lambda x, t_continuous, c: self.model.apply_model(x, t_continuous, c),
#             noise_schedule=ns_instance, # model_wrapper might need ns on device
#             model_type=getattr(self.model, 'model_type', "noise"),
#             guidance_type="classifier-free",
#             condition=conditioning,
#             unconditional_condition=unconditional_conditioning,
#             guidance_scale=unconditional_guidance_scale,
#         )

#         uni_pc_solver = UniPC(
#             model_fn=model_fn_for_unipc,
#             noise_schedule=ns_instance, # UniPC might need ns on device
#             predict_x0=uni_pc_predict_x0,
#             thresholding=uni_pc_thresholding,
#             max_val=uni_pc_max_val, 
#             variant=uni_pc_variant
#         )
        
#         effective_steps = S
#         timesteps_for_call = None
#         t_start_call = ns_instance.T
#         t_end_call = (1. / ns_instance.total_N) if ns_instance.schedule == 'discrete' and hasattr(ns_instance, 'total_N') and ns_instance.total_N > 0 else 1e-3

#         if custom_timesteps_tensor is not None:
#             if not isinstance(custom_timesteps_tensor, torch.Tensor):
#                 timesteps_for_call = torch.tensor(custom_timesteps_tensor, dtype=img.dtype, device=device)
#             else:
#                 timesteps_for_call = custom_timesteps_tensor.to(device=device, dtype=img.dtype)
            
#             if len(timesteps_for_call) > 1 and timesteps_for_call[0].item() < timesteps_for_call[-1].item():
#                 timesteps_for_call = torch.flip(timesteps_for_call, dims=[0]) # Ensure descending
            
#             effective_steps = len(timesteps_for_call) - 1
#             if S != effective_steps and verbose:
#                  logging.info(f"UniPCSampler: Effective steps set to {effective_steps} based on custom_timesteps_tensor (original S was {S}).")
#             t_start_call = timesteps_for_call[0].item()
#             t_end_call = timesteps_for_call[-1].item()
#         elif verbose:
#             logging.info(f"UniPCSampler: No custom_timesteps_tensor provided. UniPC will use its default for {S} steps.")
            
#         if verbose:
#             logging.info(f"UniPCSampler: Calling UniPC.sample with steps={effective_steps}, order={uni_pc_order}, "
#                          f"t_start={t_start_call:.4f}, t_end={t_end_call:.4f}")
#             if timesteps_for_call is not None:
#                  logging.info(f"Using custom timesteps (first 5): {timesteps_for_call[:5].cpu().tolist()}")
        
#         # Core sampling call
#         sampling_start_time = time.time()
#         x_sampled = uni_pc_solver.sample(
#             x=img,
#             steps=effective_steps,
#             t_start=t_start_call,
#             t_end=t_end_call,
#             order=uni_pc_order,
#             skip_type=kwargs.get('uni_pc_skip_type', 'time_uniform'), 
#             method="multistep",
#             lower_order_final=kwargs.get('lower_order_final', True),
#             denoise_to_zero=kwargs.get('denoise_to_zero', False),
#             custom_timesteps=timesteps_for_call # Pass to modified UniPC.sample
#         )
#         sampling_duration = time.time() - sampling_start_time
#         if verbose:
#             logging.info(f"⏱️ UniPC Core Sampling (inside UniPCSampler) took: {sampling_duration:.4f} seconds for {effective_steps} steps.")

#         return x_sampled.to(device), {"core_sampling_time": sampling_duration}