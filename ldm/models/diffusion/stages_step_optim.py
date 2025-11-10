import json
import random
import torch
import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint, differential_evolution
import logging
import time
import os

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def interpolate_fn_for_ns(x_query, x_pts, y_pts, device=torch.device('cpu'), dtype=torch.float32):
    """
    A simple 1D linear interpolation function using PyTorch.
    It finds where `x_query` values would fit within `x_pts` and interpolates the corresponding `y_pts`.
    """
    x_query_orig_shape = x_query.shape
    x_query_flat = x_query.reshape(-1).to(device, dtype=dtype)
    x_pts_flat = x_pts.reshape(-1).to(device, dtype=dtype)
    y_pts_flat = y_pts.reshape(-1).to(device, dtype=dtype)

    indices = torch.searchsorted(x_pts_flat, x_query_flat)
    indices = torch.clamp(indices, 1, len(x_pts_flat) - 1)

    x_prev, x_next = x_pts_flat[indices - 1], x_pts_flat[indices]
    y_prev, y_next = y_pts_flat[indices - 1], y_pts_flat[indices]

    denom = x_next - x_prev
    weight = torch.where(denom.abs() < 1e-12, torch.zeros_like(denom), (x_query_flat - x_prev) / denom)
    
    interpolated_y_flat = y_prev + weight * (y_next - y_prev)
    
    try:
        if len(x_query_orig_shape) > 1 and interpolated_y_flat.numel() == x_query.numel():
              return interpolated_y_flat.reshape(x_query_orig_shape)
        return interpolated_y_flat.reshape(x_query_flat.shape[0], -1)
    except RuntimeError:
        logging.warning("Interpolate_fn_for_ns: Reshape failed, returning flat tensor.")
        return interpolated_y_flat

class NoiseScheduleVP:
    """
    Manages the noise schedule for a variance-preserving (VP) SDE framework.
    It can handle both discrete schedules (like DDPM's) and continuous ones.
    The main job is to provide functions that map time `t` (from 0 to T)
    to key diffusion parameters like alpha, sigma, and lambda.
    """
    def __init__(self, schedule_name='discrete', betas=None, alphas_cumprod=None,
                 continuous_beta_0=0.1, continuous_beta_1=20.,
                 num_trained_timesteps=1000,
                 dtype=torch.float64, device=torch.device('cpu')):
        
        self.schedule_name = schedule_name
        self._device = device
        self.dtype = dtype
        self.num_trained_timesteps = int(num_trained_timesteps)

        if self.schedule_name == 'discrete':
            if alphas_cumprod is not None:
                if not isinstance(alphas_cumprod, torch.Tensor): alphas_cumprod = torch.tensor(alphas_cumprod, dtype=self.dtype)
                self.alphas_cumprod_model = alphas_cumprod.to(self._device)
            else:
                if betas is None:
                    logging.info("Discrete schedule: No betas/alphas_cumprod. Using default DDPM linear betas for 1000 steps.")
                    self.num_trained_timesteps = 1000
                    betas = torch.linspace(1e-4, 0.02, self.num_trained_timesteps, dtype=self.dtype)
                if not isinstance(betas, torch.Tensor): betas = torch.tensor(betas, dtype=self.dtype)
                self.betas = betas.to(self._device)
                self.alphas = 1.0 - self.betas
                self.alphas_cumprod_model = torch.cumprod(self.alphas, dim=0)
            
            if len(self.alphas_cumprod_model) != self.num_trained_timesteps:
                logging.warning(f"Discrete schedule: Length of alphas_cumprod ({len(self.alphas_cumprod_model)}) "
                                f"differs from num_trained_timesteps ({self.num_trained_timesteps}). "
                                f"Using length of alphas_cumprod as num_trained_timesteps.")
                self.num_trained_timesteps = len(self.alphas_cumprod_model)

            self.total_N = self.num_trained_timesteps
            self.T = 1.0
            self.min_t_schedule = 1.0 / self.num_trained_timesteps
            
            self._t_discrete_map = (torch.arange(end=self.num_trained_timesteps, device=self._device, dtype=self.dtype) + 1.) / self.num_trained_timesteps
            self._log_alpha_discrete_map = 0.5 * torch.log(self.alphas_cumprod_model.clamp(min=1e-40)) 
            
            self._sorted_log_alphas_for_inverse, self._sort_indices_for_inverse = torch.sort(self._log_alpha_discrete_map)
            self._sorted_ts_for_inverse = self._t_discrete_map[self._sort_indices_for_inverse]

        elif self.schedule_name == 'linear_vp':
            self.beta_0 = float(continuous_beta_0)
            self.beta_1 = float(continuous_beta_1)
            self.T = 1.0
            self.min_t_schedule = 1e-5
            self.total_N = self.num_trained_timesteps
        else:
            raise ValueError(f"Unsupported schedule_name: {self.schedule_name}")
        logging.info(f"NoiseScheduleVP initialized: type={self.schedule_name}, T={self.T:.4f}, min_t={self.min_t_schedule:.4e}, N_model={self.num_trained_timesteps}")

    def to(self, device):
        """Moves all relevant tensors to the specified device."""
        self._device = device
        if self.schedule_name == 'discrete':
            if hasattr(self, 'alphas_cumprod_model'): self.alphas_cumprod_model = self.alphas_cumprod_model.to(device)
            if hasattr(self, '_log_alpha_discrete_map'): self._log_alpha_discrete_map = self._log_alpha_discrete_map.to(device)
            if hasattr(self, '_t_discrete_map'): self._t_discrete_map = self._t_discrete_map.to(device)
            if hasattr(self, '_sorted_log_alphas_for_inverse'): self._sorted_log_alphas_for_inverse = self._sorted_log_alphas_for_inverse.to(device)
            if hasattr(self, '_sorted_ts_for_inverse'): self._sorted_ts_for_inverse = self._sorted_ts_for_inverse.to(device)
        return self

    def marginal_log_mean_coeff(self, t):
        """ Computes log(alpha_t), where alpha_t = sqrt(alpha_bar_t) """
        t_tensor = torch.as_tensor(t, device=self._device, dtype=self.dtype).clamp(min=self.min_t_schedule, max=self.T)
        if self.schedule_name == 'discrete':
            return interpolate_fn_for_ns(t_tensor.reshape(-1, 1), 
                                         self._t_discrete_map.reshape(1, -1), 
                                         self._log_alpha_discrete_map.reshape(1, -1), 
                                         device=self._device, dtype=self.dtype).reshape(t_tensor.shape)
        elif self.schedule_name == 'linear_vp':
            return -0.25 * t_tensor**2 * (self.beta_1 - self.beta_0) - 0.5 * t_tensor * self.beta_0
        else: raise NotImplementedError(self.schedule_name)

    def marginal_alpha(self, t):
        """ Computes alpha_t = exp(log(alpha_t)) """
        return torch.exp(self.marginal_log_mean_coeff(t))
    
    def marginal_std(self, t):
        """ Computes sigma_t = sqrt(1 - alpha_t^2) """
        log_alpha_t_sq = 2. * self.marginal_log_mean_coeff(t)
        return torch.sqrt((1. - torch.exp(log_alpha_t_sq)).clamp(min=1e-12))

    def marginal_lambda(self, t):
        """ Computes lambda_t = log(alpha_t / sigma_t) """
        log_alpha_t = self.marginal_log_mean_coeff(t)
        log_sigma_t = 0.5 * torch.log((1. - torch.exp(2. * log_alpha_t)).clamp(min=1e-40))
        return log_alpha_t - log_sigma_t

    def inverse_lambda(self, lamb):
        """ Computes t given lambda_t. This is the inverse of the marginal_lambda function. """
        lamb_tensor = torch.as_tensor(lamb, device=self._device, dtype=self.dtype)
        log_alpha_target = -0.5 * torch.logaddexp(torch.zeros_like(lamb_tensor), -2. * lamb_tensor)
        
        if self.schedule_name == 'discrete':
            return interpolate_fn_for_ns(log_alpha_target.reshape(-1,1), 
                                         self._sorted_log_alphas_for_inverse.reshape(1,-1),
                                         self._sorted_ts_for_inverse.reshape(1,-1),
                                         device=self._device, dtype=self.dtype).reshape(lamb_tensor.shape)
        elif self.schedule_name == 'linear_vp':
            if abs(self.beta_1 - self.beta_0) < 1e-9 :
                if abs(self.beta_0) < 1e-9: return torch.full_like(lamb, self.T)
                t_candidate = -2 * log_alpha_target / self.beta_0
            else:
                tmp_for_inv = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb_tensor, torch.zeros_like(lamb_tensor))
                delta_sqrt_term_inv = torch.sqrt((self.beta_0**2 + tmp_for_inv).clamp(min=0))
                t_candidate = tmp_for_inv / (delta_sqrt_term_inv + self.beta_0) / (self.beta_1 - self.beta_0)
            return t_candidate.clamp(min=self.min_t_schedule, max=self.T)
        else:
            raise NotImplementedError(f"inverse_lambda for schedule {self.schedule_name}")

    def edm_sigma(self, t):
        """ Computes the equivalent sigma from the EDM framework (sigma_t / alpha_t). """
        t_tensor = torch.as_tensor(t, device=self._device, dtype=self.dtype)
        return self.marginal_std(t_tensor) / self.marginal_alpha(t_tensor).clamp(min=1e-9)

    def inverse_edm_sigma(self, edm_s):
        """ Computes t given the EDM sigma. """
        edm_s_tensor = torch.as_tensor(edm_s, device=self._device, dtype=self.dtype).clamp(min=0)
        alpha_vp_target = 1.0 / torch.sqrt(edm_s_tensor**2 + 1.0)
        log_alpha_vp_target = torch.log(alpha_vp_target.clamp(min=1e-40))

        if self.schedule_name == 'discrete':
            return interpolate_fn_for_ns(log_alpha_vp_target.reshape(-1,1), 
                                         self._sorted_log_alphas_for_inverse.reshape(1,-1), 
                                         self._sorted_ts_for_inverse.reshape(1,-1), 
                                         device=self._device, dtype=self.dtype).reshape(edm_s_tensor.shape)
        elif self.schedule_name == 'linear_vp':
            t_low = torch.full_like(edm_s_tensor, self.min_t_schedule)
            t_high = torch.full_like(edm_s_tensor, self.T)
            for _ in range(30):
                t_mid = (t_low + t_high) / 2.0
                log_alpha_mid = self.marginal_log_mean_coeff(t_mid)
                t_low = torch.where(log_alpha_mid > log_alpha_vp_target, t_mid, t_low)
                t_high = torch.where(log_alpha_mid <= log_alpha_vp_target, t_mid, t_high)
            res_t = (t_low + t_high) / 2.0
            return res_t.clamp(min=self.min_t_schedule, max=self.T)
        else:
            raise NotImplementedError(f"inverse_edm_sigma for schedule {self.schedule_name} not implemented.")


class StepOptim(object):
    """
    This class handles the optimization of sampler timesteps.
    It computes an objective function representing the theoretical sampling error
    and uses scipy.minimize to find the timestep distribution that minimizes this error.
    
    SIMPLIFIED: All unused parameters related to dynamic p and rho scaling have been removed.
    """
    def __init__(self, ns: NoiseScheduleVP,
                 p_fixed_val=2.0,
                 objective_type='midpoint'):
        
        self.ns = ns 
        self.T_val = ns.T 
        self.min_t_val = ns.min_t_schedule 
        logging.debug(f"StepOptim initialized with T_val={self.T_val}, min_t_val={self.min_t_val}")

        self.p_fixed_val = float(p_fixed_val)
        self.objective_type = objective_type
        
        valid_objectives = ['midpoint', 'legacy_unipc', 'midpoint_unipc']
        if self.objective_type not in valid_objectives:
            raise ValueError(f"Unknown objective_type: {self.objective_type}. Valid: {valid_objectives}")

    # --- Numpy-based helper functions for the scipy optimizer ---
    def _to_numpy(self, x_tensor): return x_tensor.cpu().detach().numpy()
    def alpha(self, t_val_np): return self._to_numpy(self.ns.marginal_alpha(t_val_np))
    def sigma(self, t_val_np): return self._to_numpy(self.ns.marginal_std(t_val_np))
    def lambda_func(self, t_val_np): return self._to_numpy(self.ns.marginal_lambda(t_val_np))
    def inverse_lambda(self, lamb_val_np): return self._to_numpy(self.ns.inverse_lambda(lamb_val_np))
    
    # --- Functions from the UniPC for calculating error coefficients ---
    def H0(self, h): return np.expm1(h) # exp(h) - 1
    def H1(self, h): h_exp = np.exp(h); return h_exp * h - (h_exp - 1)
    def H2(self, h): h_exp = np.exp(h); return h_exp * h**2 - 2 * self.H1(h)
    def H3(self, h): h_exp = np.exp(h); return h_exp * h**3 - 3 * self.H2(h)

    def _calculate_epsilon_tilde_vec(self, lambda_points_for_eval):
        """
        Calculates the model-independent error term ε̃_t at the given lambda points.
        SIMPLIFIED: This now only uses a fixed `p` value.
        """
        t_points = self.inverse_lambda(lambda_points_for_eval)
        alpha_vals = self.alpha(t_points)
        sigma_vals = self.sigma(t_points)
        
        # Use the fixed p value for the error term calculation.
        p_values_for_epsilon_tilde = np.full_like(lambda_points_for_eval, self.p_fixed_val)
        
        # Calculate the base error term: ε̃_base = (σ^p) / α
        epsilon_tilde_base = (sigma_vals**p_values_for_epsilon_tilde) / alpha_vals.clip(min=1e-9)
        
        return epsilon_tilde_base

    def _sel_lambdas_obj_calculator(self, lambda_vec_opt_part, N_intervals, eps_t_0_val, trunc_num_setting):
        """
        The core objective function passed to the optimizer.
        SIMPLIFIED: No longer needs arguments for dynamic p or rho scaling.
        """
        # --- 1. Setup: Reconstruct the full schedule ---
        lambda_eps_val = self.lambda_func(eps_t_0_val).item()
        lambda_T_val = self.lambda_func(self.T_val).item()
        _lambdas_boundary = sorted([lambda_T_val, lambda_eps_val]) 
        
        lambda_vec_opt_part_np = np.array(lambda_vec_opt_part, dtype=np.float64)
        temp_concat = np.concatenate(([_lambdas_boundary[0]], lambda_vec_opt_part_np, [_lambdas_boundary[1]]))
        lambda_vec_ext_np = np.sort(np.unique(temp_concat))

        if len(lambda_vec_ext_np) != N_intervals + 1:
            logging.debug(f"Length mismatch in optimizer: len(unique lambda_vec)={len(lambda_vec_ext_np)} vs N_intervals+1={N_intervals+1}.")
            return 1e12
            
        hv_np = np.diff(lambda_vec_ext_np)
        hv_np = np.maximum(hv_np, 1e-9)
        elv_np = np.exp(lambda_vec_ext_np)
        
        # --- 2. Calculate Objective Value ---
        if self.objective_type == 'midpoint':
            lambda_midpoints = (lambda_vec_ext_np[:-1] + lambda_vec_ext_np[1:]) / 2.0
            epsilon_tilde_values = self._calculate_epsilon_tilde_vec(lambda_midpoints)
            exp_lambda_diffs = np.diff(elv_np)
            if len(epsilon_tilde_values) != len(exp_lambda_diffs): 
                logging.error(f"Size mismatch: eps_tilde ({len(epsilon_tilde_values)}) vs exp_lambda_diffs ({len(exp_lambda_diffs)})")
                return 1e12
            objective_value = np.sum(epsilon_tilde_values * exp_lambda_diffs)
            return objective_value

        elif self.objective_type == 'legacy_unipc':
            t_points = self.inverse_lambda(lambda_vec_ext_np)
            alpha_vec = self.alpha(t_points)
            sigma_vec = self.sigma(t_points)
            data_err_vec = (sigma_vec**2) / alpha_vec.clip(min=1e-9)
            res = 0.
            c_vec = np.zeros(N_intervals)
            for s in range(N_intervals):
                if s in [0, N_intervals - 1]:
                    n, kp = s, 1
                    if n + 1 >= len(elv_np): continue
                    J_n_kp_0 = elv_np[n+1] - elv_np[n]
                    res += abs(J_n_kp_0 * data_err_vec[n])
                elif s in [1, N_intervals - 2]:
                    n, kp = s-1, 2
                    if n + 1 >= len(hv_np) or n + 1 >= len(data_err_vec): continue
                    J_n_kp_0 = -elv_np[n+1] * self.H1(hv_np[n+1]) / hv_np[n].clip(min=1e-12)
                    J_n_kp_1 = elv_np[n+1] * (self.H1(hv_np[n+1]) + hv_np[n] * self.H0(hv_np[n+1])) / hv_np[n].clip(min=1e-12)
                    if s >= trunc_num_setting:
                        c_vec[n] += data_err_vec[n] * J_n_kp_0
                        c_vec[n+1] += data_err_vec[n+1] * J_n_kp_1
                    else:
                        res += np.sqrt((data_err_vec[n] * J_n_kp_0)**2 + (data_err_vec[n+1] * J_n_kp_1)**2)
                else:
                    n, kp = s-2, 3
                    if n + 2 >= len(hv_np) or n + 2 >= len(data_err_vec) or n+2 >= len(elv_np): continue
                    h_n, h_np1, h_np2 = hv_np[n], hv_np[n+1], hv_np[n+2]
                    J_n_kp_0 = elv_np[n+2] * (self.H2(h_np2) + h_np1*self.H1(h_np2)) / (h_n*(h_n+h_np1)).clip(min=1e-12)
                    J_n_kp_1 = -elv_np[n+2] * (self.H2(h_np2) + (h_n+h_np1)*self.H1(h_np2)) / (h_n*h_np1).clip(min=1e-12)
                    J_n_kp_2 = elv_np[n+2] * (self.H2(h_np2) + (2*h_np1+h_n)*self.H1(h_np2) + h_np1*(h_n+h_np1)*self.H0(h_np2)) / (h_np1*(h_n+h_np1)).clip(min=1e-12)
                    if s >= trunc_num_setting:
                        c_vec[n] += data_err_vec[n] * J_n_kp_0
                        c_vec[n+1] += data_err_vec[n+1] * J_n_kp_1
                        c_vec[n+2] += data_err_vec[n+2] * J_n_kp_2
                    else:
                        res += np.sqrt((data_err_vec[n] * J_n_kp_0)**2 + (data_err_vec[n+1] * J_n_kp_1)**2 + (data_err_vec[n+2] * J_n_kp_2)**2)
            res += sum(abs(c_vec))
            return res

        elif self.objective_type == 'midpoint_unipc':
            aggregated_W = np.zeros(N_intervals)
            for s_loop_idx in range(N_intervals): 
                if N_intervals == 1: n_orig, kp_orig = 0, 1
                elif s_loop_idx == 0: n_orig, kp_orig = 0, 1
                elif s_loop_idx == N_intervals - 1 and N_intervals > 1: n_orig, kp_orig = s_loop_idx, 1
                elif s_loop_idx == 1 or (s_loop_idx == N_intervals - 2 and N_intervals > 2): n_orig, kp_orig = s_loop_idx - 1, 2
                else: n_orig, kp_orig = s_loop_idx - 2, 3
                kp_orig = min(kp_orig, 3, N_intervals - n_orig if N_intervals - n_orig > 0 else 1)
                if n_orig < 0: n_orig = 0
                if kp_orig == 1:
                    if n_orig + 1 >= len(elv_np): continue
                    J0 = elv_np[n_orig+1] - elv_np[n_orig]
                    if n_orig < N_intervals: aggregated_W[n_orig] += J0
                elif kp_orig == 2:
                    if n_orig+1 >= len(hv_np) or n_orig+2 >= len(elv_np): continue
                    h_n, h_np1 = hv_np[n_orig], hv_np[n_orig+1]; denom_J = h_n.clip(min=1e-12)
                    J0 = -elv_np[n_orig+2] * self.H1(h_np1) / denom_J
                    J1 =  elv_np[n_orig+2] * (self.H1(h_np1) + h_n * self.H0(h_np1)) / denom_J
                    if n_orig < N_intervals: aggregated_W[n_orig] += J0
                    if n_orig + 1 < N_intervals: aggregated_W[n_orig+1] += J1
                elif kp_orig == 3:
                    if n_orig+2 >= len(hv_np) or n_orig+3 >= len(elv_np): continue
                    h_n,h_np1,h_np2 = hv_np[n_orig],hv_np[n_orig+1],hv_np[n_orig+2]
                    d0=(h_n*(h_n+h_np1)).clip(min=1e-12); d1=(h_n*h_np1).clip(min=1e-12); d2=(h_np1*(h_n+h_np1)).clip(min=1e-12)
                    J0=elv_np[n_orig+3]*(self.H2(h_np2)+h_np1*self.H1(h_np2))/d0
                    J1=-elv_np[n_orig+3]*(self.H2(h_np2)+(h_n+h_np1)*self.H1(h_np2))/d1
                    J2=elv_np[n_orig+3]*(self.H2(h_np2)+(2*h_np1+h_n)*self.H1(h_np2)+h_np1*(h_n+h_np1)*self.H0(h_np2))/d2
                    if n_orig < N_intervals: aggregated_W[n_orig] += J0
                    if n_orig + 1 < N_intervals: aggregated_W[n_orig+1] += J1
                    if n_orig + 2 < N_intervals: aggregated_W[n_orig+2] += J2
            
            lambda_points_for_f_eval = (lambda_vec_ext_np[:-1] + lambda_vec_ext_np[1:]) / 2.0
            epsilon_tilde_values = self._calculate_epsilon_tilde_vec(lambda_points_for_f_eval)
            if len(epsilon_tilde_values) != len(aggregated_W): 
                logging.error(f"Size mismatch: eps_tilde ({len(epsilon_tilde_values)}) vs W ({len(aggregated_W)})")
                return 1e12
            return np.sum(epsilon_tilde_values * np.abs(aggregated_W))
        else:
            raise NotImplementedError(f"Objective type {self.objective_type} is not implemented.")

    def get_ts_lambdas(self, N_intervals, eps_t_0_val=None, initType='edm', init_rho=7.0, trunc_num_setting_input=None):
        """
        Main method to find the optimal timesteps (t_i) and their lambda transformations (λ_i).
        """
        if N_intervals <= 0:
            raise ValueError("N_intervals must be positive.")
        
        if trunc_num_setting_input is None:
            if N_intervals <= 5: effective_trunc_num_setting = 0
            elif N_intervals <= 7: effective_trunc_num_setting = 3
            else: effective_trunc_num_setting = 0
        else:
            effective_trunc_num_setting = min(trunc_num_setting_input, N_intervals)

        if eps_t_0_val is None:
            eps_t_0_val = self.min_t_val
        
        lambda_eps_val = self.lambda_func(eps_t_0_val).item()
        lambda_T_val = self.lambda_func(self.T_val).item()
        lambda_bounds = sorted([lambda_T_val, lambda_eps_val])
        actual_min_lambda, actual_max_lambda = lambda_bounds[0], lambda_bounds[1]
        
        num_opt_vars = N_intervals - 1
        if num_opt_vars <= 0:
            t_res_np = np.array([self.T_val, eps_t_0_val], dtype=np.float64)
            lambda_res_ext_np = np.array(lambda_bounds, dtype=np.float64)
            return torch.from_numpy(t_res_np).to(self.ns.dtype), torch.from_numpy(lambda_res_ext_np).to(self.ns.dtype)

        # --- Stage 1: Generate Initial Schedule ---
        if initType.startswith('edm'):
            sigma_min_edm = self.ns.edm_sigma(torch.tensor(self.T_val)).cpu().numpy()
            sigma_max_edm = self.ns.edm_sigma(torch.tensor(eps_t_0_val)).cpu().numpy()
            inv_rho = 1.0 / init_rho
            steps_lin = np.linspace(0.0, 1.0, N_intervals + 1)
            sigma_schedule = (sigma_max_edm**inv_rho * (1 - steps_lin) + sigma_min_edm**inv_rho * steps_lin)**init_rho
            t_from_sigma_edm = self.ns.inverse_edm_sigma(torch.from_numpy(sigma_schedule.clip(min=1e-9)).to(self.ns._device, self.ns.dtype))
            lambda_full_init = np.sort(self.lambda_func(t_from_sigma_edm.cpu().numpy()))
        else:
            lambda_full_init = np.linspace(actual_min_lambda, actual_max_lambda, N_intervals + 1)
        
        lambda_opt_vars_init = lambda_full_init[1:-1]
        
        # --- Stage 2: Optimize ---
        lambda_res_opt_part_np = lambda_opt_vars_init
        if not initType.endswith("_origin"):
            opt_result = minimize(
                self._sel_lambdas_obj_calculator,
                lambda_opt_vars_init,
                args=(
                    N_intervals,
                    eps_t_0_val,
                    effective_trunc_num_setting,
                ),
                method='trust-constr',
                options={'maxiter': 80, 'xtol': 1e-6, 'gtol': 1e-6, 'finite_diff_rel_step': 1e-7}
            )
            if opt_result.success:
                lambda_res_opt_part_np = opt_result.x
            else:
                logging.warning(f"Lambda optimization failed for {N_intervals} steps. Using initial values. Message: {opt_result.message}")

        # --- Stage 3: Return Final Schedule ---
        lambda_res_ext_np = np.sort(np.concatenate(([actual_min_lambda], lambda_res_opt_part_np, [actual_max_lambda])))
        t_res_np = np.sort(self.inverse_lambda(lambda_res_ext_np))[::-1]
        return torch.from_numpy(t_res_np.copy()).to(self.ns.dtype), torch.from_numpy(lambda_res_ext_np.copy()).to(self.ns.dtype)


def find_optimal_schedule(
    nfe: int, 
    noise_schedule: NoiseScheduleVP, 
    initial_rho: float, 
    initial_epsilon: float,
    t_max: float,
    return_fitness: bool
):
    """
    This is the "inner loop" function called by the evolutionary search.
    """
    logging.debug(f"Executing UniPC inner optimization for NFE={nfe}, rho={initial_rho:.2f}, epsilon={initial_epsilon:.5f}, t_max={t_max:.3f}")

    original_T = noise_schedule.T
    noise_schedule.T = t_max
    
    # SIMPLIFIED: StepOptim now takes fewer arguments.
    step_optimizer = StepOptim(
        ns=noise_schedule,
        objective_type='midpoint',
        p_fixed_val=2.0
    )

    try:
        optimized_t_steps, optimized_lambda_steps = step_optimizer.get_ts_lambdas(
            N_intervals=nfe,
            initType='edm',
            init_rho=initial_rho,
            eps_t_0_val=initial_epsilon
        )
        
        if not return_fitness:
            noise_schedule.T = original_T
            return optimized_t_steps

        # --- Calculate Fitness Score ---
        final_lambda_for_eval = optimized_lambda_steps.cpu().numpy()[1:-1]
        fitness_score = step_optimizer._sel_lambdas_obj_calculator(
            lambda_vec_opt_part=final_lambda_for_eval,
            N_intervals=nfe,
            eps_t_0_val=initial_epsilon,
            trunc_num_setting=0,
        )
        
        # Add a penalty to the fitness score if timesteps are too close together.
        t_schedule_np = optimized_t_steps.cpu().numpy()
        t_diffs = np.abs(np.diff(np.sort(t_schedule_np)))
        nfe_low, dist_at_low = 4.0, 0.15; nfe_high, dist_at_high = 20.0, 0.01
        slope = (dist_at_high - dist_at_low) / (nfe_high - nfe_low)
        min_t_distance = np.clip(dist_at_low + slope * (nfe - nfe_low), dist_at_high, dist_at_low)
        violations = min_t_distance - t_diffs
        spacing_penalty = 1e9 * np.sum(np.maximum(0, violations)**2)

        noise_schedule.T = original_T
        return optimized_t_steps, fitness_score + spacing_penalty

    except Exception as e:
        noise_schedule.T = original_T
        logging.warning(f"UniPC Inner optimization failed. Reason: {e}")
        return None, float('inf')


def fitness_function_for_unipc_es(params, nfe, noise_schedule):
    """
    A simple wrapper that the `differential_evolution` optimizer can call.
    """
    rho, epsilon, t_max = params[0], params[1], params[2]
    
    _, fitness_score = find_optimal_schedule(
        nfe=nfe, noise_schedule=noise_schedule, 
        initial_rho=rho,
        initial_epsilon=epsilon, 
        t_max=t_max,
        return_fitness=True
    )
    return fitness_score

def run_unipc_search(nfe: int, noise_schedule: NoiseScheduleVP, seed):
    """
    Runs the top-level evolutionary search to find the best hyperparameters.
    """
    logging.info(f"===== Starting UniPC Evolutionary Search for NFE={nfe} =====")
    
    param_bounds = [
        (3.0, 16.0),    # Bounds for rho
        (0.01, 0.03),   # Bounds for epsilon
        (0.96, 1.0)     # Bounds for T_max
    ]
    
    result = differential_evolution(
        fitness_function_for_unipc_es,
        bounds=param_bounds, args=(nfe, noise_schedule),
        maxiter=60, popsize=20, disp=True, tol=0.01, workers=1, updating='deferred',
        seed=seed
    )

    best_rho, best_epsilon, best_t_max = result.x
    min_error_score = result.fun
    
    logging.info(f"✅ UniPC Search Complete! Optimal parameters: rho={best_rho:.4f}, epsilon={best_epsilon:.5f}, T_max={best_t_max:.4f}")
    logging.info(f"   📉 Minimum Theoretical Error Score Found: {min_error_score:.6e}")

    return best_rho, best_epsilon, best_t_max, min_error_score


def find_optimal_schedule_ddim(
    nfe: int, 
    noise_schedule: NoiseScheduleVP, 
    initial_rho: float, 
    initial_epsilon: float,
    t_max: float,
    return_fitness: bool
):
    """
    Core optimization for DDIM.
    """
    logging.debug(f"Executing DDIM inner optimization for NFE={nfe}, rho={initial_rho:.2f}, epsilon={initial_epsilon:.5f}")

    original_T = noise_schedule.T
    noise_schedule.T = t_max

    if nfe <= 1:
        final_t = np.linspace(noise_schedule.T, initial_epsilon, nfe, dtype=np.float64)
        final_t_tensor = torch.from_numpy(final_t).to(noise_schedule.dtype, noise_schedule._device)
        return final_t_tensor, 0 if return_fitness else final_t_tensor

    ddim_intervals = nfe - 1

    # SIMPLIFIED: StepOptim now takes fewer arguments.
    step_optimizer = StepOptim(
        ns=noise_schedule, objective_type='midpoint', p_fixed_val=2.0
    )

    try:
        optimized_t_steps, optimized_lambda_steps = step_optimizer.get_ts_lambdas(
            N_intervals=ddim_intervals, initType='edm', init_rho=initial_rho, eps_t_0_val=initial_epsilon
        )
    except Exception as e:
        logging.warning(f"DDIM Inner optimization failed for rho={initial_rho:.2f}, eps={initial_epsilon:.4f}. Reason: {e}")
        noise_schedule.T = original_T
        return None, float('inf') if return_fitness else None
    
    noise_schedule.T = original_T

    if not return_fitness:
        return optimized_t_steps

    final_lambda_for_eval = optimized_lambda_steps.cpu().numpy()[1:-1]
    
    try:
        fitness_score = step_optimizer._sel_lambdas_obj_calculator(
            lambda_vec_opt_part=final_lambda_for_eval, N_intervals=ddim_intervals, eps_t_0_val=initial_epsilon,
            trunc_num_setting=0
        )
        t_schedule_np = optimized_t_steps.cpu().numpy()
        t_diffs = np.abs(np.diff(np.sort(t_schedule_np)))
        nfe_low, dist_at_low = 4.0, 0.15; nfe_high, dist_at_high = 20.0, 0.01
        slope = (dist_at_high - dist_at_low) / (nfe_high - nfe_low)
        min_t_distance = np.clip(dist_at_low + slope * (nfe - nfe_low), dist_at_high, dist_at_low)
        violations = min_t_distance - t_diffs
        spacing_penalty = 1e9 * np.sum(np.maximum(0, violations)**2)
        
        return optimized_t_steps, fitness_score + spacing_penalty
    except Exception as e:
        logging.error(f"Error calculating fitness for optimized DDIM schedule: {e}")
        return None, float('inf')

def fitness_function_for_ddim_es(params, nfe, noise_schedule):
    """ Fitness function wrapper for the DDIM evolutionary search. """
    rho, epsilon, t_max = params[0], params[1], params[2]
    _, fitness_score = find_optimal_schedule_ddim(
        nfe=nfe, noise_schedule=noise_schedule, 
        initial_rho=rho,
        initial_epsilon=epsilon, t_max=t_max, 
        return_fitness=True
    )
    return fitness_score

def run_ddim_search(nfe: int, noise_schedule: NoiseScheduleVP, seed):
    """ Runs the top-level evolutionary search for the DDIM sampler. """
    logging.info(f"===== Starting DDIM Evolutionary Search for NFE={nfe} =====")

    param_bounds = [
        (3.0, 16.0),    # rho
        (0.01, 0.03),   # epsilon
        (0.96, 1.0)     # t_max
    ]

    result = differential_evolution(
        fitness_function_for_ddim_es,
        bounds=param_bounds, args=(nfe, noise_schedule),
        maxiter=60, popsize=20, disp=True, tol=0.01, workers=1, updating='deferred',
        seed=seed
    )

    best_rho, best_epsilon, best_t_max = result.x
    min_error_score = result.fun
    
    logging.info(f"✅ DDIM Search Complete! Optimal parameters: rho={best_rho:.4f}, epsilon={best_epsilon:.5f}, t_max={best_t_max:.4f}")
    logging.info(f"   📉 Minimum Theoretical Error Score Found: {min_error_score:.6e}")

    return best_rho, best_epsilon, best_t_max


def set_global_seed(seed_value='None'):
    """Sets the seed for all major sources of randomness for reproducibility."""
    if seed_value is None:
        return
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logging.info(f"Global random seed set to {seed_value}")
    

if __name__ == '__main__':

    seed = 66
    set_global_seed(seed)
    
    # --- Setup the Noise Schedule ---
    model_total_timesteps = 1000
    betas_ddpm = torch.linspace(0.00085, 0.0120, model_total_timesteps, dtype=torch.float64) 
    alphas_cumprod_sd = torch.cumprod(1.0 - betas_ddpm, dim=0)
    ns_instance = NoiseScheduleVP(
        schedule_name='discrete', 
        alphas_cumprod=alphas_cumprod_sd, 
        dtype=torch.float64
    )
    
    NFE_TO_SOLVE = 4


    # --- 1. UniPC ---
    logging.info(f"\n{'='*25}\n===== 1. Optimizing Schedule for UniPC =====\n{'='*25}")
    best_rho_unipc, best_epsilon_unipc, best_t_max_unipc, error_score_unipc = run_unipc_search(nfe=NFE_TO_SOLVE, noise_schedule=ns_instance, seed=seed)
    
    final_schedule_unipc = find_optimal_schedule(
        nfe=NFE_TO_SOLVE,
        noise_schedule=ns_instance,
        initial_rho=best_rho_unipc,
        initial_epsilon=best_epsilon_unipc,
        t_max=best_t_max_unipc,
        return_fitness=False
    )
    
    logging.info(f"\n--- Final Results (UniPC) ---")
    logging.info(f"Optimal Hyperparameters: rho={best_rho_unipc:.4f}, epsilon={best_epsilon_unipc:.5f}, t_max={best_t_max_unipc:.4f}")
    logging.info(f"📈 Final Theoretical Error Score: {error_score_unipc:.6e}")
    if final_schedule_unipc is not None:
        logging.info(f"Generated Optimal UniPC Schedule ({len(final_schedule_unipc)} values, from T to ε):")
        print(np.array2string(final_schedule_unipc.cpu().numpy(), formatter={'float_kind':lambda x: "%.8f" % x}))
    else:
        logging.error("Failed to find an optimal schedule for UniPC.")

    # --- 2. DDIM ---
    logging.info(f"\n{'='*25}\n===== 2. Optimizing Schedule for DDIM =====\n{'='*25}")
    best_rho_ddim, best_epsilon_ddim, best_t_max_ddim = run_ddim_search(
        nfe=NFE_TO_SOLVE, noise_schedule=ns_instance, seed=seed
    )

    final_schedule_ddim_t = find_optimal_schedule_ddim(
        nfe=NFE_TO_SOLVE,
        noise_schedule=ns_instance,
        initial_rho=best_rho_ddim,
        initial_epsilon=best_epsilon_ddim,
        t_max=best_t_max_ddim,
        return_fitness=False
    )

    logging.info(f"\n--- Final Results (DDIM) ---")
    logging.info(f"Optimal Hyperparameters: rho={best_rho_ddim:.4f}, epsilon={best_epsilon_ddim:.5f}, t_max={best_t_max_ddim:.4f}")

    if final_schedule_ddim_t is not None:
        logging.info(f"Generated Optimal DDIM t-Schedule ({len(final_schedule_ddim_t)} points):")
        print(np.array2string(final_schedule_ddim_t.cpu().numpy(), formatter={'float_kind':lambda x: "%.8f" % x}))
        
        n_model_steps = ns_instance.num_trained_timesteps
        ddim_timesteps_indices = (final_schedule_ddim_t.cpu().to(torch.float64) * n_model_steps).round().long() - 1
        ddim_timesteps_indices = torch.clamp(ddim_timesteps_indices, min=0)
        ddim_timesteps = ddim_timesteps_indices.cpu().numpy()

        logging.info(f"\nConverted Discrete DDIM Timesteps ({len(ddim_timesteps)} values):")
        print(ddim_timesteps)
    else:
        logging.error("Failed to find an optimal schedule for DDIM.")