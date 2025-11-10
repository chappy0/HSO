"""SAMPLING ONLY."""
import torch

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

MODEL_TYPES = {
    "eps": "noise",
    "v": "v"
}


# In file: sampler.py

# ===================================================================
# 请用下面的完整类，替换您脚本中现有的 DPMSolverSampler 类
# ===================================================================
class DPMSolverSampler(object):
    def __init__(self, model, device=torch.device("cuda"), **kwargs):
        super().__init__()
        self.model = model
        self.device = device
        # 确保在初始化时就将alphas_cumprod转换为正确的设备和类型
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def get_timesteps(self, steps):
        # 这个辅助函数用于获取教师轨迹的时间步，以便在 generate_teacher_trajectory 中使用
        # 确保它与 DPM_Solver 中的 get_time_steps 逻辑一致
        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
        dpm_solver = DPM_Solver(lambda x, t: self.model.apply_model(x, t, None), ns)
        t_T = ns.T
        t_0 = 1. / ns.total_N
        return dpm_solver.get_time_steps(skip_type='time_uniform', t_T=t_T, t_0=t_0, N=steps, device=self.device)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               return_intermediates=False, # <-- 确认接收此参数
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                if isinstance(ctmp, torch.Tensor) and ctmp.shape[0] != batch_size:
                    print(f"Warning: Got {ctmp.shape[0]} conditionings but batch-size is {batch_size}")
            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {ctmp.shape[0]} conditionings but batch-size is {batch_size}")
            elif isinstance(conditioning, torch.Tensor) and conditioning.shape[0] != batch_size:
                print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        C, H, W = shape
        size = (batch_size, C, H, W)
        device = self.device # 使用类属性中保存的device

        if verbose:
            print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T.to(device)

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            ns,
            model_type=MODEL_TYPES.get(self.model.parameterization, "eps"), # 使用.get增加鲁棒性
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        # 推荐使用 DPM-Solver++ (predict_x0=True) 和 multistep
        dpm_solver = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False)
        
        # --- 核心修复：调用底层的sample，并正确处理返回值 ---
        # 确保将 return_intermediates 参数传递下去
        results = dpm_solver.sample(
            img, 
            steps=S, 
            skip_type="time_uniform", 
            method="multistep", 
            order=2, # 对于高质量教师轨迹，2阶或3阶都可以
            lower_order_final=True,
            return_intermediates=return_intermediates # <-- 传递关键参数
        )

        if return_intermediates:
            # 如果请求了中间结果，dpm_solver.sample会返回一个元组 (x, intermediates)
            if isinstance(results, tuple) and len(results) == 2:
                x_final, intermediates = results
                return x_final.to(device), intermediates
            else:
                # 如果返回的不是预期的元组，进行错误处理
                raise TypeError(f"Expected a tuple of (samples, intermediates) from dpm_solver.sample, but got {type(results)}")
        else:
            # 否则，只返回最终结果
            x_final = results
            return x_final.to(device), None

# class DPMSolverSampler(object):
#     def __init__(self, model, device=torch.device("cuda"), **kwargs):
#         super().__init__()
#         self.model = model
#         self.device = device
#         to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
#         self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

#     def register_buffer(self, name, attr):
#         if type(attr) == torch.Tensor:
#             if attr.device != self.device:
#                 attr = attr.to(self.device)
#         setattr(self, name, attr)

#     @torch.no_grad()
#     def sample(self,
#                S,
#                batch_size,
#                shape,
#                conditioning=None,
#                callback=None,
#                normals_sequence=None,
#                img_callback=None,
#                quantize_x0=False,
#                eta=0.,
#                mask=None,
#                x0=None,
#                temperature=1.,
#                noise_dropout=0.,
#                score_corrector=None,
#                corrector_kwargs=None,
#                verbose=True,
#                x_T=None,
#                log_every_t=100,
#                unconditional_guidance_scale=1.,
#                unconditional_conditioning=None,
#                # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
#                **kwargs
#                ):
#         if conditioning is not None:
#             if isinstance(conditioning, dict):
#                 ctmp = conditioning[list(conditioning.keys())[0]]
#                 while isinstance(ctmp, list): ctmp = ctmp[0]
#                 if isinstance(ctmp, torch.Tensor):
#                     cbs = ctmp.shape[0]
#                     if cbs != batch_size:
#                         print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
#             elif isinstance(conditioning, list):
#                 for ctmp in conditioning:
#                     if ctmp.shape[0] != batch_size:
#                         print(f"Warning: Got {ctmp.shape[0]} conditionings but batch-size is {batch_size}")
#             else:
#                 if isinstance(conditioning, torch.Tensor):
#                     if conditioning.shape[0] != batch_size:
#                         print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

#         # sampling
#         C, H, W = shape
#         size = (batch_size, C, H, W)

#         print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

#         device = self.model.betas.device
#         if x_T is None:
#             img = torch.randn(size, device=device)
#         else:
#             img = x_T

#         ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

#         model_fn = model_wrapper(
#             lambda x, t, c: self.model.apply_model(x, t, c),
#             ns,
#             model_type=MODEL_TYPES[self.model.parameterization],
#             guidance_type="classifier-free",
#             condition=conditioning,
#             unconditional_condition=unconditional_conditioning,
#             guidance_scale=unconditional_guidance_scale,
#         )

#         dpm_solver = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False)
#         x = dpm_solver.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=2,
#                               lower_order_final=True)

#         return x.to(device), None
