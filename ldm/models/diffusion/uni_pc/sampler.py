"""SAMPLING ONLY."""

import torch

from .uni_pc import NoiseScheduleVP, model_wrapper, UniPC

class UniPCSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

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
               custom_timesteps=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               return_intermediates=False,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        if getattr(self.model, 'parameterization', 'eps') == 'v':
            model_type = 'v'
            print("INFO: UniPCSampler detected a v-prediction model.")
        else:
            model_type = 'noise'
            print("INFO: UniPCSampler using standard epsilon-prediction model.")

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            ns,
            # model_type="noise",
            model_type=model_type,
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        uni_pc = UniPC(model_fn, ns, predict_x0=True, thresholding=False)

       # --- 核心修正 2: 调用修改后的uni_pc.sample并处理其返回值 ---
        # 我们将 return_intermediates 参数传递下去
        result = uni_pc.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=3, 
                               lower_order_final=True, custom_timesteps=custom_timesteps,
                               return_intermediates=return_intermediates)

        if return_intermediates:
            # 如果请求返回轨迹，那么result是一个元组 (final_x, intermediates_dict)
            x, intermediates = result
            # 将CPU上的潜变量列表转换为GPU张量
            intermediates['x_inter'] = torch.stack([t.to(device) for t in intermediates['x_inter']])
            return x.to(device), intermediates
        else:
            # 否则，result就是最终的x
            x = result
            return x.to(device), None
        
        # # 如果提供了自定义时间步，则使用它
        # if custom_timesteps is not None:
        #     print(f'use pass custom_timesteps:{custom_timesteps}')
        #     # 确保时间步是按降序排列的
        #     custom_timesteps = sorted(custom_timesteps, reverse=True)
        #     # 将自定义时间步传递给 UniPC 的 sample 方法
        #     x = uni_pc.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=3, lower_order_final=True, custom_timesteps=custom_timesteps)
        # else:
        #     # 使用默认的时间步
        #     x = uni_pc.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=3, lower_order_final=True)

        # return x.to(device), None