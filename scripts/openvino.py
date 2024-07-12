# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: AGPL-3.0
from contextlib import closing
import cv2
import os
import torch
import time
import hashlib
import functools
import gradio as gr
from modules.ui import plaintext_to_html
import numpy as np
import sys
from typing import List


import modules
import modules.paths as paths
import modules.scripts as scripts

from modules import processing, sd_unet

from modules import images, devices, extra_networks, masking, shared, sd_models_config, prompt_parser
from modules.processing import (
    StableDiffusionProcessing, Processed, apply_overlay, apply_color_correction,
    get_fixed_seed, create_infotext, setup_color_correction
)
from modules.sd_models import CheckpointInfo, get_checkpoint_state_dict
from modules.shared import opts, state
from modules.ui_common import create_refresh_button
from modules.timer import Timer

from PIL import Image, ImageOps
from types import MappingProxyType
from typing import Optional

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.frontend.pytorch.torchdynamo import backend # noqa: F401
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.runtime import Core, Type, PartialShape, serialize

from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten

from hashlib import sha256

from diffusers import (
    LCMScheduler,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    StableDiffusionLatentUpscalePipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    AutoencoderKL,
)

#ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##hack eval_frame.py for windows support, could be removed after official windows support from pytorch
def check_if_dynamo_supported():
    import sys
    # Skip checking for Windows support for the OpenVINO backend
    if sys.version_info >= (3, 12):
        raise RuntimeError("Python 3.12+ not yet supported for torch.compile")

torch._dynamo.eval_frame.check_if_dynamo_supported = check_if_dynamo_supported




## hack for pytorch
def BUILD_MAP_UNPACK(self, inst):
        items = self.popn(inst.argval)
        # ensure everything is a dict
        items = [BuiltinVariable(dict).call_function(self, [x], {}) for x in items] # noqa: F821
        result = dict()
        for x in items:
            assert isinstance(x, ConstDictVariable) # noqa: F821
        result.update(x.items)
        self.push(
            ConstDictVariable( # noqa: F821
                result,
                dict,
                mutable_local=MutableLocal(), # noqa: F821
                **VariableTracker.propagate(items), # noqa: F821
            )
        )
tmp_torch = sys.modules["torch"]
tmp_torch.BUILD_MAP_UNPACK_WITH_CALL = BUILD_MAP_UNPACK

class ModelState:
    def __init__(self):
        self.recompile = 1
        self.device = "CPU"
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.mode = 0
        self.partition_id = 0
        self.model_hash = ""
        self.control_models = []
        self.is_sdxl = False
        self.cn_model = "None"
        self.lora_model = "None"
        self.vae_ckpt = "None"
        self.refiner_ckpt = "None"


model_state = ModelState()

DEFAULT_OPENVINO_PYTHON_CONFIG = MappingProxyType(
    {
        "use_python_fusion_cache": True,
        "allow_single_op_fusion": True,
    },
)

compiled_cache = {}
max_openvino_partitions = 0
partitioned_modules = {}



def openvino_clear_caches():
    global partitioned_modules
    global compiled_cache

    compiled_cache.clear()
    partitioned_modules.clear()

def sd_diffusers_model(self):
    import modules.sd_models
    return modules.sd_models.model_data.get_sd_model()

def cond_stage_key(self):
    return None

shared.sd_diffusers_model = sd_diffusers_model
shared.sd_refiner_model = None


#sdxl invisible-watermark pixel artifact workaround
class NoWatermark:
    def apply_watermark(self, img):
        return img


def get_diffusers_upscaler(upscaler: str):
    torch._dynamo.reset()
    openvino_clear_caches()
    model_name = "stabilityai/sd-x2-latent-upscaler"
    print("OpenVINO Script: loading upscaling model: " + model_name)
    sd_model = StableDiffusionLatentUpscalePipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    sd_model.safety_checker = None
    sd_model.cond_stage_key = functools.partial(cond_stage_key, shared.sd_model)
    sd_model.unet = torch.compile(sd_model.unet, backend="openvino")
    sd_model.vae.decode = torch.compile(sd_model.vae.decode, backend="openvino")
    shared.sd_diffusers_model = sd_model
    del sd_model

    return shared.sd_diffusers_model


def get_diffusers_sd_model(model_config, vae_ckpt, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_ckpt, refiner_frac):
    if (model_state.recompile == 1):
        model_state.partition_id = 0
        torch._dynamo.reset()
        openvino_clear_caches()
        curr_dir_path = os.getcwd()
        checkpoint_name = shared.opts.sd_model_checkpoint.split(" ")[0]
        checkpoint_path = os.path.join(curr_dir_path, 'models', 'Stable-diffusion', checkpoint_name)
        checkpoint_info = CheckpointInfo(checkpoint_path)
        timer = Timer()
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
        checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
        print("OpenVINO Script:  created model from config : " + checkpoint_config)
        if(is_xl_ckpt):
            if model_config != "None":
                local_config_file = os.path.join(curr_dir_path, 'configs', model_config)
                sd_model = StableDiffusionXLPipeline.from_single_file(checkpoint_path, original_config_file=local_config_file, use_safetensors=True, add_watermark=False, variant="fp32", dtype=torch.float32)
            else:
                sd_model = StableDiffusionXLPipeline.from_single_file(checkpoint_path, original_config_file=checkpoint_config, use_safetensors=True, add_watermark=False, variant="fp32", dtype=torch.float32)
            if (mode == 1):
                sd_model = StableDiffusionXLImg2ImgPipeline(**sd_model.components)
            elif (mode == 2):
                sd_model = StableDiffusionXLInpaintPipeline(**sd_model.components)
            elif (mode == 3):
                if (len(model_state.control_models) > 1):
                    controlnet = []
                    for cn_model in model_state.control_models:
                        cn_model_dir_path = os.path.join(curr_dir_path,'models', 'ControlNet')
                        cn_model_path = os.path.join(cn_model_dir_path, cn_model)
                        if os.path.isfile(cn_model_path + '.pt'):
                            cn_model_path = cn_model_path + '.pt'
                        elif os.path.isfile(cn_model_path + '.safetensors'):
                            cn_model_path = cn_model_path + '.safetensors'
                        elif os.path.isfile(cn_model_path + '.pth'):
                            cn_model_path = cn_model_path + '.pth'
                        controlnet.append(ControlNetModel.from_single_file(cn_model_path, local_files_only=True))
                else:
                    cn_model_dir_path = os.path.join(curr_dir_path,'models', 'ControlNet')
                    cn_model_path = os.path.join(cn_model_dir_path, model_state.control_models[0])
                    if os.path.isfile(cn_model_path + '.pt'):
                        cn_model_path = cn_model_path + '.pt'
                    elif os.path.isfile(cn_model_path + '.safetensors'):
                        cn_model_path = cn_model_path + '.safetensors'
                    elif os.path.isfile(cn_model_path + '.pth'):
                        cn_model_path = cn_model_path + '.pth'
                    controlnet = ControlNetModel.from_single_file(cn_model_path, local_files_only=True)
                sd_model = StableDiffusionControlNetPipeline(**sd_model.components, controlnet=controlnet)
                sd_model.controlnet = torch.compile(sd_model.controlnet, backend="openvino_fx_ext")
        else:
            if model_config != "None":
                local_config_file = os.path.join(curr_dir_path, 'configs', model_config)
                sd_model = StableDiffusionPipeline.from_single_file(checkpoint_path, original_config_file=local_config_file, use_safetensors=True, variant="fp32", dtype=torch.float32)
            else:
                sd_model = StableDiffusionPipeline.from_single_file(checkpoint_path, original_config_file=checkpoint_config, use_safetensors=True, variant="fp32", dtype=torch.float32)

            if (mode == 1):
                sd_model = StableDiffusionImg2ImgPipeline(**sd_model.components)
            elif (mode == 2):
                sd_model = StableDiffusionInpaintPipeline(**sd_model.components)
            elif (mode == 3):
                if (len(model_state.control_models) > 1):
                    controlnet = []
                    for cn_model in model_state.control_models:
                        cn_model_dir_path = os.path.join(curr_dir_path,'models', 'ControlNet')
                        cn_model_path = os.path.join(cn_model_dir_path, cn_model)
                        if os.path.isfile(cn_model_path + '.pt'):
                            cn_model_path = cn_model_path + '.pt'
                        elif os.path.isfile(cn_model_path + '.safetensors'):
                            cn_model_path = cn_model_path + '.safetensors'
                        elif os.path.isfile(cn_model_path + '.pth'):
                            cn_model_path = cn_model_path + '.pth'
                        controlnet.append(ControlNetModel.from_single_file(cn_model_path, local_files_only=True))
                else:
                    cn_model_dir_path = os.path.join(curr_dir_path,'models', 'ControlNet')
                    cn_model_path = os.path.join(cn_model_dir_path, model_state.control_models[0])
                    if os.path.isfile(cn_model_path + '.pt'):
                        cn_model_path = cn_model_path + '.pt'
                    elif os.path.isfile(cn_model_path + '.safetensors'):
                        cn_model_path = cn_model_path + '.safetensors'
                    elif os.path.isfile(cn_model_path + '.pth'):
                        cn_model_path = cn_model_path + '.pth'
                    controlnet = ControlNetModel.from_single_file(cn_model_path, local_files_only=True)
                sd_model = StableDiffusionControlNetPipeline(**sd_model.components, controlnet=controlnet)
                sd_model.controlnet = torch.compile(sd_model.controlnet, backend="openvino_fx_ext")

        #load lora


        if ('lora' in modules.extra_networks.extra_network_registry):
            import lora
            if lora.loaded_loras:
                lora_model = lora.loaded_loras[0]
                sd_model.load_lora_weights(os.path.join(os.getcwd(), "models", "Lora"), weight_name=lora_model.name + ".safetensors", low_cpu_mem_usage=True)
        sd_model.watermark = NoWatermark()
        sd_model.sd_checkpoint_info = checkpoint_info
        sd_model.sd_model_hash = checkpoint_info.calculate_shorthash()
        sd_model.safety_checker = None
        sd_model.cond_stage_key = functools.partial(cond_stage_key, shared.sd_model)
        sd_model.scheduler = set_scheduler(sd_model, sampler_name)
        sd_model.unet = torch.compile(sd_model.unet, backend="openvino_fx_ext")
        ## VAE
        if vae_ckpt == "Disable-VAE-Acceleration":
            sd_model.vae.decode = sd_model.vae.decode
        elif vae_ckpt == "None":
            sd_model.vae.decode = torch.compile(sd_model.vae.decode, backend="openvino_fx_ext")
        else:
            vae_path = os.path.join(curr_dir_path, 'models', 'VAE', vae_ckpt)
            print("OpenVINO Script:  loading vae from : " + vae_path)
            sd_model.vae = AutoencoderKL.from_single_file(vae_path, local_files_only=True)
            sd_model.vae.decode = torch.compile(sd_model.vae.decode, backend="openvino_fx_ext")
        shared.sd_diffusers_model = sd_model
        del sd_model
    return shared.sd_diffusers_model


##get refiner model
def get_diffusers_sd_refiner_model(model_config, vae_ckpt, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_ckpt, refiner_frac):
    if (model_state.recompile == 1):
        curr_dir_path = os.getcwd()
        if refiner_ckpt != "None":
            refiner_checkpoint_path= os.path.join(curr_dir_path, 'models', 'Stable-diffusion', refiner_ckpt)
            refiner_checkpoint_info = CheckpointInfo(refiner_checkpoint_path)
            refiner_model = StableDiffusionXLImg2ImgPipeline.from_single_file(refiner_checkpoint_path, use_safetensors=True, torch_dtype=torch.float32)
            refiner_model.watermark = NoWatermark()
            refiner_model.sd_checkpoint_info = refiner_checkpoint_info
            refiner_model.sd_model_hash = refiner_checkpoint_info.calculate_shorthash()
            refiner_model.unet = torch.compile(refiner_model.unet,  backend="openvino_fx_ext")
            ## VAE
            if vae_ckpt == "Disable-VAE-Acceleration":
                refiner_model.vae.decode = refiner_model.vae.decode
            elif vae_ckpt == "None":
                refiner_model.vae.decode = torch.compile(refiner_model.vae.decode, backend="openvino_fx_ext")
            else:
                vae_path = os.path.join(curr_dir_path, 'models', 'VAE', vae_ckpt)
                print("OpenVINO Script:  loading vae from : " + vae_path)
                refiner_model.vae = AutoencoderKL.from_single_file(vae_path, local_files_only=True)
                refiner_model.vae.decode = torch.compile(refiner_model.vae.decode, backend="openvino_fx_ext")
            shared.sd_refiner_model = refiner_model
        del refiner_model
    return shared.sd_refiner_model



def on_change(mode):
    return gr.update(visible=mode)

from modules.processing import StableDiffusionProcessing

POSITIVE_MARK_TOKEN = 1024
NEGATIVE_MARK_TOKEN = - POSITIVE_MARK_TOKEN
MARK_EPS = 1e-3


def prompt_context_is_marked(x):
    t = x[..., 0, :]
    m = torch.abs(t) - POSITIVE_MARK_TOKEN
    m = torch.mean(torch.abs(m)).detach().cpu().float().numpy()
    return float(m) < MARK_EPS


def mark_prompt_context(x, positive):
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = mark_prompt_context(x[i], positive)
        return x
    if isinstance(x, MulticondLearnedConditioning):
        x.batch = mark_prompt_context(x.batch, positive)
        return x
    if isinstance(x, ComposableScheduledPromptConditioning):
        x.schedules = mark_prompt_context(x.schedules, positive)
        return x
    if isinstance(x, ScheduledPromptConditioning):
        if isinstance(x.cond, dict):
            cond = x.cond['crossattn']
            if prompt_context_is_marked(cond):
                return x
            mark = POSITIVE_MARK_TOKEN if positive else NEGATIVE_MARK_TOKEN
            cond = torch.cat([torch.zeros_like(cond)[:1] + mark, cond], dim=0)
            return ScheduledPromptConditioning(end_at_step=x.end_at_step, cond=dict(crossattn=cond, vector=x.cond['vector']))
        else:
            cond = x.cond
            if prompt_context_is_marked(cond):
                return x
            mark = POSITIVE_MARK_TOKEN if positive else NEGATIVE_MARK_TOKEN
            cond = torch.cat([torch.zeros_like(cond)[:1] + mark, cond], dim=0)
            return ScheduledPromptConditioning(end_at_step=x.end_at_step, cond=cond)
    return x


disable_controlnet_prompt_warning = True
# You can disable this warning using disable_controlnet_prompt_warning.


def unmark_prompt_context(x):
    if not prompt_context_is_marked(x):
        # ControlNet must know whether a prompt is conditional prompt (positive prompt) or unconditional conditioning prompt (negative prompt).
        # You can use the hook.py's `mark_prompt_context` to mark the prompts that will be seen by ControlNet.
        # Let us say XXX is a MulticondLearnedConditioning or a ComposableScheduledPromptConditioning or a ScheduledPromptConditioning or a list of these components,
        # if XXX is a positive prompt, you should call mark_prompt_context(XXX, positive=True)
        # if XXX is a negative prompt, you should call mark_prompt_context(XXX, positive=False)
        # After you mark the prompts, the ControlNet will know which prompt is cond/uncond and works as expected.
        # After you mark the prompts, the mismatch errors will disappear.
        if not disable_controlnet_prompt_warning:
            logger.warning('ControlNet Error: Failed to detect whether an instance is cond or uncond!')
            logger.warning('ControlNet Error: This is mainly because other extension(s) blocked A1111\'s \"process.sample()\" and deleted ControlNet\'s sample function.')
            logger.warning('ControlNet Error: ControlNet will shift to a backup backend but the results will be worse than expectation.')
            logger.warning('Solution (For extension developers): Take a look at ControlNet\' hook.py '
                  'UnetHook.hook.process_sample and manually call mark_prompt_context to mark cond/uncond prompts.')
        mark_batch = torch.ones(size=(x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device)
        context = x
        return mark_batch, [], [], context
    mark = x[:, 0, :]
    context = x[:, 1:, :]
    mark = torch.mean(torch.abs(mark - NEGATIVE_MARK_TOKEN), dim=1)
    mark = (mark > MARK_EPS).float()
    mark_batch = mark[:, None, None, None].to(x.dtype).to(x.device)

    mark = mark.detach().cpu().numpy().tolist()
    uc_indices = [i for i, item in enumerate(mark) if item < 0.5]
    c_indices = [i for i, item in enumerate(mark) if not item < 0.5]

    StableDiffusionProcessing.cached_c = [None, None]
    StableDiffusionProcessing.cached_uc = [None, None]

    return mark_batch, uc_indices, c_indices, context


OV_df_unet = None
class OVUnetOption(sd_unet.SdUnetOption):
    def __init__(self, name: str):
        self.label = f"[OV] {name}"
        self.model_name = name
        self.configs = None

    def create_unet(self):
        return OVUnet(self.model_name)


from diffusers import DiffusionPipeline 
class OVUnet(sd_unet.SdUnet):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__()
        self.process = None

        self.model_name = model_name
        #self.configs = configs

        self.loaded_config = None

        self.engine_vram_req = 0
        self.refitted_keys = set()

        self.engine = None
        self.controlnet = None
        self.control_images = None
        self.canny_images = []
        self.vae = None
        self.current_uc_indices = None
        self.current_c_indices = None
    
 

    
    def forward_OV(self, x, timesteps=None, context=None, y=None, **kwargs):
        print('OV hooked controlnet unet forward called!')
        input()

    def forward_original(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        if "y" in kwargs:
            print('sd-xl')
            # todo: add sdxl support
            #feed_dict["y"] = kwargs["y"].float()

        
        print('OVUnet forward called!!! begin calling unet forward!!!')
        print('x.shape: ',x.shape,'timesteps.shape: ',timesteps.shape,'context.shape: ',context.shape, 'args: ',args, 'kwargs: ',kwargs)
        
        print('extra_generation_params: ',self.process.extra_generation_params)
        

        
        
        out = self.engine(x, timesteps, context, *args, **kwargs).sample
        

        return out
    
    
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        
        
        down_block_res_samples, mid_block_res_sample = None, None 
        print('timesteps:', timesteps)
        print('x: min(x), max(x)', torch.min(x), torch.max(x))
        import pdb; pdb.set_trace()

        if self.process.extra_generation_params:
            print('controlnet detected')
            cond_mark, self.current_uc_indices, self.current_c_indices, context = unmark_prompt_context(context)

            
            
            print('check processed image shape')

            control_model = OV_df_unet.controlnet
            image = OV_df_unet.canny_images[0]
            print('image min and max:', torch.min(image), torch.max(image))
            print('x min and max:', torch.min(x), torch.max(x)) #  tensor(-13.3997) tensor(11.1092)
            print('context min and max:', torch.min(context), torch.max(context),'shape:', context.shape ) #  tensor(-1024.) tensor(1024.)  [2, 78, 768])
            #control_model = torch.compile(OV_df_unet.controlnet, backend = 'openvino')  # ControlNetModel.from_single_file('./extensions/sd-webui-controlnet/models/control_v11p_sd15_canny_fp16.safetensors', local_files_only=True)
            down_block_res_samples, mid_block_res_sample = control_model(
                x,
                timesteps,
                encoder_hidden_states=context,
                controlnet_cond=OV_df_unet.canny_images[0],
                conditioning_scale=1.0,
                guess_mode = False,
                return_dict=False,
                #y=y
            )
            ''''
            [diffusers ref]
            image min and max:  tensor(0.) tensor(1.)
            control model input  min and max:  tensor(-3.3987) tensor(2.9278)
            controlnet_prompt_embeds, min and max: tensor(-28.0912) tensor(33.0632)
            cond_scale: 1.0
            guess_mode: False
            min_block sample min and max: tensor(-18.6162) tensor(23.7768)
            '''
            print('mid_block_res_sample min and max:', torch.min(mid_block_res_sample), torch.max(mid_block_res_sample))
            print([d.shape for d in down_block_res_samples], mid_block_res_sample.shape)
            print('check diffusers  controlnet output')
        else:
            print('no controlnet detected')
            #import pdb; pdb.set_trace()

        
        #OV_df_unet.engine = torch.compile(OV_df_unet.engine, backend = 'openvino')
        noise_pred = OV_df_unet.engine(
                x,
                timesteps,
                context,
                #timestep_cond=timestep_cond,
                #cross_attention_kwargs=({}),
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                #added_cond_kwargs=({}), #added_cond_kwargs,
                #return_dict=False,
                ).sample

        
        return noise_pred
    
    

    def apply_loras(self, refit_dict: dict):
        if not self.refitted_keys.issubset(set(refit_dict.keys())):
            # Need to ensure that weights that have been modified before and are not present anymore are reset.
            self.refitted_keys = set()
            self.switch_engine()

        self.engine.refit_from_dict(refit_dict, is_fp16=True)
        self.refitted_keys = set(refit_dict.keys())

    def switch_engine(self):
        self.loaded_config = self.configs[self.profile_idx]
        self.engine.reset(os.path.join(OV_MODEL_DIR, self.loaded_config["filepath"]))
        self.activate(p)
        
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):  
        from diffusers.image_processor import VaeImageProcessor
        self.vae_scale_factor = 2 ** (len(OV_df_unet.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def activate(self, p):
        #self.loaded_config = self.configs[self.profile_idx]
        # model state
        #### controlnet ####
        print(p.extra_generation_params)

        
        print('loading OV unet model')
        checkpoint_name = shared.opts.sd_model_checkpoint.split(" ")[0]
        checkpoint_path = os.path.join(scripts.basedir(), 'models', 'Stable-diffusion', checkpoint_name)
        checkpoint_info = CheckpointInfo(checkpoint_path)
        timer = Timer()
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
        checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
        print("OpenVINO Extension:  created model from config : " + checkpoint_config)
        pipe = StableDiffusionPipeline.from_single_file(checkpoint_path, original_config_file=checkpoint_config, use_safetensors=True, variant="fp32", dtype=torch.float32)
        OV_df_unet.engine = pipe.unet 
        OV_df_unet.vae = pipe.vae 
        #OV_df_unet.engine = torch.compile(OV_df_unet.engine, backend="openvino")
        print('OpenVINO Extension: loaded unet model')
            
        input('check p.extra generation params')
        if p.extra_generation_params:
            input('controlnet detected')
    
            cn_model="None"
            control_models = []
            control_images = []
            print("p.extra_generation_params", p.extra_generation_params)
            from internal_controlnet.external_code import ControlNetUnit
            for param in p.script_args: 
                if isinstance(param, ControlNetUnit): 
                    if param.enabled == False: continue 
                    print('controlnet detected')
                    print(param)
                    control_models.append(param.model.split(' ')[0])
                    print('param.image:', param.image['image'])
                    input('check img')
                    control_images.append(param.image['image'])
            
            model_state.control_models = control_models
            OV_df_unet.control_images = control_images
            import cv2
            import numpy as np
            from PIL import Image
            for image in OV_df_unet.control_images:

                image = np.array(image)
                image = cv2.Canny(image, 100, 200)
                image = image[:, :, None]
                image = np.concatenate([image, image, image], axis=2)
                canny_image = Image.fromarray(image)
                OV_df_unet.canny_images.append(canny_image)
            
            
            print("model_state.control_models:", model_state.control_models)
            
            
            
            input('begin loading controlnet model(s)')
            
            if (len(model_state.control_models) > 1):
                controlnet = []
                for cn_model in model_state.control_models:
                    cn_model_dir_path = os.path.join(scripts.basedir(),'extensions','sd-webui-controlnet','models')
                    cn_model_path = os.path.join(cn_model_dir_path, cn_model)
                    if os.path.isfile(cn_model_path + '.pt'):
                        cn_model_path = cn_model_path + '.pt'
                    elif os.path.isfile(cn_model_path + '.safetensors'):
                        cn_model_path = cn_model_path + '.safetensors'
                    elif os.path.isfile(cn_model_path + '.pth'):
                        cn_model_path = cn_model_path + '.pth'
                    controlnet.append(ControlNetModel.from_single_file(cn_model_path, local_files_only=True))
                OV_df_unet.controlnet = controlnet
            else:
                cn_model_dir_path = os.path.join(scripts.basedir(),'extensions','sd-webui-controlnet','models')
                cn_model_path = os.path.join(cn_model_dir_path, model_state.control_models[0])
                if os.path.isfile(cn_model_path + '.pt'):
                    cn_model_path = cn_model_path + '.pt'
                elif os.path.isfile(cn_model_path + '.safetensors'):
                    cn_model_path = cn_model_path + '.safetensors'
                elif os.path.isfile(cn_model_path + '.pth'):
                    cn_model_path = cn_model_path + '.pth'
                controlnet = ControlNetModel.from_single_file(cn_model_path, local_files_only=True)
                OV_df_unet.controlnet = controlnet
            
            # process image
            print('begin self.prepare_image')
            for i in range(len(OV_df_unet.control_images)):
                OV_df_unet.canny_images[i] = self.prepare_image(OV_df_unet.canny_images[i], 512, 512, 1, 1, torch.device('cpu'), torch.float32, True, False)
            print('end self.prepare_image')
            
            
            
            
            
        else:
            input('no controlnet detected')
            
            

                

            
            
            
                
            
            
            
            
             
            
            
            
                
                
        
        #print(self.engine) # not none
        print('end of activate')
        

    def deactivate(self):
        del self.engine


def get_instances_of(cls):
    return [obj for obj in gc.get_objects() if isinstance(obj, cls)]
        

class Script(scripts.Script):
    def title(self):
        return "Accelerate with OpenVINO Extension"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        core = Core()

        def get_config_list():
            config_dir_list = os.listdir(os.path.join(os.getcwd(), 'configs'))
            config_list = []
            config_list.append("None")
            for file in config_dir_list:
                if file.endswith('.yaml'):
                    config_list.append(file)
            return config_list
        def get_vae_list():
            vae_dir_list = os.listdir(os.path.join(os.getcwd(), 'models', 'VAE'))
            vae_list = []
            vae_list.append("None")
            vae_list.append("Disable-VAE-Acceleration")
            for file in vae_dir_list:
                if file.endswith('.safetensors') or file.endswith('.ckpt') or file.endswith('.pt'):
                    vae_list.append(file)
            return vae_list
        def get_refiner_list():
            refiner_dir_list = os.listdir(os.path.join(os.getcwd(), 'models', 'Stable-diffusion'))
            refiner_list = []
            refiner_list.append("None")
            for file in refiner_dir_list:
                if file.endswith('.safetensors') or file.endswith('.ckpt') or file.endswith('.pt'):
                    refiner_list.append(file)
            return refiner_list

        with gr.Accordion('OV Extension Template', open=False):
            
            enable_ov_extension = gr.Checkbox(label='check to enable OV extension', value=False)
            
            

        
        def enable_change(choice):
                if choice:
                    processing._process_images = processing.process_images
                    print("enable vo extension")
                    processing.process_images = self.run
                else:
                    if hasattr(processing, '_process_images'):
                        processing.process_images = processing._process_images
                    print('disable ov extension')
        
        
        def device_change(choice):
            if (model_state.device == choice):
                return gr.update(value="Device selected is " + choice, visible=True)
            else:
                model_state.device = choice
                model_state.recompile = 1
                return gr.update(value="Device changed to " + choice + ". Model will be re-compiled", visible=True)
        #openvino_device.change(device_change, openvino_device, warmup_status)
        def vae_change(choice):
            if (model_state.vae_ckpt == choice):
                return gr.update(value="vae_ckpt selected is " + choice, visible=True)
            else:
                model_state.vae_ckpt = choice
                model_state.recompile = 1
                return gr.update(value="Custom VAE changed to " + choice + ". Model will be re-compiled", visible=True)
        #vae_ckpt.change(vae_change, vae_ckpt, vae_status)
        def refiner_ckpt_change(choice):
            if (model_state.refiner_ckpt == choice):
                return gr.update(value="Custom Refiner selected is " + choice, visible=True)
            else:
                model_state.refiner_ckpt = choice
        #refiner_ckpt.change(refiner_ckpt_change, refiner_ckpt)
        return [enable_ov_extension]
    
    def hook(self, p, model):
        def prepare_image(
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
        ):
            from diffusers.image_processor import VaeImageProcessor
            control_image_processor = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True, do_normalize=False)
            image = control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
            image_batch_size = image.shape[0]

            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # image batch size is the same as prompt batch size
                repeat_by = num_images_per_prompt

            image = image.repeat_interleave(repeat_by, dim=0)

            #image = image.to(device=device, dtype=dtype)

            if do_classifier_free_guidance and not guess_mode:
                image = torch.cat([image] * 2)

            return image
        def ov_cnet_forward(self,x, timesteps, context, y=None, **kwargs):
            print('ov_cnet forward called!@!!!!!!!!!!!')
            print(x.shape, timesteps.shape, context.shape, y.shape if y else None, kwargs, len(OV_df_unet.control_images), OV_df_unet.control_images[0].shape ) # (952, 982, 3)
            print('check shapes')
            
           

            
            
            image = prepare_image(OV_df_unet.control_images[0], 512,512, 1, 1, 'cpu', 'fp32', False, False)

            print('processed image shape:', image.shape) # torch.Size([1, 3, 512, 512])
            
            print('check processed image shape')
            
            ''' 
                def forward(self, x, timesteps, context, y=None, **kwargs):
                    

                    import gc
                    
                    
                    
                    
                    

                    print(x.shape, timesteps.shape, context.shape, hint.shape)
                    input('check shapes') # torch.Size([2, 4, 64, 64]) torch.Size([2]) torch.Size([2, 77, 768]) torch.Size([1, 3, 512, 512])
                        
                    #print('replace with OV cnet') # error compiling
                    #param.control_model = torch.compile(param.control_model, backend = 'openvino')  replace this with the diffusers model works? 
                    from diffusers import ControlNetModel

                    control = param.control_model(
                        x=x_in, # latent [2,4,64,64]
                        hint=hint, # input image 
                        timesteps=timesteps, # shape [2]
                        context=controlnet_context, # [2,78,768]
                        y=y
                    )
                    for c in control: print(c.shape)
                    input('check ref shape')
                    
                    torch.Size([2, 320, 64, 64])
                    torch.Size([2, 320, 64, 64])
                    torch.Size([2, 320, 64, 64])
                    torch.Size([2, 320, 32, 32])
                    torch.Size([2, 640, 32, 32])
                    torch.Size([2, 640, 32, 32])
                    torch.Size([2, 640, 16, 16])
                    torch.Size([2, 1280, 16, 16])
                    torch.Size([2, 1280, 16, 16])
                    torch.Size([2, 1280, 8, 8])
                    torch.Size([2, 1280, 8, 8])
                    torch.Size([2, 1280, 8, 8])
                    torch.Size([2, 1280, 8, 8])
            '''
                    
            height, width = image.shape[-2:]

            # 6.5 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if OV_df_unet.engine.config.time_cond_proj_dim is not None:
                print('guidance scale')
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            
            

            control_model = OV_df_unet.controlnet
            #control_model = torch.compile(OV_df_unet.controlnet, backend = 'openvino')  # ControlNetModel.from_single_file('./extensions/sd-webui-controlnet/models/control_v11p_sd15_canny_fp16.safetensors', local_files_only=True)
            down_block_res_samples, mid_block_res_sample = control_model(
                x,
                timesteps,
                context,
                image,
                return_dict=False,
                #y=y
            )
            print([d.shape for d in down_block_res_samples], mid_block_res_sample.shape)
            print('check diffusers  controlnet output')
            #import pdb; pdb.set_trace()

            
            #OV_df_unet.engine = torch.compile(OV_df_unet.engine, backend = 'openvino')
            noise_pred = OV_df_unet.engine(
                    x,
                    timesteps,
                    context,
                    #timestep_cond=timestep_cond,
                    #cross_attention_kwargs=({}),
                    #down_block_additional_residuals=down_block_res_samples,
                    #mid_block_additional_residual=mid_block_res_sample,
                    #added_cond_kwargs=({}), #added_cond_kwargs,
                    #return_dict=False,
                    ).sample

            
            return noise_pred

            

            
            
            
            
            
            ### end of adaption from controlnet extensino forward #### 

            
    
    
        def forward_webui(*kargs, **kwargs):
            print('forward_webui called')
            #print(kargs, kwargs)
            print('len(kargs):', len(kargs)) # 3
            print('len(kwargs):', len(kwargs)) # 1
            
            return ov_cnet_forward(*kargs, **kwargs)
        
        sd_ldm = p.sd_model
        unet = sd_ldm.model.diffusion_model
        
        from ldm.modules.diffusionmodules.openaimodel import UNetModel
        
        model.forward = forward_webui.__get__(model, UNetModel)
        
        

    def process(self, p, *args):
        print("ov process called")
        import sys
        import os
        from modules import scripts
        current_extension_directory = scripts.basedir() + '/extensions/sd-webui-controlnet/scripts'
        print('current_extension_directory', current_extension_directory)
        sys.path.append(current_extension_directory)
        from hook import UnetHook

        
        
        #print(args)
        enable_ov = args[0] != False and args[0] != 'False'
        
        if not enable_ov:
            print('ov disabled, do nothing')
            return
        
        global OV_df_unet
        #if OV_df_unet == None:
        OV_df_unet = OVUnet(p.sd_model_name)
        OV_df_unet.process = p
        self.apply_unet(p)
        
        
        print('ov enabled')
        print('sd_unet.current_unet_option',sd_unet.current_unet_option) # NOne
        #print("current_unet:",sd_unet.current_unet)
        
        print("shared.opts:",shared.opts.data_labels)
        print("p.sd_model_name:",p.sd_model_name)

        #apply A1111 styled prompt weighting
        '''
        if p.extra_generation_params:
            print('unet hook begin')
            sd_ldm = p.sd_model
            unet = sd_ldm.model.diffusion_model
            self.hook(p, unet)
    
            print('end of unet hook')
        else:
            print('no cnet, do nothing')
        '''
        
        print('finished process')
        
                
        
    def apply_unet(self, p):
        if sd_unet.current_unet is not None:
            print("Deactivating unet: ", sd_unet.current_unet)
            sd_unet.current_unet.deactivate()
        
        print("begin activate unet")
        sd_unet.current_unet = OV_df_unet

        sd_ldm = p.sd_model
        unet = sd_ldm.model.diffusion_model
        print('force forward ')
        unet.forward = OV_df_unet.forward
        print('finish force forward ')
        
        #sd_unet.current_unet.option = sd_unet_option
        #sd_unet.current_unet_option = sd_unet_option

        #print(f"Activating unet: {sd_unet.current_unet.option.label}")
        sd_unet.current_unet.activate(p)