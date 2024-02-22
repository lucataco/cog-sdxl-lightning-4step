# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from typing import List
from transformers import CLIPImageProcessor
from diffusers import (
    StableDiffusionXLPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

UNET = "sdxl_lightning_4step_unet.pth"
MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
UNET_CACHE = "unet-cache"
BASE_CACHE = "checkpoints"
SAFETY_CACHE = "safety-cache"
FEATURE_EXTRACTOR = "feature-extractor"
MODEL_URL = "https://weights.replicate.delivery/default/sdxl-lightning/sdxl-1.0-base-lightning.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
UNET_URL = "https://weights.replicate.delivery/default/comfy-ui/unet/sdxl_lightning_4step_unet.pth.tar"


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DPM++2MSDE": KDPM2AncestralDiscreteScheduler,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        print("Loading model")
        if not os.path.exists(BASE_CACHE):
            download_weights(MODEL_URL, BASE_CACHE)
        print("Loading Unet")
        if not os.path.exists(UNET_CACHE):
            download_weights(UNET_URL, UNET_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)
        print("Loading txt2img pipeline...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_BASE,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=BASE_CACHE,
            local_files_only=True,
        ).to("cuda")
        unet_path = os.path.join(UNET_CACHE, UNET)
        self.pipe.unet.load_state_dict(torch.load(unet_path, map_location="cuda"))
        print("setup took: ", time.time() - start)

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A superhero smiling"),
        negative_prompt: str = Input(
            description="Negative Input prompt", default="worst quality, low quality"
        ),
        width: int = Input(
            description="Width of output image. Recommended 1024 or 1280", default=1024
        ),
        height: int = Input(
            description="Height of output image. Recommended 1024 or 1280", default=1024
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. 4 for best results",
            ge=1,
            le=10,
            default=4,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance. Recommended 7-8",
            ge=0,
            le=50,
            default=0,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        sdxl_kwargs = {}
        print(f"Prompt: {prompt}")
        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height
        pipe = self.pipe

        pipe.scheduler = SCHEDULERS[scheduler].from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        output = pipe(**common_args, **sdxl_kwargs)

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
