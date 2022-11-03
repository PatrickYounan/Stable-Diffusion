import os
import random

import torch
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline


class StableDiffusion:

    def __init__(self, model, beta_start=0.00085, beta_end=0.012, inference_steps=50):
        self.model = model
        self.starting_seed = random.randint(0, 99999999)
        self.inference_steps = inference_steps

        torch.manual_seed(self.starting_seed)

        lms = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear")
        lms.set_timesteps(inference_steps)

        self.pipe = StableDiffusionPipeline.from_pretrained(self.model, revision="fp16", torch_dtype=torch.float16, scheduler=lms)
        self.pipe = self.pipe.to("cuda")

    @staticmethod
    def generate_path(directory):
        if not os.path.isdir(directory):
            print(f"Creating new directory: {directory}")
            os.mkdir(directory)

        image_count = len(os.listdir(directory))

        if image_count > 0:
            print(f"Found {image_count} images in {directory}...")
        return image_count

    def generate_by_image(self, directory, prompt, image, guidance_scale=7.5, width=512, height=512, iterations=1):
        local_directory = f"images/{directory}"
        image_count = self.generate_path(local_directory)

        for index in range(iterations):
            name_index = image_count + index
            torch.manual_seed(torch.seed() + name_index)
            image = self.pipe(prompt=prompt, image=image, guidance_scale=guidance_scale, width=width, height=height).images[0]
            image.save(f"{local_directory}/{name_index:04}.png")
            print(f"Saved image: {name_index:04} Seed=[{torch.seed()}]")

    def generate_by_text(self, directory, prompt, guidance_scale=7.5, width=512, height=512, iterations=1):
        local_directory = f"images/{directory}"
        image_count = self.generate_path(local_directory)

        for index in range(iterations):
            name_index = image_count + index
            torch.manual_seed(torch.seed() + name_index)
            image = self.pipe(prompt=prompt, guidance_scale=guidance_scale, width=width, height=height).images[0]
            image.save(f"{local_directory}/{name_index:04}.png")
            print(f"Saved image: {name_index:04} Seed=[{torch.seed()}]")
