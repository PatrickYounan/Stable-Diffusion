from stable_diffusion import StableDiffusion

diffusion = StableDiffusion(model="CompVis/stable-diffusion-v1-4")

diffusion.generate_by_text(
    directory="borat",
    prompt="Bang bang skit skit nigger, borat, 8k, concept art",
    iterations=50
)
