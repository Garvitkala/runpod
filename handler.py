mport runpod
import torch
from diffusers import LTXVideoPipeline
from diffusers.utils import export_to_video

# Load LTX-Video model
pipe = LTXVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video", 
    torch_dtype=torch.bfloat16
).to("cuda")

def handler(job):
    job_input = job['input']
    prompt = job_input.get('prompt', 'A cat running in a field')
    num_frames = job_input.get('num_frames', 16)

    video_frames = pipe(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=20,
    ).frames[0]

    # Export and return
    return {"status": "success", "message": "Video generated successfully"}

runpod.serverless.start({"handler": handler})