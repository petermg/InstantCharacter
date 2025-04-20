import torch
import random
import numpy as np
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

from pipeline import InstantCharacterFluxPipeline

# Global variables
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Pre-trained weights
ip_adapter_path = hf_hub_download(repo_id="Tencent/InstantCharacter", filename="instantcharacter_ip-adapter.bin")
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
birefnet_path = 'ZhengPeng7/BiRefNet'
makoto_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai", filename="Makoto_Shinkai_style.safetensors")
ghibli_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Ghibli", filename="ghibli_style.safetensors")

# Initialize InstantCharacter pipeline
pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
pipe.to(device)
pipe.init_adapter(
    image_encoder_path=image_encoder_path,
    image_encoder_2_path=image_encoder_2_path,
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024),
)

# Load matting model for background removal
birefnet = AutoModelForImageSegmentation.from_pretrained(birefnet_path, trust_remote_code=True)
birefnet.to('cuda' if device == 'cuda' else 'cpu')
birefnet.eval()
birefnet_transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def remove_background(subject_image):
    """Remove background from the input image using BiRefNet."""
    if subject_image is None:
        raise ValueError("No image provided. Please upload an image.")
    
    # Convert input to PIL Image if it's a NumPy array
    if isinstance(subject_image, np.ndarray):
        subject_image = Image.fromarray(subject_image.astype('uint8')).convert("RGB")
    elif not isinstance(subject_image, Image.Image):
        subject_image = Image.open(subject_image).convert("RGB")
    
    def infer_matting(img_pil):
        input_images = birefnet_transform_image(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(img_pil.size)
        mask = np.array(mask)
        mask = mask[..., None]
        return mask

    def get_bbox_from_mask(mask, th=128):
        height, width = mask.shape[:2]
        x1, y1, x2, y2 = 0, 0, width - 1, height - 1
        sample = np.max(mask, axis=0)
        for idx in range(width):
            if sample[idx] >= th:
                x1 = idx
                break
        sample = np.max(mask[:, ::-1], axis=0)
        for idx in range(width):
            if sample[idx] >= th:
                x2 = width - 1 - idx
                break
        sample = np.max(mask, axis=1)
        for idx in range(height):
            if sample[idx] >= th:
                y1 = idx
                break
        sample = np.max(mask[::-1], axis=1)
        for idx in range(height):
            if sample[idx] >= th:
                y2 = height - 1 - idx
                break
        x1 = np.clip(x1, 0, width-1).round().astype(np.int32)
        y1 = np.clip(y1, 0, height-1).round().astype(np.int32)
        x2 = np.clip(x2, 0, width-1).round().astype(np.int32)
        y2 = np.clip(y2, 0, height-1).round().astype(np.int32)
        return [x1, y1, x2, y2]

    def pad_to_square(image, pad_value=255):
        h, w = image.shape[:2]
        if h == w:
            return image
        padd = abs(h - w)
        padd_1 = padd // 2
        padd_2 = padd - padd_1
        if h > w:
            pad_param = ((0, 0), (padd_1, padd_2), (0, 0))
        else:
            pad_param = ((padd_1, padd_2), (0, 0), (0, 0))
        image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
        return image

    salient_object_mask = infer_matting(subject_image)[..., 0]
    x1, y1, x2, y2 = get_bbox_from_mask(salient_object_mask)
    subject_image = np.array(subject_image)
    salient_object_mask[salient_object_mask > 128] = 255
    salient_object_mask[salient_object_mask < 128] = 0
    sample_mask = np.concatenate([salient_object_mask[..., None]] * 3, axis=2)
    obj_image = sample_mask / 255 * subject_image + (1 - sample_mask / 255) * 255
    crop_obj_image = obj_image[y1:y2, x1:x2]
    crop_pad_obj_image = pad_to_square(crop_obj_image, 255)
    return Image.fromarray(crop_pad_obj_image.astype(np.uint8))

def randomize_seed_fn(seed, randomize_seed):
    """Randomize seed if enabled."""
    if randomize_seed:
        return random.randint(0, MAX_SEED)
    return seed

def generate_image(input_image, prompt, scale, style, guidance_scale, num_steps, seed, randomize_seed):
    """Generate an image using the InstantCharacter pipeline."""
    if input_image is None:
        raise ValueError("No image uploaded. Please upload an image.")
    
    seed = randomize_seed_fn(seed, randomize_seed)
    input_image = remove_background(input_image)
    
    if style == "None":
        style = None
    
    if style is None:
        images = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=1024,
            height=1024,
            subject_image=input_image,
            subject_scale=scale,
            generator=torch.manual_seed(seed),
        ).images
    else:
        lora_file_path = makoto_style_lora_path if style == "Makoto Shinkai style" else ghibli_style_lora_path
        trigger = "Makoto Shinkai style" if style == "Makoto Shinkai style" else "ghibli style"
        images = pipe.with_style_lora(
            lora_file_path=lora_file_path,
            trigger=trigger,
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=1024,
            height=1024,
            subject_image=input_image,
            subject_scale=scale,
            generator=torch.manual_seed(seed),
        ).images
    
    return images

# Gradio interface
with gr.Blocks(title="InstantCharacter") as demo:
    gr.Markdown("""
    # InstantCharacter: Personalize Any Character
    Upload an image, describe the scene, and customize your character with different styles!
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Character Image", type="numpy")
            prompt = gr.Textbox(label="Prompt", value="A character is riding a bike in snow")
            scale = gr.Slider(0, 1.5, value=1.0, step=0.01, label="Subject Scale")
            style = gr.Dropdown(["None", "Makoto Shinkai style", "Ghibli style"], value="Makoto Shinkai style", label="Style")
            
            with gr.Accordion("Advanced Settings", open=False):
                guidance_scale = gr.Slider(1, 7, value=3.5, step=0.1, label="Guidance Scale")
                num_steps = gr.Slider(5, 50, value=28, step=1, label="Inference Steps")
                seed = gr.Slider(-1000000, 1000000, value=123456, step=1, label="Seed")
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            
            generate_button = gr.Button("Generate Image")
        
        with gr.Column():
            output_gallery = gr.Gallery(label="Generated Images")
    
    # Event handlers
#    input_image.change(
#        fn=lambda img: img,
#        inputs=input_image,
#        outputs=input_image,
#        queue=False
#    )
    
    generate_button.click(
        fn=generate_image,
        inputs=[input_image, prompt, scale, style, guidance_scale, num_steps, seed, randomize_seed],
        outputs=output_gallery
    )

demo.launch(server_name="0.0.0.0", server_port=80)
