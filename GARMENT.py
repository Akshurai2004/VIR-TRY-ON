import os
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from u2net import U2NET
from torch_tps import ThinPlateSpline
from diffusers import StableDiffusionInpaintPipeline

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. YOLO Person Detection
person_img_path = "images/person.jpg"
yolo_model = YOLO("checkpoints/yolov8n.pt")
yolo_results = yolo_model(person_img_path)
os.makedirs("output", exist_ok=True)
yolo_results[0].save(filename="output/yolo_crop.jpg")

# 2. U2Net Segmentation
u2net_model = U2NET(3, 1)
u2net_model.load_state_dict(torch.load("checkpoints/u2net.pth", map_location=device))
u2net_model.to(device).eval()

img = Image.open(person_img_path).convert("RGB")
transform = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor()])
input_tensor = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    pred_mask = u2net_model(input_tensor)[0]

def save_output(pred, output_path):
    pred = pred.squeeze().cpu().data.numpy()
    pred = (pred * 255).astype(np.uint8)
    Image.fromarray(pred).save(output_path)

mask_path = "output/pred_mask.png"
save_output(pred_mask, mask_path)

# 3. GMM + TryOnGenerator (Checkpoints used)
class FeatureExtractor(torch.nn.Module):
    # ... (your definition, as in your repo)

class TPSRegressor(torch.nn.Module):
    # ... (your definition, as in your repo)

class GMM(torch.nn.Module):
    # ... (your definition, as in your repo)

class TryOnGenerator(torch.nn.Module):
    # ... (your definition, as in your repo)

def warp_image_tps(source_image, theta, device):
    # ... (your warp logic)

def upscale_image(image_pil, upscale_factor=4):
    # ... (your upscale logic)

def run_inference(agnostic_path, cloth_path, mask_path, output_path, checkpoint_dir):
    target_size = (512, 768)
    transform = transforms.Compose([
        transforms.Resize(target_size[::-1]),
        transforms.ToTensor(),
    ])

    gmm = GMM().to(device)
    tryon_gen = TryOnGenerator().to(device)

    # --- CHECKPOINTS USED HERE ---
    gmm.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'gmm_pretrained.pth'), map_location=device))
    tryon_gen.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'tryon_gen_pretrained.pth'), map_location=device))
    gmm.eval()
    tryon_gen.eval()

    agnostic_img = Image.open(agnostic_path).convert('RGB')
    cloth_img = Image.open(cloth_path).convert('RGB')
    mask_img = Image.open(mask_path).convert('L')
    agnostic_tensor = transform(agnostic_img).unsqueeze(0).to(device)
    cloth_tensor = transform(cloth_img).unsqueeze(0).to(device)
    mask_tensor = transform(mask_img).unsqueeze(0).to(device)
    with torch.no_grad():
        theta = gmm(agnostic_tensor, cloth_tensor)
        warped_cloth = warp_image_tps(cloth_tensor, theta, device)
        warped_mask = warp_image_tps(mask_tensor, theta, device)
        warped_cloth = warped_cloth * warped_mask
        render_img, comp_mask = tryon_gen(agnostic_tensor, warped_cloth)
        comp_mask_blend = comp_mask.expand_as(render_img)
        final_result = comp_mask_blend * warped_cloth + (1 - comp_mask_blend) * render_img
    output_tensor = final_result.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(output_tensor)
    output_image = upscale_image(output_image, upscale_factor=4)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_image.save(output_path, quality=100)

agnostic_file = "images/output_agnostic.jpg"
cloth_file = "images/cloth1.png"
mask_file = mask_path
checkpoint_folder = "checkpoints"
output_file = "images/highres_img.jpg"

run_inference(
    agnostic_path=agnostic_file,
    cloth_path=cloth_file,
    mask_path=mask_file,
    output_path=output_file,
    checkpoint_dir=checkpoint_folder
)

# 4. Diffusion Model Inpainting (Checkpoints used in diffusers pipeline)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)

init_image = Image.open(output_file).convert("RGB")
mask_image = Image.open(mask_file).convert("L")
if mask_image.size != init_image.size:
    mask_image = mask_image.resize(init_image.size)
width, height = init_image.size
if width % 8 != 0 or height % 8 != 0:
    new_width = (width // 8) * 8
    new_height = (height // 8) * 8
    init_image = init_image.resize((new_width, new_height))
    mask_image = mask_image.resize((new_width, new_height))

prompt = (
    "a realistic photo of a person wearing simple clothes; "
    "preserve original face, hands, and background; "
    "do not add any buttons, collars, text, or logos; "
    "only improve realism and texture of the existing upper clothing"
)
negative_prompt = (
    "buttons, collars, logos, text, accessories, jewelry, unnatural patterns, distorted face, distorted hands"
)
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=mask_image,
    guidance_scale=6.5,
    strength=0.75,
    num_inference_steps=50
).images[0]
final_output_path = "images/final_image.png"
os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
result.save(final_output_path)

print(f"✅ Inpainting and smoothing complete! Result saved at: {final_output_path}")
