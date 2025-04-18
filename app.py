import gradio as gr
import subprocess
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, check=True, shell=True)
except subprocess.CalledProcessError as e:
    print(f"Error installing flash-attn: {e}")
    print("Continuing without flash-attn.")

device = "cuda" if torch.cuda.is_available() else "cpu"
vision_language_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True).to(device).eval()
vision_language_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)

def describe_image(uploaded_image):
    """
    Generates a detailed description of the input image.

    Args:
        uploaded_image (PIL.Image.Image or numpy.ndarray): The image to describe.

    Returns:
        str: A detailed textual description of the image.
    """
    if not isinstance(uploaded_image, Image.Image):
        uploaded_image = Image.fromarray(uploaded_image)

    inputs = vision_language_processor(text="<MORE_DETAILED_CAPTION>", images=uploaded_image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = vision_language_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    generated_text = vision_language_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    processed_description = vision_language_processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(uploaded_image.width, uploaded_image.height)
    )
    image_description = processed_description["<MORE_DETAILED_CAPTION>"]
    print("\nImage description generated!:", image_description)
    return image_description

image_description_interface = gr.Interface(
    fn=describe_image,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Textbox(label="Generated Caption", lines=4, show_copy_button=True),
    live=False,
)

image_description_interface.launch(debug=True)
