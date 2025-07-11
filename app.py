import gradio as gr
import subprocess
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Attempt to install flash-attn
try:
    subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, check=True, shell=True)
except subprocess.CalledProcessError as e:
    print(f"Error installing flash-attn: {e}")
    print("Continuing without flash-attn.")

# Determine the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the base model and processor
try:
    vision_language_model_base = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True).to(device).eval()
    vision_language_processor_base = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
except Exception as e:
    print(f"Error loading base model: {e}")
    vision_language_model_base = None
    vision_language_processor_base = None

# Load the large model and processor
try:
    vision_language_model_large = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True).to(device).eval()
    vision_language_processor_large = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)
except Exception as e:
    print(f"Error loading large model: {e}")
    vision_language_model_large = None
    vision_language_processor_large = None

def describe_image(uploaded_image, model_choice):
    """
    Generates a detailed description of the input image using the selected model.

    Args:
        uploaded_image (PIL.Image.Image): The image to describe.
        model_choice (str): The model to use, either "Base" or "Large".

    Returns:
        str: A detailed textual description of the image or an error message.
    """
    if uploaded_image is None:
        return "Please upload an image."

    if model_choice == "Base":
        if vision_language_model_base is None:
            return "Base model failed to load."
        model = vision_language_model_base
        processor = vision_language_processor_base
    elif model_choice == "Large":
        if vision_language_model_large is None:
            return "Large model failed to load."
        model = vision_language_model_large
        processor = vision_language_processor_large
    else:
        return "Invalid model choice."

    if not isinstance(uploaded_image, Image.Image):
        uploaded_image = Image.fromarray(uploaded_image)

    inputs = processor(text="<MORE_DETAILED_CAPTION>", images=uploaded_image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    processed_description = processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(uploaded_image.width, uploaded_image.height)
    )
    image_description = processed_description["<MORE_DETAILED_CAPTION>"]
    print("\nImage description generated!:", image_description)
    return image_description

# Description for the interface
description = "Select the model to use for generating the image description. 'Base' is smaller and faster, while 'Large' is more accurate but slower."
if device == "cpu":
    description += " Note: Running on CPU, which may be slow for large models."

# Create the Gradio interface
image_description_interface = gr.Interface(
    fn=describe_image,
    inputs=[
        gr.Image(label="Upload Image", type="pil"),
        gr.Radio(["Base", "Large"], label="Model Choice", value="Base")
    ],
    outputs=gr.Textbox(label="Generated Caption", lines=4, show_copy_button=True),
    live=False,
    title="Florence-2 Models Image Captions",
    description=description
)

# Launch the interface
image_description_interface.launch(debug=True, ssr_mode=False)
