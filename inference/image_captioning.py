from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from PIL import Image

def load_processor(processor_path):
    # processor = AutoProcessor.from_pretrained("microsoft/git-base")
    processor = AutoProcessor.from_pretrained(processor_path)
    return processor

def load_model(model_path):
    # model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model

def generate_caption(image_path: str):

    image = Image.read(image_path)

    # load image
    width, height = image.size

    model_path = "result"
    model = load_model(model_path)
    processor = load_processor(model_path)

    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt").to('cpu')
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption