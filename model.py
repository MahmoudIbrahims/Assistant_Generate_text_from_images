from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
import torch
from llava.utils import disable_torch_init
disable_torch_init()

#disable_torch_init()

def llava_model():
    MODEL = "4bit/llava-v1.5-13b-3GB"
    model_name = get_model_name_from_path(MODEL)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=MODEL, model_base=None, model_name=model_name, load_4bit=True)
    return tokenizer, model, image_processor, context_len
    

