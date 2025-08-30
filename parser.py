#!/usr/bin/env python3


from typing import Optional
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.renderers.html import HTMLRenderer
from marker.config.parser import ConfigParser
from marker.output import json_to_html
from guidance import system, user, assistant, gen, select
from guidance.models import LlamaCpp
from guidance.library import one_or_more, capture, with_temperature

from chat_template import Qwen3ChatTemplate


    
lm = LlamaCpp(model="models/Qwen3-4B-Thinking-2507-F16.gguf", 
            chat_template=Qwen3ChatTemplate,
            #n_ctx=8000, 
            n_gpu_layers=-1)

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)

cfg = ConfigParser({"output_format": "html"})
converter = PdfConverter(
    artifact_dict=create_model_dict(),
    config=cfg.generate_config_dict(),
    processor_list=cfg.get_processors(),
    renderer=cfg.get_renderer(),
    llm_service=cfg.get_llm_service()
)

def html(path: str, mime: Optional[str]):
    document = converter(path)
    html, images = document.html, document.images


if __name__ == "__main__":
    html("test_files/Doczilla Pro - Требования к ПАК.pdf", "application/pdf")