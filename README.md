# Caterpillar

Caterpillar is a document to XML parser. It use [guidance](https://github.com/guidance-ai/guidance) framework, marker library,
[Llama.cpp](https://github.com/ggerganov/llama.cpp) and Qwen3 to parse pdf to XML. Caterpillar understands document layout, tables in it and creates clean machine-readable document representation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python parser.py path/to/document.pdf
```

The script converts the PDF to HTML and streams the resulting XML to stdout.
Sample files are available in `test_files/` for experimentation.