# Caterpillar

Caterpillar provides experiments for parsing documents and formatting them
with the [guidance](https://github.com/guidance-ai/guidance) framework and
[Llama.cpp](https://github.com/ggerganov/llama.cpp).  The code converts PDFs to
HTML, feeds the HTML to a Qwen 3 model, and asks the model to return a
book‑like XML representation of the document.

## Features

- **PDF conversion** – Uses `marker-pdf` to convert a PDF into HTML before
  processing it with the language model【F:parser.py†L14-L32】
- **Book‑style formatting** – Supplies the model with detailed rules for
  structuring chapters, tables, and metadata so that the output resembles a
  reader‑friendly book【F:parser.py†L34-L70】【F:parser.py†L109-L148】
- **Custom chat template** – Implements `Qwen3ChatTemplate` so that Guidance
  can speak the native Qwen 3 format and handle tool messages properly【F:chat_template.py†L193-L210】
- **Tool calling framework** – The `Tools` helper registers Python functions,
  validates arguments with Pydantic, and emits a system preface describing
  the available tools for the model to call【F:chat_template.py†L257-L299】
- **Markdown list capture** – `md_list` consumes numbered or bullet lists
  directly from the model stream and returns them as Python lists【F:chat_template.py†L355-L396】

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project depends on:

- `marker-pdf`
- `guidance`
- `llama_cpp_python`

These are listed in `requirements.txt`【F:requirements.txt†L1-L3】.

## Usage

```bash
python parser.py path/to/document.pdf
```

The script converts the PDF to HTML and streams the resulting XML to stdout.
Sample files are available in `test_files/` for experimentation.

## Development

Run basic checks:

```bash
python -m py_compile chat_template.py parser.py
python -m pytest
```

## License

This repository does not currently specify a license.

