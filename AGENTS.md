# AGENTS

## Project Goal
Caterpillar is a document to XML parser. It uses the guidance framework, marker library, Llama.cpp and Qwen3 to parse PDFs to XML. Caterpillar understands document layout, tables and creates clean machine-readable document representation.

## Code Conventions
- Follow PEP 8 style guidelines.
- Use tabs for indentation instead of spaces.

## Project Structure
- `parser.py`: Entry point script that converts PDFs into XML via HTML conversion.
- `chat_template.py`: Contains chat template for the guidance framework.
- `caterpillar.xsd`: XML schema for output documents.
- `test_files/`: Sample PDF files for experimentation.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies.
