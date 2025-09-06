#!/usr/bin/env python3


import sys
from typing import Optional
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from guidance.models import LlamaCpp
from guidance import system, user, assistant, gen,  special_token, select

from chat_template import Qwen3ChatTemplate, md_list, thoughts

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

def analize_section(lm, section_name: str, section_raw_html: str, level: int = 3, prefix: str = "sec") -> str:
	"""Recursively analyze a section and return its XML content."""
	lm += f"Exploring '{section_name}' section structure and potential subsections.\n\n"
	lm += section_raw_html + "\n"
	lm += select(["Subsections:\n\n", "No subsections.\n\n"], name=f"{prefix}_subs_choice")
	subsections: list[str] = []
	if lm[f"{prefix}_subs_choice"].startswith("Subsections"):
		subsections = md_list(lm, style="numbered")
		lm += "\n".join(subsections) + "\n\n"
	indent = "\t" * (level - 1)
	sub_indent = "\t" * level
	result = ""
	if subsections:
		result += f"{indent}<nav>\n{sub_indent}<ul>\n"
		for idx, sub in enumerate(subsections):
			title = sub.split('. ')[1] if '. ' in sub else sub
			anchor = f"{prefix}_{idx+1}"
			result += f"{sub_indent}\t<li><a href=\"#{anchor}\">{title}</a></li>\n"
		result += f"{sub_indent}</ul>\n{indent}</nav>\n"
		for idx, sub in enumerate(subsections):
			title = sub.split('. ')[1] if '. ' in sub else sub
			anchor = f"{prefix}_{idx+1}"
			lm += f"Raw HTML for subsection '{title}':\n<raw>\n"
			lm += gen(name=f"{anchor}_raw", stop="</raw>")
			lm += "</raw>\n"
			raw_html = lm[f"{anchor}_raw"]
			result += f"{indent}<section>\n{sub_indent}<h{level} id=\"{anchor}\">{title}</h{level}>\n"
			result += analize_section(lm, title, raw_html, level + 1, anchor)
			result += f"{indent}</section>\n"
	else:
		lm += "format_text:\n"
		lm += gen(name=f"{prefix}_xml")
		result += f"{indent}" + lm[f"{prefix}_xml"] + "\n"
	return result

def _parse_html(html: str):
	lm = LlamaCpp(model="models/Qwen3-4B-Thinking-2507-F16.gguf", 
			chat_template=Qwen3ChatTemplate,
			n_ctx=31000, 
			echo=True,
			n_gpu_layers=-1)

	with system():
		lm += '''
Your task is to format the given HTML so that it looks like a book and is easy to read. Output the result as XML that must:

* Reflect the logical structure of the document: chapters, cards, and metadata are separate blocks.
* Be free of purely visual instructions such as `<br>`.
* Determine heading levels by the document’s structure, not by visual text size.
* Have a book‑like hierarchy: document, chapter, sections, paragraphs.
* Include all information from the original HTML (except visual/aesthetic details) and structure it correctly in the final XML.
* Allow typos to be corrected.
* Put author of the document (if presented) to the `<author>` tag.
* Put document date (if presented) to the `<date>` tag.

Convert tables to a rectangular (CSV‑like) form, but make sure the new tables are understood the same way as the original tables. You may merge/split and rearrange cells. You may also split one table into several. The resulting table must consist of homogeneous, independent rows, have a header, and resemble CSV.

Use the following rules:

* If the original table has no header, its header spans multiple rows, or parts of the header are merged, create a single header row that can describe the records in the table without losing information.
* Examine the table’s rows and group them by field types. Split the original table into several tables (one table per group) if necessary.
* Add rows if needed. Each row must represent a single record and consist of the fields defined in the header.
* Don’t hesitate to duplicate information in cells if that’s required for correct understanding.
* Rows must be independent: each row should include all information necessary for understanding.
* Put header to `<thead>` and body to `<tbody>`.
* Extract or invent caption for the table and put it in a `<caption>` tag.

The resulting XML must use the following tags:

* Structure information only into `<section>` (for sections and chapters), `<article>` (for a standalone article), and `<aside>` (for information not directly related to the current block).
* Every `<section>`, `<aside>`, and `<article>` must include heading tags `<h1>`, `<h2>`, `<h3>`, `<h4>`, `<h5>`, `<h6>`. Create a heading if needed.
* Every `<article>` must include a table of contents in `<nav>`; create it if necessary.
* Text must be wrapped in `<p>` tags.
* Links must use `<a>` tags.
* Tables must use `<table>`, `<tr>`, `<th>`, `<td>`, `<caption>`, `<thead>`, `<tbody>`.
* Images must use `<img>`.
* Use `<date>` and `<author>` tags for document metadata.

Example of the resulting XML:

<article>
	<h1>...</h1>
	<author>...</author>
	<date>...</date>
	<nav>
		<h2>Table of Contents</h2>
		<ul>
			<li><a href="#section1">Section 1</a></li>
			<li><a href="#section2">Section 2</a></li>
		</ul>
	</nav>

	<section>
		<h2>Section 1</h2>
		<p>...</p>
		<aside>...</aside>
		<table>
			<tr>
				<th>...</th>
				<th>...</th>
			</tr>
			<tr>
				<td>...</td>
				<td>...</td>
			</tr>
		</table>
	</section>

	<section>
		<h2>Section 2</h2>
		<img src="..." alt="...">
		<p>...<a href="...">...</a></p>
		<p>...</p>
	</section>
</article>
	'''.strip()

	with user():
		lm += html

	with assistant():
		lm += special_token("<think>") + "\n\n"

		lm += "Exploring the document structure, extracting metadata, and formatting each part recursively.\n\n"
		lm += "Top-level sections to build the document <nav>:\n\n"
		sections = md_list(lm, style="numbered")
		lm += "\n".join(sections) + "\n\n"

		lm += "Extract or invent the title, date, and author for the document.\n"
		lm += "Title: <h1>"
		lm += gen(name="title", stop="</h1>")
		lm += "</h1>\n"
		title = lm["title"]
		lm += "Date: <date>"
		lm += gen(name="date", stop="</date>")
		lm += "</date>\n"
		date = lm["date"]
		lm += "Author: <author>"
		lm += gen(name="author", stop="</author>")
		lm += "</author>\n\n"
		author = lm["author"]

		lm += "I'll use the metadata to frame an XML skeleton with <title>, <author>, <date>, and a <nav> listing each section. For every section, I'll keep its raw HTML inside a <section_html> tag for recursive analysis later.\n\n"
		lm += f"<article>\n\t<h1>{title}</h1>\n\t<author>{author}</author>\n\t<date>{date}</date>\n\t<nav>\n\t\t<h2>Table of Contents</h2>\n\t\t<ul>\n"
		for i, section in enumerate(sections):
			lm += f"\t\t\t<li><a href=\"#section_{i+1}\">{section.split('. ')[1]}</a></li>\n"
		lm += "\t\t</ul>\n\t</nav>\n\n"

		for i, section in enumerate(sections):
			name = section.split('. ')[1]
			lm += "\t<section>\n"
			lm += f"\t\t<h2 id=\"section_{i+1}\">{name}</h2>\n"
			lm += "\t\t<!-- Here I should process the following raw section html: <section_html> -->\n"
			lm += "\t\t<section_html>\n"
			lm += gen(name=f"section_{i+1}_raw", stop="</section_html>")
			lm += "\t\t</section_html>\n"
			section_raw = lm[f"section_{i+1}_raw"]
			lm += analize_section(lm, name, section_raw, 3, f"section_{i+1}")
			lm += "\t</section>\n\n"

		lm += "</article>\n\n"

		lm += gen(name="think", max_tokens=1024)
		lm += special_token("</think>")

		lm += gen(name="article")

	return (
		"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
		"<document-container xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "
		"xsi:noNamespaceSchemaLocation=\"caterpillar.xsd\">\n"
		f"{lm['article']}\n"
		"</document-container>"
	)

def html(path: str, mime: Optional[str]):
	document = converter(path)
	html, images = document.html, document.images

	return _parse_html(html)


if __name__ == "__main__":
	print(html(sys.argv[1], "application/pdf"))
