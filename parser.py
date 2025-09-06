#!/usr/bin/env python3


from dataclasses import dataclass
import sys
from turtle import title
from typing import Dict, List, Optional, Tuple, Union
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from guidance.models import LlamaCpp
from guidance import system, user, assistant, gen,  special_token, select

from chat_template import Qwen3ChatTemplate, md_list, thoughts


@dataclass
class Metainfo:
	title: str
	date: str
	author: str

@dataclass
class Paragraph:
	content: str

@dataclass
class Table:
	headers: List[str]
	rows: List[Dict[str, str]]

@dataclass
class ImgRef:
	target: str
	name: str

@dataclass
class Ref:
	target: str
	name: str

@dataclass
class Nav:
	items: List[Ref]

@dataclass
class Article:
	nav: Optional[Nav]
	metainfo: Metainfo
	content: List[Union[Paragraph, Table, ImgRef]]


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

def map_sections(llm, section_name: str) -> Tuple[Metainfo, List[str]]:
	lm = llm

	lm += f"Now, I will extract chapters/subsections of the {section_name}. "
	lm += "I pay attention to the size of extracted content and make separate section for each sensitive block:\n\n"
	sections = md_list(lm, style="numbered")
	lm += "\n".join(sections) + "\n\n"

	lm += f"Now, I'll extract the title, date, and author for the {section_name}.\n"
	lm += "Title: <title>"
	lm += gen(name="title", stop="</title>")
	lm += "</title>\n"
	title = lm["title"]
	lm += "Date: <date>"
	lm += gen(name="date", stop="</date>")
	lm += "</date>\n"
	date = lm["date"]
	lm += "Author: <author>"
	lm += gen(name="author", stop="</author>")
	lm += "</author>\n\n"
	author = lm["author"]

	return Metainfo(title=title, date=date, author=author), sections

def parse_content_block(llm, section: str) -> str:
	#TODO
	return section

def reduce_sections(sections: List[str]) -> str:
	#TODO
	return "\n\n".join(sections)

def analyze_section(llm, section_name: str, level=0) -> str:
	lm = llm
	lm += f"Now, I'll explore '{section_name}' structure and determinate potential sections. "
	lm += f"I see '{section_name}' " + select([
		"has Subsections:\n\n",
		"has No subsections.\n\n"
	], name=f"{section_name}_subs_choice")

	if lm[f"{section_name}_subs_choice"] == "has Subsections:\n\n":
		_, sections = map_sections(lm, f"{section_name}")
		subsections = []
		for sub in sections:
			_content = analyze_section(llm, f"{section_name}/{sub}")
			subsections.append(_content)
		content = reduce_sections(subsections)
	else:
		content = parse_content_block(llm, section_name)

	return content

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
* Use `<date>`, `<title>` and `<author>` tags for document metadata.

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

		lm += "First, I'll explore the document structure, extract metadata, and formatting each chapter.\n\n"
		title, date, author, sections = extract_metainfo(lm, "document")
		lm += f'I see document has the following props: Title: "{title}"; Date: "{date}"; Author: "{author}"; Sections:\n{"\n\t".join(sections)}\n'
		
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
			lm += analyze_section(lm, name, section_raw, 3, f"section_{i+1}")
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
