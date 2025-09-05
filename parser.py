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
		lm += special_token("<think>") + '\n\n'

		lm += "Okay, I need to format this HTML into XML so that it looks like a book and is easy to read. "
		lm += "To begin, I'll look at the original HTML and think about how to title the document and structure it.\n\n"
		lm += "First, I'll think about how to split the document into chapters. The document includes the following chapters:\n\n"
		sections = md_list(lm, style="numbered")
		lm += '\n'.join(sections) + '\n\n'

		lm += "Now, based on the document and the table of contents, I'll come up with a title for the document. "
		lm += "Based on the table of contents, the following title can be proposed for the entire document: <h1>"
		lm += gen(name="title", stop="</h1>")
		lm += "</h1>. "
		title = lm["title"]

		lm += "Now I can create a skeleton of the XML document that includes the title, the table of contents, and placeholders for the chapters:\n\n"
		lm += f"<article>\n\t<h1>{title}</h1>\n\t<nav>\n\t\t<h2>Table of Contents</h2>\n\t\t<ul>\n"
		for i, section in enumerate(sections):
			lm += f"			<li><a href=\"#section_{i+1}\">{section.split('. ')[1]}</a></li>\n"
		lm += "		</ul>\n	</nav>\n\n"

		for i, section in enumerate(sections):
			lm += "	<section>\n"
			lm += f"		<h2 id=\"section_{i+1}\">{section.split('. ')[1]}</h2>\n"
			lm += f'		<!-- chapter content "{section}" -->\n'
			lm += "	</section>\n\n"

		lm += "</article>\n\n"

		lm += "Now I have the overall structure of the XML response. Next, I will fill in the content of each chapter, fix typos and formatting, and add any missing elements.\n\n"

		lm += gen(name="think", max_tokens=1024)
		lm += special_token("</think>")

		lm += gen(name="result")

	print(lm)
	return lm['result']

def html(path: str, mime: Optional[str]):
	document = converter(path)
	html, images = document.html, document.images

	print(_parse_html(html))


if __name__ == "__main__":
	html(sys.argv[1], "application/pdf")