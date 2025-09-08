#!/usr/bin/env python3


from dataclasses import dataclass
from datetime import date
import sys
from turtle import title
from typing import Dict, List, Optional, Tuple, Union
import textwrap
from pydantic import BaseModel, Field, ConfigDict, field_validator
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from guidance.models import LlamaCpp
from guidance import system, user, assistant, gen,  special_token, select, sequence
from guidance import json as gjson

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

class Header(BaseModel):
	title: str
	authors: List[str]
	date: Optional[date]
	language: str
	keywords: List[str]
	summary: str

	@field_validator("authors", "keywords", mode="after")
	@classmethod
	def _strip_and_dedup(cls, items: List[str]) -> List[str]:
		seen = set()
		out = []
		for x in items:
			s = x.strip()
			k = s.lower()
			if s and k not in seen:
				seen.add(k)
				out.append(s)
		return out

def extract_metainfo(llm, html: str) -> Tuple[Header, str]:
	with system():
		llm += textwrap.dedent('''
			You are a metadata extraction tool. From the provided HTML, extract global document-level metadata to JSON.
			Follow these rules:

			- Prefer information in explicit metadata: <title>, <meta name="author">, <meta property="article:author">,
	<meta name="dc.date">, <time>, <meta property="og:locale">, etc.
			- If multiple candidates exist, choose the most specific and document-representative values.
			- Normalize the date to ISO 8601 YYYY-MM-DD when possible; otherwise leave "date" null.
			- "language" as BCP-47 (e.g., "en", "ru", "en-US"); infer from content if not explicit.
			- You MAY fix typos but MUST NOT invent metadata that is not implied by the HTML.
			- Output ONLY a single JSON object, with no commentary, no code fences.
			- Tag the document with 2-5 the most specific to document nouns. This tags are used for searching.
			- write a short summary of the document. Summary should helps understand the main points of documents for fast searching. Highlight names and labels.
			- write the clean and concise title.
			- use the "language" field as an your output primary language.

			Output JSON schema (all fields required, use null or [] if unknown):
			{
				"title": string,
				"authors": string[], // personal names, order preserved
				"date": string|null, // ISO YYYY-MM-DD or null
				"language": string,  // BCP-47 or null
				"keywords": string[], // deduplicate and trim
				"summary": string,   // 1–3 sentences
			}
		'''.strip())
	with user():
		llm += html
	with assistant():
		llm += special_token("<think>") + "\n"
		llm += gen("thoughts", max_tokens=10000)
		llm += special_token("</think>") + '\n'
		llm += gjson(name="header", schema=Header, max_tokens=1024)

	return Header.model_validate_json(llm['header']), llm["thoughts"]

class TablePtr(BaseModel):
	caption: str
	raw_html: str

class SectionPtr(BaseModel):
	heading: str
	raw_html: str
	base_level: int

class Section(BaseModel):
	heading: str
	base_level: int
	xml_shell: str

	child_blocks: List[SectionPtr]
	tables: List[TablePtr]


def extract_sections(llm,section: SectionPtr) -> Tuple[Section, str]:
	with system():
		llm += textwrap.dedent(f'''
			You task is to split a given raw HTML into its logical section and (if present) child subsections. 
			You also produce the section's own XML shell content WITHOUT embedding the child sections, and you extract raw tables for a dedicated table-normalizer.
			Raw HTML structure more visual: heading levels not reliable, some hidings may be omitted or wrong. So it must be switched to a logical structure at first.
			The current block has the following heading: `<h{section.base_level}>{section.heading}</h{section.base_level}>`.
			This section is a chapter or article. Current block may contains subsections.

			Rules:
			- base_level is the heading depth to use here (h{section.base_level}). Children must use h{min(section.base_level+1, 6)}.
			- In the shell XML for THIS block:
				- Wrap text into <p>; move unrelated side notes to <aside> (with its own heading).
				- Keep links as <a>, images as <img>.
				- Strip purely visual tags like <br>, preserve only logical structure of the block.
				- DO NOT include child subsection bodies; DO include any inline content that belongs to THIS block only.
				- Replace each original HTML <table> within THIS block (not <table> in child sections) with a placeholder: <table data-source-id="<appropriate table caption>" />.
			- Child subsections must be returned as RAW HTML fragments, each with a heading, for recursive processing.
			- You MAY correct typos. Do not invent facts.
			- Output ONLY the JSON object defined below. No extra text or code fences.

			Output JSON schema:
			{{
				"heading": string,     // text heading for this block
				"base_level": {section.base_level},
				"xml_shell": string,   // XML string: <section> (or <article>/<aside>) with <h{section.base_level}> and content, but without child sections; table placeholders inserted
				"child_blocks": [
					{{
						"heading": string,  // child title
						"raw_html": string, // raw HTML for the child and its subsections
						"base_level": {section.base_level + 1}
					}}
				],
				"tables": [
					{{
						"caption": string,  // matches data-source-id in xml_shell, this is table caption/name
						"raw_html": string // the original <table> HTML
					}}
				]
			}}
		'''.strip())
	with user():
		llm += section.raw_html

	with assistant():
		llm += special_token("<think>") + "\n"
		llm += gen("thoughts", max_tokens=15000)
		llm += special_token("</think>") + '\n'
		#llm += gjson(name="sections", schema=Section, max_tokens=1024)
		llm += gen("sections", max_tokens=1024)
		print(llm['sections'])

	return Section.model_validate_json(llm['sections']), llm["thoughts"]

def html(path: str, mime: Optional[str]):
	document = converter(path)
	html, images = document.html, document.images

	lm = LlamaCpp(model="models/Qwen3-4B-Thinking-2507-F16.gguf", 
			chat_template=Qwen3ChatTemplate,
			n_ctx=31000, 
			echo=True,
			n_gpu_layers=-1)
	
	
	header, thoughts = extract_metainfo(lm, html)
	print(thoughts)
	print(header)
	print('----')
	section, thoughts = extract_sections(lm, SectionPtr(heading=header.title, raw_html=html, base_level=1))
	print(thoughts)
	print(section)
	print('----')


if __name__ == "__main__":
	print(html(sys.argv[1], "application/pdf"))
