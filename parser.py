#!/usr/bin/env python3


from dataclasses import dataclass
from datetime import date
import sys
from turtle import title
from typing import Dict, Generator, List, Optional, Tuple, Union
import textwrap
from pydantic import BaseModel, Field, ConfigDict, field_validator
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from guidance.models import LlamaCpp
from guidance import system, user, assistant, gen,  special_token, select, sequence
from guidance import json as gjson
import csv
from chat_template import Qwen3ChatTemplate, md_list, thoughts
from llama_cpp import Llama


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
	document_class: List[str]

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

class Classifier:
	def __init__(self, value: str):
		self.value = value
		self.children: List['Classifier'] = []

	def _children(self, path: List[str]):
		if not path:
			return self.children
		for child in self.children:
			if child.value == path[0]:
				return child._children(path[1:])
		return []

	def _append_child(self, path: List[str], value: str = "") -> bool:
		# Удаляем пустые элементы пути по краям и в середине
		path = [p for p in path if p]  # фильтруем '', None

		if not path:
			return True

		# Текущий узел должен совпадать с первым элементом пути
		if self.value != path[0]:
			return False

		# Если путь заканчивается на текущем узле — ничего добавлять не надо
		if len(path) == 1:
			return True

		# Дальше идём к/создаём узел со значением следующего уровня
		next_key = path[1]

		# Ищем уже существующего ребёнка
		for child in self.children:
			if child.value == next_key:
				return child._append_child(path[1:], value)

		# Не найден — создаём
		new_child = Classifier(next_key)
		self.children.append(new_child)
		return new_child._append_child(path[1:], value)

	def to_str(self, indent=0):
		print('\t' * indent + self.value, '\n')
		for child in self.children:
			child.to_str(indent + 1)

	@staticmethod
	def from_csv(catalog='corporate_classifier.csv') -> 'Classifier':
		root = Classifier('.')
		with open(catalog, 'r', encoding='utf-8') as f:
			reader = csv.reader(f)
			next(reader)  # пропустить заголовок
			for level1, level2, level3, level4, level5, description in reader:
				# Строим путь из непустых уровней; точка — явный корень
				path = ['.', level1, level2, level3, level4, level5]
				path = [p for p in path if p]  # убираем пустые уровни
				root._append_child(path)       # description НЕ используем для структуры
		return root

	def get_next_variants(self, path: List[str]) -> List[str]:
		"""
		Возвращает варианты следующего уровня для узла, заданного path.
		path может начинаться с '.' или сразу с уровня 1.
		"""
		# нормализуем путь: уберём пустые и ведущую точку
		path = [p for p in path if p and p != '.']

		# Спускаемся по дереву согласно path
		node = self
		for key in path:
			for child in node.children:
				if child.value == key:
					node = child
					break
			else:
				return []  # не нашли указанный путь

		# Возвращаем детей найденного узла
		return [child.value for child in node.children]

class TablePtr(BaseModel):
	caption: str
	raw_html: str

class SectionPtr(BaseModel):
	heading: str
	raw_html: str
	base_level: int

class ExtractResult(BaseModel):
	"""Что возвращает LLM для одного раздела: оболочка + сырые подглавы + сырые таблицы."""
	heading: str
	base_level: int
	xml_shell: str
	child_blocks: List[SectionPtr] = []
	tables: List[TablePtr] = []

class Section(BaseModel):
	"""Итоговый узел дерева: оболочка + таблицы + рекурсивно распарсенные дети."""
	heading: str
	base_level: int
	xml_shell: str
	tables: List[TablePtr] = []
	children: List["Section"] = []


CLASSIFIER = Classifier.from_csv()

def classify_document(llm, html: str) -> Tuple[str]:
	with system():
		llm += textwrap.dedent(f'''
			You are a document classifier. Check the following document against the given category.
			Use 'other' only if the document doesn't fit any existing category.

			Document MetaInfo: 

			Document: 
			
			```
				{html}
			```
		''')

	suffix = ["."]

	while True:
		variants = CLASSIFIER.get_next_variants(suffix)
		if not variants:
			break

		variants = ['/'.join(suffix + [variant]) for variant in variants] + ['other']

		tlm = llm
		with user():
			tlm += "Choose the most appropriate category: " + '\n'.join(variants) + '\n\n'
			tlm += "Category: "
		with assistant():
			tlm += special_token("<think>") + '\n'
			tlm += gen("thoughts")
			tlm += special_token("</think>") + '\n'
			tlm += select(name="category", options=variants)
		selected = tlm['category'].split('/')[-1].strip()
		if selected == 'other':
			break
		suffix.append(selected)

	return suffix[1:]

def extract_metainfo(llm, document_class: str, html: str) -> Header:
	with system():
		llm += textwrap.dedent(f'''
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
			{{
				"title": string,
				"authors": string[], // personal names, order preserved
				"date": string|null, // ISO YYYY-MM-DD or null
				"language": string,  // BCP-47 or null
				"keywords": string[], // deduplicate and trim
				"summary": string,   // 1–3 sentences
				"document_class": [{', '.join([f'"{v}"' for v in document_class])}],
			}}
		'''.strip())
	with user():
		llm += html
	with assistant():
		llm += special_token("<think>") + "\n"
		llm += gen("thoughts")
		llm += special_token("</think>") + '\n'
		llm += gjson(name="header", schema=Header, max_tokens=1024)

	return Header.model_validate_json(llm['header'])

def normalize_table_stub(table: TablePtr) -> TablePtr:
	"""Пока просто возвращает исходный HTML (на будущее подключите реальный нормализатор)."""
	return ""

def extract_sections(lm, section: SectionPtr, *, max_json_tokens: int = 2048) -> ExtractResult:
	"""
	Выполняет независимый прогон LLM для одного раздела.
	lm — объект модели (например, LlamaCpp/OpenAI), НЕ «накапливаем» состояние.
	"""
	llm = lm
	with system():
		llm += textwrap.dedent(f"""
				You task is to split a given raw HTML into its logical section and (if present) child subsections. 
				You also produce the section's own XML shell content WITHOUT embedding the child sections, and you extract raw tables for a dedicated table-normalizer.
				Raw HTML structure more visual: heading levels not reliable, some hidings may be omitted or wrong. So it must be switched to a logical structure at first.
				The current block has the following heading: `<h{section.base_level}>{section.heading}</h{section.base_level}>`.
				This section is a chapter or article. Current block may contains subsections.

				Rules:
				- base_level is the heading depth to use here (h{section.base_level}). Children must use h{min(section.base_level + 1, 6)}.
				- In the shell XML for THIS block:
					- Wrap text into <p>; move unrelated side notes to <aside> (with its own heading).
					- Keep links as <a>, images as <img>.
					- Strip purely visual tags like <br>, preserve only logical structure of the block.
					- DO NOT include child subsection bodies; DO include any inline content that belongs to THIS block only.
					- Replace each original HTML <table> within THIS block (not <table> in child sections) with a placeholder: <table data-source-id="<appropriate table caption>" />.
				- Child subsections must be returned as RAW HTML fragments, each with a heading, for recursive processing.
				- You MAY correct minor typos. Do not invent facts.
				- Output ONLY the JSON object defined below. No extra text or code fences.

				Output JSON schema:
				{{
					"heading": string,
					"base_level": {section.base_level},
					"xml_shell": string,
					"child_blocks": [
						{{
							"heading": string,
							"raw_html": string, // escaped string
							"base_level": {min(section.base_level + 1, 6)}
						}}
					],
					"tables": [
						{{
							"caption": string,
							"raw_html": string
						}}
					]
				}}
			""").strip()

	with user():
		llm += section.raw_html

	with assistant():
		# Мыслительный блок: отдельная генерация, затем JSON.
		llm += special_token("<think>") + "\n"
		llm += gen("thoughts")
		llm += special_token("</think>") + "\n"

		# Строго структурированный JSON
		llm += gjson(name="section", schema=ExtractResult, max_tokens=max_json_tokens)

	print(llm["section"])
	return ExtractResult.model_validate_json(llm["section"])

def parse_section_recursive(
	llm,
	section_ptr: SectionPtr,
	*,
	max_depth: int = 32,
	_depth: int = 0
) -> Section:
	lm = llm
	if _depth >= max_depth:
		# Жёсткая защита от зацикливания/глубины
		return Section(
			heading=section_ptr.heading,
			base_level=section_ptr.base_level,
			xml_shell=f"<section><h{section_ptr.base_level}>{section_ptr.heading}</h{section_ptr.base_level}><p>[depth limit]</p></section>",
			tables=[],
			children=[]
		)

	# 1) LLM-разбор текущей главы
	result = extract_sections(lm, section_ptr)

	# 2) Заглушечная нормализация таблиц
	normalized_tables = [normalize_table_stub(t) for t in result.tables]

	# 3) Рекурсия по подглавам
	children: List[Section] = []
	for child_ptr in result.child_blocks:
		# Перепроверим уровень
		next_level = min(section_ptr.base_level + 1, 6)
		child_ptr = SectionPtr(
			heading=child_ptr.heading,
			raw_html=child_ptr.raw_html,
			base_level=next_level
		)
		child_node = parse_section_recursive(lm, child_ptr, max_depth=max_depth, _depth=_depth + 1)
		children.append(child_node)

	# 4) Собираем дерево
	return Section(
		heading=result.heading,
		base_level=result.base_level,
		xml_shell=result.xml_shell,
		tables=normalized_tables,
		children=children
	)



def chunk_text(text: str, tok, max_chunk_size: int = 25_000) -> Generator[str, None, None]:
	if max_chunk_size <= 0:
		raise ValueError("max_chunk_size must be > 0")
	if not hasattr(tok, "tokenize") or not hasattr(tok, "detokenize"):
		raise TypeError("tok must иметь методы tokenize(...) и detokenize(...) как у llama_cpp.Llama")

	# Токенизируем текст (без BOS, со спецтокенами)
	token_ids = tok.tokenize(text.encode("utf-8"), add_bos=False, special=True)
	n = len(token_ids)
	if n == 0:
		return  # пустой генератор

	# Идём блоками по max_chunk_size токенов и детокенизируем каждый блок
	for start in range(0, n, max_chunk_size):
		end = min(start + max_chunk_size, n)
		chunk_ids = token_ids[start:end]
		# detokenize -> bytes; декодируем в UTF-8
		chunk_text_bytes = tok.detokenize(chunk_ids)
		# В норме это корректный UTF-8; используем strict, чтобы сохранить точность
		chunk_str = chunk_text_bytes.decode("utf-8", errors="strict")
		yield chunk_str

def process_chunk(lm, chunk: str) -> str:
	document_class = classify_document(lm, chunk)
	header = extract_metainfo(lm, document_class, chunk)
	root_ptr = SectionPtr(heading=header.title, raw_html=chunk, base_level=1)
	section = parse_section_recursive(lm, root_ptr)

	return header, section

def html(path: str, mime: Optional[str]):
	document = converter(path)
	html, images = document.html, document.images

	lm = LlamaCpp(model="models/Qwen3-4B-Thinking-2507-F16.gguf", 
			chat_template=Qwen3ChatTemplate,
			n_ctx=45000, 
			echo=True,
			n_gpu_layers=-1)
	tok = Llama(model_path="models/Qwen3-4B-Thinking-2507-F16.gguf", vocab_only=True, verbose=False)

	header, section = process_chunk(lm, html)
	print(header)
	print(section)

if __name__ == "__main__":
	print(html(sys.argv[1], "application/pdf"))
