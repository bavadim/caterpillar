#!/usr/bin/env python3


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
Твоя задача отформатировать заданый html так, чтобы он стал похож на книгу и был удобен в чтении. Выведи результат в XML, который должен:

- отображать логическую структуру документа: главы, карточки, метаинформация - это отдельные блоки.
- быть свободен от визуальных инструкций, таких как <br>.
- уровень заголовков должен определяться структурой документа, а не визуальным размером текста.
- иметь структуру, похожей на книгу: документ, глава, разделы, параграфы.
- вся информация (кроме визуальной/эстетической) из оригинального HTML должна быть сохранена и правильно структурирована в итоговом XML.

Результирующий XML должен использовать следующие тэги:
- информацию оформи только в блоки <section> (для разделов и глав), <article> (для обозначения отдельной статьи), <aside> (для обозначения не имеющих к текущему блоку сведений).
- каждый <section>, <aside>, <article> должен включать тэги заголовков <h1>, <h2>, <h3>, <h4>, <h5>, <h6>. Если нужно - создай заголовок.
- каждый <article> должен включать оглавление в <nav>, если нужно - создай это оглавление.
- текст должен быть оформлен тэгами <p>.
- ссылки должны быть оформлены тэгами <a>.
- таблицы должны быть оформлены тэгами <table>, <tr>, <th>, <td>.
- изображения должны быть оформлены тэгами <img>.

Пример результирующего XML:

<article>
    <h1>...</h1>
    <nav>
        <h2>Оглавление</h2>
        <aside>...</aside>
        <ul>
            <li><a href="#section1">Раздел 1</a></li>
            <li><a href="#section2">Раздел 2</a></li>
        </ul>
        <p>...</p>
    </nav>

    <section>
        <h2>Раздел 1</h2>
        <p>...</p>
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
        <h2>Раздел 2</h2>
        <img src="..." alt="...">
        <p>...<a href="...">...</a></p>
        <p>...</p>
    </section>
</article>
    '''.strip()

    with user():
        lm += html

    with assistant():
        with thoughts(lm) as llm:
            llm += "Хорошо, мне нужно отформатировать этот HTML в XML, который будет похож на книгу и удобен в чтении. Давайте посмотрю на исходный HTML и подумаю, как его структурировать.\n\n"
            llm += "Сначала посмотрю на структуру документа. Документ можно разбить на следующие разделы или главы:\n"
            llm, sections = md_list(llm, name="sections")
            llm += "Теперь можно сформировать оглавение:\n\nxml```\n<nav>\n"
            llm += gen(name="nav", stop="```")
            llm += "```"
            llm += gen(name="thoughts")
            #llm += "Теперь оформлю XML в правильном формате."
        #lm += "Ответ:"
        #lm += gen(name="answer")

    print(sections)
    print()
    print(llm["nav"])

    print(llm["thoughts"])
    return html

def html(path: str, mime: Optional[str]):
    document = converter(path)
    html, images = document.html, document.images

    _parse_html(html)


if __name__ == "__main__":
    html("test_files/Doczilla Pro - Требования к ПАК.pdf", "application/pdf")