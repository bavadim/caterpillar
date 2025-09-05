#! /usr/bin/env python3

from contextlib import contextmanager
from guidance import guidance
from guidance import system, user, assistant, gen,  special_token, select
from guidance import json as gjson
from guidance.chat import ChatTemplate, UnsupportedRoleException
from guidance.models import LlamaCpp
from typing import Any, Annotated, Literal, Union, get_origin, get_args

import inspect, json, typing
from dataclasses import dataclass
from typing import Any, get_origin

from pydantic import BaseModel, TypeAdapter, create_model, ConfigDict, Field
from pydantic.types import StrictInt, StrictStr, StrictFloat, StrictBool

QWEN3_TMPL = """{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = ((content.split('</think>')|first).rstrip('\n').split('<think>')|last).lstrip('\n') %}
                {%- set content = (content.split('</think>')|last).lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}
Using chat eos_token: <|im_end|>
Using chat bos_token: ,
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = ((content.split('</think>')|first).rstrip('\n').split('<think>')|last).lstrip('\n') %}
                {%- set content = (content.split('</think>')|last).lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}"""

class Qwen3ChatTemplate(ChatTemplate):
    template_str = QWEN3_TMPL

    # Map roles to the literal strings Qwen3 expects
    def get_role_start(self, role_name: str) -> str:
        if role_name == "system":
            return "<|im_start|>system\n"
        if role_name == "user":
            return "<|im_start|>user\n"
        if role_name == "assistant":
            return "<|im_start|>assistant\n"
        if role_name == "tool":
            # Qwen3 wraps tool responses inside a 'user' turn
            return "<|im_start|>user\n"
        raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name: str | None = None) -> str:
        return "<|im_end|>"

@dataclass
class ToolSpec:
    name: str
    description: str
    Params: type[BaseModel]
    fn: typing.Callable

class Tools:
    def __init__(self, *, strict_primitives: bool = False):
        self._by_name: dict[str, ToolSpec] = {}
        self._strict_primitives = strict_primitives

    def _strictify(self, annot: Any) -> Any:
        """Optionally map primitives to Strict* types (no coercion)."""
        if not self._strict_primitives:
            return annot

        origin = get_origin(annot)
        if origin is None:
            return {
                int: StrictInt,
                str: StrictStr,
                float: StrictFloat,
                bool: StrictBool,
            }.get(annot, annot)

        # Recurse for common containers / unions
        args = tuple(self._strictify(a) for a in get_args(annot))
        if origin in (list, typing.List):
            return list[args[0] if args else Any]
        if origin in (set, typing.Set):
            return set[args[0] if args else Any]
        if origin in (tuple, typing.Tuple):
            return tuple[args] if args else tuple[Any, ...]
        if origin in (dict, typing.Dict):
            k = args[0] if args else Any
            v = args[1] if len(args) > 1 else Any
            return dict[k, v]
        if origin is Union:
            return Union[args]  # handles Optional[T] too
        return annot

    class _ForbidExtra(BaseModel):
        model_config = ConfigDict(extra='forbid')  # no unknown keys anywhere

    def _register(self, *functions: typing.Callable, description: str | None = None, name: str | None = None):
        for fn in functions:
            sig = inspect.signature(fn)
            fields: dict[str, tuple[Any, Any]] = {}
            for name, p in sig.parameters.items():
                if p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                    # **kwargs / *args are not representable in JSON
                    continue
                annot = p.annotation if p.annotation is not inspect._empty else Any
                annot = self._strictify(annot)
                default = p.default if p.default is not inspect._empty else ...
                fields[name] = (annot, default)

            tool_name = name or fn.__name__
            desc = (description or fn.__doc__ or tool_name).strip()
            Params = create_model(f"{fn.__name__}Params", __base__=self._ForbidExtra, **fields)
            self._by_name[tool_name] = ToolSpec(tool_name, desc, Params, fn)

    def system_preface(self) -> str:

        defs = []
        for spec in self._by_name.values():
            defs.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.Params.model_json_schema(),
                },
            })
        tools = "<tools>\n" + "\n".join(json.dumps(d, ensure_ascii=False) for d in defs) + "\n</tools>"

        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"{tools}\n\n"
            "For each function call, return a JSON object with function name and arguments within <tool_call> tags:\n"
            "<tool_call>\n"
            "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
            "</tool_call>"
        )

    def execute(self, name: str, arguments: dict) -> Any:
        """Validate with Pydantic, then call the Python function."""
        if name not in self._by_name:
            raise ValueError(f"Unknown tool: {name!r}")
        spec = self._by_name[name]
        args = spec.Params.model_validate(arguments).model_dump()
        return spec.fn(**args)

    # Decorator form
    def tool(self, *dargs, **dkwargs):
        def wrap(fn):
            self._register(fn, **dkwargs)
            return fn
        if dargs and callable(dargs[0]) and not dkwargs:
            return wrap(dargs[0])
        return wrap

    def _tool_call_union_type(self) -> Any:
        """Build a discriminated union over name -> arguments schema."""
        if not self._by_name:
            raise ValueError("No tools registered")

        call_models = []
        for spec in self._by_name.values():
            CallModel = create_model(
                f"{spec.name}Call",
                __base__=self._ForbidExtra,
                # The discriminator; only this literal is allowed:
                name=(Literal[spec.name], ...),
                # Arguments must match this tool's Params exactly:
                arguments=(spec.Params, ...),
            )
            call_models.append(CallModel)

        # Discriminated union keyed by "name"
        return TypeAdapter(Annotated[Union[tuple(call_models)], Field(discriminator="name")])

    # Guidance template: now takes a schema argument
    @guidance(stateless=True)
    def tool_call(self, lm, var_name):
        """Emit: <tool_call>...</tool_call> for a tool call."""
        lm += special_token("<tool_call>") + "\n"
        lm += gjson(name=var_name, schema=self._tool_call_union_type())
        lm += special_token("</tool_call>")
        return lm

@contextmanager
def thoughts(lm):
    lm += special_token("<think>") + "\n"
    yield lm
    lm += special_token("</think>") + "\n\n"

LINE_OR_BLANK = r"(?:- [^\n]{1,160}\n|\n)"


def md_list(lm, style: str = "bullet") -> list[str]:
    """
    Generate a markdown list directly into `lm`, stopping when the model emits a blank line.
    Returns a Python list[str] of the captured items (without the prefix).

    style: "bullet"  -> lines like "- item\n"
           "numbered"-> lines like "1. item\n", "2. item\n", ...

    Usage:
        lm += "List 3–5 concise tips, then end with a blank line:\n"
        tips = md_list(lm, style="bullet")
    """
    if style not in ("bullet", "numbered"):
        raise ValueError("style must be 'bullet' or 'numbered'")

    items: list[str] = []
    MAX_ITEMS = 64  

    for i in range(1, MAX_ITEMS):
        if style == "bullet":
            pattern = r"(?:- [^\n]+\n|\n)"
        else:
            pattern = rf"(?:{i}\. [^\n]+\n|\n)"

        name = f"mdline{i}x"
        lm += gen(name=name, regex=pattern, max_tokens=160)
        line = lm[name]

        if line == "\n":
            break

        # Strip trailing newline and the visible prefix
        line_no_nl = line[:-1]  # drop '\n'
        if style == "bullet":
            text = line_no_nl[2:] if line_no_nl.startswith("- ") else line_no_nl
        else:
            pref = f"{i+1}. "
            text = line_no_nl[len(pref):] if line_no_nl.startswith(pref) else line_no_nl

        items.append(text.strip())

    return items

if __name__ == "__main__":
    lm = LlamaCpp(model="models/Qwen3-4B-Thinking-2507-F16.gguf", 
            chat_template=Qwen3ChatTemplate,
            n_ctx=31000, 
            echo=True,
            n_gpu_layers=-1)

    tools = Tools()
    @tools.tool(description="Multiply two integers")
    def multiply(a1: int, a2: int) -> int:
        return a1 * a2

    @tools.tool(description="Get current time for a timezone")
    def get_time(timezone: str) -> str:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")

    with system():
        lm += "You are a helpful assistant.\n\n" + tools.system_preface()

    with user():
        lm += "Multiply 2 and 2"

    with assistant():
        lm += thoughts()
        lm += tools.tool_call(var_name="tool_args")
        #lm += gen(name="answer")
    print(lm["thoughts"])
    #print(lm["answer"])
    print(lm["tool_args"])

