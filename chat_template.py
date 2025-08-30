#! /usr/bin/env python3

import json as pyjson
import guidance
from guidance import system, user, assistant, gen,  special_token, select
from guidance import json as gjson
from guidance.chat import ChatTemplate, UnsupportedRoleException
from dataclasses import is_dataclass, asdict
from pydantic import BaseModel
import llama_cpp
from guidance.models import LlamaCpp
from typing import List, Literal

import inspect, json, typing
from dataclasses import dataclass
from typing import Any, get_origin

from pydantic import BaseModel, create_model
from pydantic import TypeAdapter

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
    def __init__(self):
        self._by_name: dict[str, ToolSpec] = {}

    def _model_from_fn(self, fn: typing.Callable) -> type[BaseModel]:
        """Build a Pydantic model from function annotations (v2)."""
        sig = inspect.signature(fn)
        fields: dict[str, tuple[Any, Any]] = {}
        for name, p in sig.parameters.items():
            if p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                continue  # keep it representable as JSON
            annot = p.annotation if p.annotation is not inspect._empty else Any
            default = p.default if p.default is not inspect._empty else ...
            fields[name] = (annot, default)
        return create_model(f"{fn.__name__}Params", **fields)  # type: ignore

    def register(self, *functions: typing.Callable, description: str | None = None, name: str | None = None):
        for fn in functions:
            tool_name = name or fn.__name__
            desc = (description or fn.__doc__ or tool_name).strip()
            Params = self._model_from_fn(fn)
            self._by_name[tool_name] = ToolSpec(tool_name, desc, Params, fn)

    # Decorator form
    def tool(self, *dargs, **dkwargs):
        def wrap(fn):
            self.register(fn, **dkwargs)
            return fn
        if dargs and callable(dargs[0]) and not dkwargs:
            return wrap(dargs[0])
        return wrap

    # Qwen3-style <tools> block for the system prompt
    def tools_block(self) -> str:
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
        return "<tools>\n" + "\n".join(json.dumps(d, ensure_ascii=False) for d in defs) + "\n</tools>"

    def system_preface(self) -> str:
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"{self.tools_block()}\n\n"
            "For each function call, return a JSON object with function name and arguments within <tool_call> tags:\n"
            "<tool_call>\n"
            "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
            "</tool_call>"
        )

    def spec_for(self, fn_or_name: str | typing.Callable) -> ToolSpec:
        return self._by_name[fn_or_name if isinstance(fn_or_name, str) else fn_or_name.__name__]

    def execute(self, name: str, arguments: dict) -> Any:
        """Validate with Pydantic, then call the Python function."""
        spec = self._by_name[name]
        args = spec.Params.model_validate(arguments).model_dump()
        return spec.fn(**args)

@guidance(stateless=True)
def thoughts(lm, var_name="thoughts"):
    """Emit: <think>...</think> for a tool call."""
    lm += special_token("<think>") + '\n'
    lm += gen(name=var_name)
    lm += special_token("</think>") + '\n\n'
    return lm

class FnArgs(BaseModel):
    """Model for function arguments."""
    name: str
    arguments: List[dict[str, Any]]

@guidance(stateless=True)
def tool(lm, var_name="tool_args"):
    """Emit: <tool_call>...</tool_call> for a tool call."""
    lm += special_token("<tool_call>") + '\n'
    lm += gjson(name=var_name, schema=TypeAdapter(FnArgs))
    lm += special_token("</tool_call>")
    return lm

if __name__ == "__main__":
    lm = LlamaCpp(model="models/Qwen3-4B-Thinking-2507-F16.gguf", 
            chat_template=Qwen3ChatTemplate,
            n_ctx=31000, 
            echo=True,
            n_gpu_layers=-1)

    tools = Tools()
    @tools.tool(description="Multiply two integers")
    def multiply(a: int, b: int) -> int:
        return a * b

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
        lm += select([gen(name="answer"), tool(var_name="tool_args")])
        #lm += gen(name="answer")
    print(lm["thoughts"])
    print(lm["answer"])
    print(lm["tool_args"])

