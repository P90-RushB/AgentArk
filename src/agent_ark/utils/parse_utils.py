import json
import re


OBSERVATION_CONTEXT_TAGS = ('reset_context', 'step_context', 'observation_context')
OBSERVATION_CONTEXT_LABELS = {
    'reset_context': 'Reset context',
    'step_context': 'Step context',
    'observation_context': 'Observation context',
}

def extract_tag_content(text, tag):
    """
    提取字符串中 <tag> 到 </tag> 之间的内容。
    返回所有匹配内容的列表。
    """
    if not isinstance(text, str):
        return []
    pattern = fr'<{re.escape(tag)}\b[^>]*>(.*?)</{re.escape(tag)}>'
    return re.findall(pattern, text, re.IGNORECASE | re.DOTALL)


def _strip_tag_blocks(text: str, tag: str) -> str:
    """Remove all <tag>...</tag> blocks from text (case-insensitive)."""
    if not isinstance(text, str) or not text:
        return ''
    if not isinstance(tag, str) or not tag:
        return text
    pattern = fr'<{re.escape(tag)}\b[^>]*>.*?</{re.escape(tag)}>'
    return re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)


def _unwrap_tag_blocks(text: str, tag: str) -> str:
    """Replace <tag>...</tag> blocks with their inner text."""
    if not isinstance(text, str) or not text:
        return ''
    if not isinstance(tag, str) or not tag:
        return text
    pattern = fr'<{re.escape(tag)}\b[^>]*>(.*?)</{re.escape(tag)}>'
    return re.sub(pattern, lambda m: m.group(1).strip(), text, flags=re.IGNORECASE | re.DOTALL)


def strip_observation_context_blocks(text: str) -> str:
    """Remove dynamic observation context blocks from text."""
    if not isinstance(text, str) or not text:
        return ''
    for tag in OBSERVATION_CONTEXT_TAGS:
        text = _strip_tag_blocks(text, tag)
    return text


def extract_observation_context(text: str, tags=OBSERVATION_CONTEXT_TAGS, prefer_english_lang=True) -> str:
    """Extract dynamic observation context blocks as plain text."""
    if not isinstance(text, str) or not text:
        return ''

    blocks = []
    for tag in tags:
        for block in extract_tag_content(text, tag):
            block = _filter_language_blocks(block, prefer_english_lang=prefer_english_lang).strip()
            if block:
                blocks.append(block)
    return '\n\n'.join(blocks)


def unwrap_observation_context_blocks(text: str, prefer_english_lang=True, with_labels=True) -> str:
    """Replace dynamic observation context tags with LLM-visible text."""
    if not isinstance(text, str) or not text:
        return ''

    def replace(tag: str, body: str) -> str:
        body = _filter_language_blocks(body, prefer_english_lang=prefer_english_lang).strip()
        if not body:
            return ''
        if not with_labels:
            return body
        return f"{OBSERVATION_CONTEXT_LABELS.get(tag, 'Observation context')}:\n{body}"

    for tag in OBSERVATION_CONTEXT_TAGS:
        pattern = fr'<{re.escape(tag)}\b[^>]*>(.*?)</{re.escape(tag)}>'
        text = re.sub(
            pattern,
            lambda m, tag=tag: replace(tag, m.group(1)),
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return text


def _filter_language_blocks(text: str, prefer_english_lang=True) -> str:
    """Select one language block while preserving non-language siblings such as tool_docs."""
    if not isinstance(text, str) or not text:
        return ''

    preferred = 'english_ver' if prefer_english_lang else 'chinese_ver'
    fallback = 'chinese_ver' if prefer_english_lang else 'english_ver'
    has_preferred = bool(extract_tag_content(text, preferred))
    has_fallback = bool(extract_tag_content(text, fallback))
    if not has_preferred and not has_fallback:
        return text

    if has_preferred:
        text = _unwrap_tag_blocks(text, preferred)
        text = _strip_tag_blocks(text, fallback)
        return text.strip()

    return _unwrap_tag_blocks(text, fallback).strip()


def _first_or_none(items):
    if isinstance(items, list) and items:
        return items[0]
    return None

def parse_system_task_prompt(text, prefer_english_lang=True):
    """
    从字符串中提取 system_prompt 和 task_prompt。
    system_prompt 是 <system_prompt> 标签中的内容。
    task_prompt 是 <task_prompt> 标签中的内容。
    返回一个元组 (system_prompt, task_prompt)。
    """
    if not isinstance(text, str):
        return '', ''

    system_prompt = _first_or_none(extract_tag_content(text, 'system_prompt')) or ''

    # Prefer explicit <task_prompt>...</task_prompt> if present; otherwise fall back
    # to legacy behavior (everything after </system_prompt>).
    explicit_task = _first_or_none(extract_tag_content(text, 'task_prompt'))
    if isinstance(explicit_task, str):
        task_prompt = explicit_task
    else:
        end_tag = '</system_prompt>'
        cut = text.lower().find(end_tag)
        if cut >= 0:
            task_prompt = text[cut + len(end_tag):]
        else:
            task_prompt = text

    task_prompt = _filter_language_blocks(task_prompt, prefer_english_lang=prefer_english_lang)

    # Strip internal blocks: these are meant for post-processing, not the LLM task prompt.
    task_prompt = _strip_tag_blocks(task_prompt, 'code_wrapper').strip()
    task_prompt = _strip_tag_blocks(task_prompt, 'tool_manifest').strip()
    task_prompt = strip_observation_context_blocks(task_prompt).strip()

    return system_prompt.strip(), task_prompt


def parse_task_prompt_payload(text, prefer_english_lang=True, include_code_wrapper=False):
    """Parse the Unity side-channel task payload into structured prompt parts."""
    if not isinstance(text, str):
        return {
            'system_prompt': '',
            'task_prompt': '',
            'tool_manifest': None,
            'tool_docs': '',
            'code_wrapper': None,
            'reset_context': '',
            'step_context': '',
            'observation_context': '',
            'raw': '',
        }

    system_prompt = _first_or_none(extract_tag_content(text, 'system_prompt')) or ''
    explicit_task = _first_or_none(extract_tag_content(text, 'task_prompt'))
    if isinstance(explicit_task, str):
        task_prompt = explicit_task
    else:
        end_tag = '</system_prompt>'
        cut = text.lower().find(end_tag)
        task_prompt = text[cut + len(end_tag):] if cut >= 0 else text

    task_prompt = _filter_language_blocks(task_prompt, prefer_english_lang=prefer_english_lang)

    tool_docs = _first_or_none(extract_tag_content(task_prompt, 'tool_docs')) or ''
    manifest_text = _first_or_none(extract_tag_content(text, 'tool_manifest'))
    manifest = None
    if manifest_text:
        try:
            manifest = json.loads(manifest_text.strip())
        except Exception:
            manifest = None

    wrapper = _first_or_none(extract_tag_content(text, 'code_wrapper'))
    reset_context = extract_observation_context(
        text,
        tags=('reset_context',),
        prefer_english_lang=prefer_english_lang,
    )
    step_context = extract_observation_context(
        text,
        tags=('step_context',),
        prefer_english_lang=prefer_english_lang,
    )
    observation_context = extract_observation_context(
        text,
        tags=('observation_context',),
        prefer_english_lang=prefer_english_lang,
    )

    if not include_code_wrapper:
        task_prompt = _strip_tag_blocks(task_prompt, 'code_wrapper')
    task_prompt = _strip_tag_blocks(task_prompt, 'tool_manifest').strip()
    task_prompt = strip_observation_context_blocks(task_prompt).strip()
    if include_code_wrapper and isinstance(wrapper, str) and '<code_wrapper' not in task_prompt.lower():
        task_prompt = (
            task_prompt.rstrip()
            + '\n<code_wrapper>\n'
            + wrapper.strip('\n')
            + '\n</code_wrapper>'
        ).strip()

    return {
        'system_prompt': system_prompt.strip(),
        'task_prompt': task_prompt,
        'tool_manifest': manifest,
        'tool_docs': tool_docs.strip(),
        'code_wrapper': wrapper.strip('\n') if isinstance(wrapper, str) else None,
        'reset_context': reset_context,
        'step_context': step_context,
        'observation_context': observation_context,
        'raw': text,
    }


def build_llm_visible_prompt(text, system_prompt='', prefer_english_lang=True, include_code_wrapper=False):
    payload = parse_task_prompt_payload(
        text,
        prefer_english_lang=prefer_english_lang,
        include_code_wrapper=include_code_wrapper,
    )
    system = (system_prompt or payload.get('system_prompt') or '').strip()
    task = (payload.get('task_prompt') or '').strip()
    if system and task:
        return system + '\n' + task
    return system or task


def to_csharp_literal(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return '%s' % value
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(str(value), ensure_ascii=False)


def _tool_call_payload(action_text):
    raw = (action_text or '').strip()
    blocks = extract_tag_content(raw, tag='tool_call')
    if blocks:
        if len(blocks) != 1:
            raise ValueError('Expected exactly one <tool_call> block')
        raw = blocks[0].strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError('Tool call payload must be a JSON object')
    name = data.get('name') or data.get('tool_name')
    arguments = data.get('arguments', {})
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Tool call requires a non-empty name')
    if not isinstance(arguments, dict):
        raise ValueError('Tool call arguments must be a JSON object')
    return name.strip(), arguments, data


def _manifest_default_value(spec):
    raw = spec.get('default') if 'default' in spec else spec.get('default_value')
    if raw is None:
        return None
    if isinstance(raw, (bool, int, float)):
        return raw

    type_name = str(spec.get('type') or '').lower()
    text = str(raw).strip()
    try:
        if type_name in ('bool', 'boolean'):
            if text.lower() in ('true', '1'):
                return True
            if text.lower() in ('false', '0'):
                return False
        if type_name in ('int', 'integer', 'int32', 'long', 'int64'):
            return int(text)
        if type_name in ('float', 'single', 'double'):
            return float(text)
    except Exception:
        return raw
    return raw


def _enum_values(spec):
    values = spec.get('enum') if isinstance(spec.get('enum'), list) else spec.get('enum_values')
    return values if isinstance(values, list) else []


def _validate_argument_value(arg_name, value, spec):
    allowed = _enum_values(spec)
    if allowed and str(value) not in {str(item) for item in allowed}:
        raise ValueError(f'Argument {arg_name} must be one of: {", ".join(map(str, allowed))}')

    pattern = spec.get('pattern')
    if pattern and isinstance(value, str) and re.fullmatch(pattern, value) is None:
        raise ValueError(f'Argument {arg_name} does not match required pattern')

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = spec.get('minimum')
        maximum = spec.get('maximum')
        try:
            if minimum is not None and value < float(minimum):
                raise ValueError(f'Argument {arg_name} must be >= {minimum}')
            if maximum is not None and value > float(maximum):
                raise ValueError(f'Argument {arg_name} must be <= {maximum}')
        except ValueError:
            raise
        except Exception:
            pass


def render_tool_call_to_csharp(action_text, tool_manifest, class_name='ArkAct_Step0'):
    """Render a validated AgentArk tool_call into a small ActRouter C# script."""
    if not isinstance(tool_manifest, dict):
        raise ValueError('Missing tool manifest for <tool_call> rendering')
    tools = tool_manifest.get('tools')
    if not isinstance(tools, list):
        raise ValueError('Tool manifest must contain a tools list')

    name, arguments, payload = _tool_call_payload(action_text)
    tool = next((item for item in tools if isinstance(item, dict) and item.get('name') == name), None)
    if tool is None:
        raise ValueError(f'Unknown tool: {name}')

    kind = str(tool.get('kind') or 'method').lower()
    access = str(tool.get('access') or '')
    specs = tool.get('arguments') if isinstance(tool.get('arguments'), list) else []
    known_args = {spec.get('name') for spec in specs if isinstance(spec, dict) and spec.get('name')}
    extra_args = set(arguments) - known_args
    if extra_args:
        raise ValueError(f'Unexpected argument(s): {", ".join(sorted(extra_args))}')

    for spec in specs:
        if not isinstance(spec, dict):
            continue
        arg_name = spec.get('name')
        if spec.get('required') and arg_name not in arguments:
            raise ValueError(f'Required argument missing: {arg_name}')
        if arg_name in arguments:
            _validate_argument_value(arg_name, arguments[arg_name], spec)

    statement = None
    if kind == 'property':
        if 'set' in access and 'value' in arguments:
            statement = f'router.Set({to_csharp_literal(name)}, {to_csharp_literal(arguments["value"])});'
        else:
            raise ValueError(f'Property tool {name} requires a settable value argument')
    else:
        ordered_values = []
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            arg_name = spec.get('name')
            if arg_name in arguments:
                ordered_values.append(to_csharp_literal(arguments[arg_name]))
            elif not spec.get('required'):
                ordered_values.append(to_csharp_literal(_manifest_default_value(spec)))
        if ordered_values:
            statement = f'router.Call({to_csharp_literal(name)}, ' + ', '.join(ordered_values) + ');'
        else:
            statement = f'router.Call({to_csharp_literal(name)});'

    return (
        'using UnityEngine;\n'
        f'public class {class_name} : MonoBehaviour\n'
        '{\n'
        '    void Start()\n'
        '    {\n'
        '        var router = GetComponent<ActRouter>();\n'
        f'        {statement}\n'
        '    }\n'
        '}\n'
    )
