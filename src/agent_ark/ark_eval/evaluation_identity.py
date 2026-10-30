"""Stable, credential-free harness identities for evaluation resumption.

Old API/Codex records have no fingerprint. They may resume the same legacy
harness, but cannot satisfy a new SimpleAgent evaluation. A fingerprinted
record always requires the same fingerprint; never downgrade that comparison.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


_DEFAULT_HTTP_ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-mode/v1'


def canonical_endpoint(value: Any) -> str:
    """Compare protocol endpoints without credentials or URL query secrets."""
    try:
        url = urlsplit(str(value))
        host = url.hostname or ''
        if ':' in host:
            host = '[' + host + ']'
        if url.port is not None:
            host += ':' + str(url.port)
        return urlunsplit((url.scheme, host, url.path.rstrip('/'), '', ''))
    except ValueError:
        return 'invalid-endpoint'


def effective_http_endpoint(value: Any, *, agent_type: str = 'api') -> str:
    # APIAgent's LLMClient strips whitespace before applying this historical
    # default. SimpleAgent uses ``base_url or default`` without that stripping.
    endpoint = str(value or '').strip() if agent_type == 'api' else value
    return canonical_endpoint(endpoint or _DEFAULT_HTTP_ENDPOINT)


def _thinking_type(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = value.get('type')
    if isinstance(value, bool):
        return 'enabled' if value else 'disabled'
    return str(value).strip().lower() if value not in (None, '') else None


def resolved_agent_name(config: Mapping[str, Any]) -> str:
    if config.get('provider') in ('human', 'replay'):
        return str(config['provider'])
    # Reading an unrelated historical record must not import/validate its
    # harness plugin. Live construction still uses the registry's validation.
    explicit = config.get('agent_type')
    if explicit is not None:
        return str(explicit).strip().lower()
    provider = str(config.get('provider', 'auto') or 'auto').strip().lower()
    return 'codex' if provider == 'codex' else 'api'


def build_evaluation_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Hash behavior settings, excluding credentials and local storage paths.

    Called on the original model configuration in both evaluators so workers
    and pre-launch resume checks calculate exactly the same identity.
    """
    agent_type = resolved_agent_name(config)
    provider = str(config.get('provider', 'auto') or 'auto').strip().lower()
    codex = provider == 'codex' or agent_type == 'codex'
    fields = {
        'agent_type': agent_type,
        'provider': provider,
        'model': str(config.get('model', 'gpt-6.1-sol' if codex else '') or ''),
        'temperature': None if codex else config.get('temperature', 0.2),
        'max_completion_tokens': config.get('max_completion_tokens', 30000),
        'timeout_s': config.get('timeout_s', config.get('request_timeout_s', 600.0 if codex else 180.0)),
        'max_retries': 0 if codex else config.get('max_retries', 0 if agent_type == 'simple' else 2),
        'reasoning_effort': config.get('reasoning_effort', config.get('effort', 'medium' if codex else None)),
    }
    fields['thinking'] = _thinking_type(config.get('thinking'))
    if not codex and agent_type not in ('human', 'replay'):
        # Userinfo and URL queries can carry credentials. They do not identify
        # the endpoint's model protocol and are intentionally excluded.
        fields['endpoint'] = effective_http_endpoint(config.get('base_url'), agent_type=agent_type)
    if agent_type == 'simple':
        from agent_ark.agent.simple_agent.runtime import SimpleRuntimeConfig
        from agent_ark.agent.simple_agent.tools import ToolLimits

        options = config.get('simple_agent', {})
        options = options if isinstance(options, Mapping) else {}
        context = options.get('context', {})
        context = context if isinstance(context, Mapping) else {}
        defaults = asdict(SimpleRuntimeConfig())
        fields['context'] = {key: context.get(key, value) for key, value in defaults.items()}
        # Explicit legacy policy must keep the exact identity of pre-policy
        # SimpleAgent records. New policies identify different evaluation
        # conditions and must never reuse those completed results.
        if fields['context'].get('compaction_policy') == 'whole_turn_v1':
            fields['context'].pop('compaction_policy')
        tools = options.get('tools', {})
        tools = tools if isinstance(tools, Mapping) else {}
        limits = tools.get('limits', {})
        limits = limits if isinstance(limits, Mapping) else {}
        defaults = asdict(ToolLimits(max_records=512))
        fields['tools'] = {
            'enabled': tools.get('enabled', False),
            'limits': {key: limits.get(key, value) for key, value in defaults.items()},
        }
        fields['tool_choice'] = options.get('tool_choice', 'required')
        fields['output_token_parameter'] = options.get('output_token_parameter', 'max_completion_tokens')
    elif agent_type == 'codex':
        fields.update({
            'thread_mode': config.get('thread_mode') or 'per_agent',
            'codex_context_mode': config.get('codex_context_mode') or 'lean',
            'sandbox': config.get('sandbox', config.get('codex_sandbox', 'read_only')) or 'read_only',
            'codex_disabled_mcp_servers': config.get('codex_disabled_mcp_servers', ['node_repl', 'unityMCP']),
        })
        feedback = config.get('player_feedback', False)
        fields['player_feedback_enabled'] = feedback.get('enabled', False) if isinstance(feedback, Mapping) else bool(feedback)
    material = json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return {'version': 1, 'agent_type': agent_type,
            'config_sha256': sha256(material.encode('utf-8')).hexdigest()}


def identity_suffix(value: Mapping[str, Any]) -> tuple[str, ...]:
    identity = value.get('evaluation_identity')
    if isinstance(identity, Mapping):
        fingerprint = identity.get('config_sha256')
        agent_type = identity.get('agent_type')
        if (identity.get('version') == 1 and isinstance(agent_type, str)
                and isinstance(fingerprint, str) and len(fingerprint) == 64
                and all(char in '0123456789abcdef' for char in fingerprint)):
            return (agent_type, fingerprint)
        return (resolved_agent_name(value), 'invalid-identity')
    # Preserve the original four-part key for old implicit API/human records.
    # Explicit SimpleAgent and Codex records must never share that key.
    agent_type = resolved_agent_name(value)
    return (agent_type, 'legacy') if agent_type not in ('api', 'human', 'replay') else ()


def legacy_result_matches(record: Mapping[str, Any], runtime: Mapping[str, Any]) -> bool:
    """Permit old API/Codex records only when their known identity agrees."""
    if isinstance(record.get('evaluation_identity'), Mapping):
        return False
    expected = resolved_agent_name(runtime)
    actual = resolved_agent_name(record)
    # Historical human rows often omitted provider as well as agent_type.
    if expected == 'human' and not record.get('provider') and not record.get('agent_type'):
        actual = 'human'
    if actual != expected or expected not in ('api', 'codex', 'human'):
        return False
    if record.get('model') is not None and runtime.get('model') is not None:
        if str(record['model']) != str(runtime['model']):
            return False
    if record.get('provider') is not None and runtime.get('provider') is not None:
        if str(record['provider']) != str(runtime['provider']):
            return False
    # An explicit null/blank URL in an old API record means the known default,
    # not an unknown endpoint. Only an entirely absent historical key is unknown.
    # Legacy CodexAgent does not use this HTTP URL at all.
    if expected == 'api' and 'base_url' in record:
        if effective_http_endpoint(record['base_url']) != effective_http_endpoint(runtime.get('base_url')):
            return False
    for key in ('thinking', 'reasoning_effort', 'max_completion_tokens', 'temperature',
                'thread_mode', 'codex_context_mode', 'codex_disabled_mcp_servers'):
        if key in record and key in runtime:
            old, new = record[key], runtime[key]
            if key == 'thinking':
                old, new = _thinking_type(old), _thinking_type(new)
            if old != new:
                return False
    return True


def matching_resume_key(
    record: Mapping[str, Any], key: tuple[str, ...] | None,
    expected: Mapping[tuple[str, ...], Mapping[str, Any]],
) -> tuple[str, ...] | None:
    if key is None:
        return None
    if key in expected:
        return key
    matches = [candidate for candidate, runtime in expected.items()
               if key[:4] == candidate[:4] and legacy_result_matches(record, runtime)]
    # A historical row cannot disambiguate two new configurations sharing a
    # display name. Rerun those jobs instead of arbitrarily assigning the row.
    return matches[0] if len(matches) == 1 else None
