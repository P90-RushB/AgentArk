import argparse
import base64
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageDraw


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = [
    "replace-with-openrouter-model-id",
]


@dataclass(frozen=True)
class Scenario:
    name: str
    include_image: bool
    explicit_cache: bool


SCENARIOS = {
    "text_implicit": Scenario("text_implicit", include_image=False, explicit_cache=False),
    "text_explicit": Scenario("text_explicit", include_image=False, explicit_cache=True),
    "image_implicit": Scenario("image_implicit", include_image=True, explicit_cache=False),
    "image_explicit": Scenario("image_explicit", include_image=True, explicit_cache=True),
}


def build_reference_text() -> str:
    lines: list[str] = []
    for index in range(72):
        lines.append(
            f"Reference line {index:03d}: alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu."
        )
    return "\n".join(lines)


def build_test_image_data_url() -> str:
    image = Image.new("RGB", (256, 256), color=(245, 244, 238))
    draw = ImageDraw.Draw(image)
    for index in range(0, 256, 32):
        draw.rectangle((index, 0, min(index + 15, 255), 255), fill=(60, 110, 180))
        draw.rectangle((0, index, 255, min(index + 15, 255)), outline=(180, 70, 70), width=2)
    draw.text((18, 112), "CACHE TEST", fill=(15, 15, 15))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def build_static_user_content(
    reference_text: str,
    image_data_url: str,
    include_image: bool,
    explicit_cache: bool,
) -> list[dict[str, Any]]:
    reference_part: dict[str, Any] = {
        "type": "text",
        "text": reference_text,
    }
    if explicit_cache:
        reference_part["cache_control"] = {"type": "ephemeral"}

    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "Memorize the following stable reference and reuse it for future questions.",
        },
        reference_part,
    ]
    if include_image:
        image_part: dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": image_data_url},
        }
        if explicit_cache:
            image_part["cache_control"] = {"type": "ephemeral"}
        parts.append(image_part)
        parts.append(
            {
                "type": "text",
                "text": "The attached image is also stable reference context.",
            }
        )
    return parts


def build_messages(
    scenario: Scenario,
    reference_text: str,
    image_data_url: str,
    label: str,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a cache probe. Answer with exactly two words: OK plus the label.",
        },
        {
            "role": "user",
            "content": build_static_user_content(
                reference_text=reference_text,
                image_data_url=image_data_url,
                include_image=scenario.include_image,
                explicit_cache=scenario.explicit_cache,
            ),
        },
        {
            "role": "assistant",
            "content": "Reference loaded.",
        },
        {
            "role": "user",
            "content": f"Return the label {label} now.",
        },
    ]


def usage_to_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cost_details = getattr(usage, "cost_details", None)

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cost": getattr(usage, "cost", None),
        "prompt_tokens_details": {
            "cached_tokens": getattr(prompt_details, "cached_tokens", None) if prompt_details else None,
            "cache_write_tokens": getattr(prompt_details, "cache_write_tokens", None) if prompt_details else None,
            "audio_tokens": getattr(prompt_details, "audio_tokens", None) if prompt_details else None,
            "video_tokens": getattr(prompt_details, "video_tokens", None) if prompt_details else None,
        },
        "completion_tokens_details": {
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None)
            if completion_details
            else None,
            "audio_tokens": getattr(completion_details, "audio_tokens", None) if completion_details else None,
            "image_tokens": getattr(completion_details, "image_tokens", None) if completion_details else None,
        },
        "cost_details": {
            "upstream_inference_prompt_cost": getattr(cost_details, "upstream_inference_prompt_cost", None)
            if cost_details
            else None,
            "upstream_inference_completions_cost": getattr(
                cost_details, "upstream_inference_completions_cost", None
            )
            if cost_details
            else None,
            "upstream_inference_cost": getattr(cost_details, "upstream_inference_cost", None)
            if cost_details
            else None,
        },
    }


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/github/copilot",
            "X-OpenRouter-Title": "AgentArk OpenRouter Cache Probe",
        },
    )


def run_call(client: OpenAI, model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=16,
    )
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", model),
        "assistant_text": getattr(response.choices[0].message, "content", None),
        "usage": usage_to_dict(response),
    }


def analyze_pair(first_call: dict[str, Any], second_call: dict[str, Any]) -> dict[str, Any]:
    first_usage = first_call.get("usage", {})
    second_usage = second_call.get("usage", {})
    first_prompt = first_usage.get("prompt_tokens")
    second_prompt = second_usage.get("prompt_tokens")
    second_details = second_usage.get("prompt_tokens_details", {}) or {}

    cached_tokens = second_details.get("cached_tokens")
    cache_write_tokens = second_details.get("cache_write_tokens")

    return {
        "second_call_hit": bool(cached_tokens and cached_tokens > 0),
        "first_call_write": bool(
            (first_usage.get("prompt_tokens_details", {}) or {}).get("cache_write_tokens")
            and (first_usage.get("prompt_tokens_details", {}) or {}).get("cache_write_tokens") > 0
        ),
        "second_call_cached_tokens": cached_tokens,
        "second_call_cache_write_tokens": cache_write_tokens,
        "prompt_growth": (second_prompt - first_prompt) if first_prompt is not None and second_prompt is not None else None,
    }


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe OpenRouter cache behavior with OpenAI SDK.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="OpenRouter model ids to test.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIOS.keys()),
        choices=list(SCENARIOS.keys()),
        help="Scenario names to execute.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable that holds the OpenRouter API key.",
    )
    parser.add_argument(
        "--output",
        default="tmp/openrouter_cache_probe_results.json",
        help="Where to store the full JSON results, relative to the current working directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"Missing required environment variable: {args.api_key_env}", file=sys.stderr)
        return 2

    client = build_client(base_url=args.base_url, api_key=api_key)
    reference_text = build_reference_text()
    image_data_url = build_test_image_data_url()

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "models": args.models,
        "scenarios": args.scenarios,
        "results": [],
    }

    for model in args.models:
        for scenario_name in args.scenarios:
            scenario = SCENARIOS[scenario_name]
            entry: dict[str, Any] = {
                "model": model,
                "scenario": asdict(scenario),
            }
            try:
                first_messages = build_messages(
                    scenario=scenario,
                    reference_text=reference_text,
                    image_data_url=image_data_url,
                    label="ALPHA",
                )
                second_messages = build_messages(
                    scenario=scenario,
                    reference_text=reference_text,
                    image_data_url=image_data_url,
                    label="BETA",
                )

                first_call = run_call(client=client, model=model, messages=first_messages)
                second_call = run_call(client=client, model=model, messages=second_messages)

                entry.update(
                    {
                        "status": "ok",
                        "first_call": first_call,
                        "second_call": second_call,
                        "analysis": analyze_pair(first_call=first_call, second_call=second_call),
                    }
                )
            except Exception as exc:
                entry.update(
                    {
                        "status": "error",
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )
            results["results"].append(entry)
            print(
                json.dumps(
                    {
                        "model": model,
                        "scenario": scenario_name,
                        "status": entry["status"],
                        "analysis": entry.get("analysis"),
                        "error": entry.get("error"),
                    },
                    ensure_ascii=False,
                )
            )

    output_path = Path(args.output)
    ensure_output_dir(output_path)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
