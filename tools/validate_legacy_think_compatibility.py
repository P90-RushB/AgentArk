"""Offline differential locks for legacy-think behavior before brief_rationale.

Run from any cwd with an existing Python environment::

    python -B tools/validate_legacy_think_compatibility.py --evidence-dir <local-output>

Only stdlib is required. The reference is read from this nested repository's Git
object database; no checkout, network, client constructor, dotenv, or Unity is
used. Selected unmodified AST methods execute in separate snapshot namespaces.
The mocks replace only API response delivery and Unity channel delivery, not
action extraction/rendering or assistant/history construction.
"""

from __future__ import annotations

import argparse
import ast
from collections import deque
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "95c41388adce35baadbbfd59f1236522a050d8e5"
DIRECT = "src/agent_ark/ark_env/direct_env.py"
API = "src/agent_ark/agent/api_agent.py"
PARSE = "src/agent_ark/utils/parse_utils.py"
CONTEXT = "src/agent_ark/ark_env/context_manager.py"
SUBMISSION = "src/agent_ark/agent/simple_agent/action_submission.py"
CODEX = "src/agent_ark/agent/simple_agent/codex_model.py"
PROMPTS = tuple("src/agent_ark/ark_env/info/" + name for name in (
    "system_prompt_func.txt", "system_prompt_code.txt", "system_prompt.txt"))
FILES = (DIRECT, API, PARSE, CONTEXT, SUBMISSION, CODEX, *PROMPTS)
OPTIONS = SimpleNamespace(reference_ref=REFERENCE, evidence_dir=None)
SOURCE = ("public sealed class AgentAction : CodeAction { "
          "public override void Start() { Context.CompleteStep(); } }")
ACTION = '<tool_call>{"name":"Paint","arguments":{"label":"门","count":2}}</tool_call>'
WRAPPER = 'TEXT={{text}}; FLAG={{flag}}; NONE={{none}}; COUNT={{count}}; KEEP={{missing}};'
MANIFEST = {"tools": [{"name": "Paint", "kind": "method", "access": "call", "arguments": [
    {"name": "label", "type": "string", "required": True},
    {"name": "count", "type": "integer", "required": True, "minimum": 1, "maximum": 4},
]}]}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE)


def text_bytes(data):
    """The bytes obtained after the production text loader's newline handling."""
    return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def class_ast(source, name, methods):
    parsed = ast.parse(source)
    original = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == name)
    node = copy.deepcopy(original)
    node.bases, node.keywords, node.decorator_list = [], [], []
    node.body = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and item.name in methods]
    present = {item.name for item in node.body}
    required = set(methods) - {"_get_explanation_mode"}
    if not required <= present:
        raise AssertionError(f"Missing {name} methods: {required - present}")
    return node


def load_class(source, name, methods, namespace, filename):
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              class_ast(source, name, methods)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace[name]


class Snapshot:
    def __init__(self, label, directory, files):
        self.files = files
        self.directory = directory / label
        for path, data in files.items():
            destination = self.directory / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        self.ns = {"__file__": str(self.directory / DIRECT), "os": os, "re": re,
                   "json": json, "deepcopy": copy.deepcopy, "deque": deque}
        exec(compile(files[PARSE].decode("utf-8"), str(self.directory / PARSE), "exec"), self.ns)
        direct = files[DIRECT].decode("utf-8")
        self.Manager = load_class(direct, "EnvInfoManager", (
            "_get_system_prompt", "_get_explanation_mode"), self.ns, str(self.directory / DIRECT))
        self.Wrapper = load_class(direct, "EnvWrapper", (
            "_get_action_mode", "_has_explicit_exact_code_action_mode", "_to_csharp_literal",
            "_render_func_wrapper", "_format_func_render_error", "_render_func_code_actions",
            "_coerce_bool", "_positive_int_or_none", "_build_rollout_budget_prompt",
            "_preferred_language_tag", "_build_llm_visible_prompt", "_build_reset_obs_payload",
            "send_code_act"), self.ns, str(self.directory / DIRECT))
        self.Agent = load_class(files[API].decode("utf-8"), "APIAgent", (
            "build_request_messages", "forward_with_trace", "forward_with_details", "forward",
            "_obs_to_messages", "_request_image_parts", "_combine_reasoning_and_content",
            "_extract_action"), self.ns, str(self.directory / API))
        context = files[CONTEXT].decode("utf-8")
        helper = next(node for node in ast.parse(context).body
                      if isinstance(node, ast.FunctionDef) and node.name == "_join_step_msg_parts")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
                     str(self.directory / CONTEXT), "exec"), self.ns)
        self.History = load_class(context, "HistoryContext", (
            "record", "_strip_prompt_from_step_msg"), self.ns, str(self.directory / CONTEXT))
        self.Messages = load_class(context, "MessageContext", (
            "_build_assistant_message",), self.ns, str(self.directory / CONTEXT))

    def env(self, env_config=None, cfg=None):
        manager = object.__new__(self.Manager)
        manager.env_config = copy.deepcopy(env_config or {})
        manager.system_prompt = manager._get_system_prompt(action_mode=manager.env_config.get("action_mode"))
        env = object.__new__(self.Wrapper)
        env.env_info_mgr, env.cfg = manager, copy.deepcopy(cfg or {})
        env.ml_unity_id_map = {11: 0, 22: 1}
        env._tool_manifest_by_unity_id = {0: copy.deepcopy(MANIFEST), 1: copy.deepcopy(MANIFEST)}
        env._code_wrapper_by_unity_id = {0: WRAPPER, 1: WRAPPER}
        return env


class LegacyThinkCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="agentark-legacy-think-")
        cls.directory = Path(OPTIONS.evidence_dir) if OPTIONS.evidence_dir else Path(cls.temporary.name)
        cls.directory.mkdir(parents=True, exist_ok=True)
        cls.reference = git("rev-parse", OPTIONS.reference_ref).decode().strip()
        old = {path: git("show", f"{cls.reference}:{path}") for path in FILES}
        new = {path: (ROOT / path).read_bytes() for path in FILES}
        cls.old = Snapshot("reference", cls.directory / "snapshots", old)
        cls.new = Snapshot("current", cls.directory / "snapshots", new)
        cls.report = {"reference_commit": cls.reference,
                      "current_commit": git("rev-parse", "HEAD").decode().strip(),
                      "method": "Separate source AST namespaces and prompt snapshots; API/Unity delivery mocked.",
                      "no_network_or_credentials_or_unity": True,
                      "scope": "Default and explicit legacy_think only; no claim of whole SimpleAgent request byte equality.",
                      "files": {}, "cases": [], "known_differences": [
                          "Working-tree CRLF versus Git LF changes physical hashes; loaded text normalizes newlines.",
                          "SimpleAgent native transport has three explanation-wording substitutions; byte equality is false.",
                          "brief_rationale is new opt-in behavior and is intentionally outside legacy compatibility."]}
        for path in FILES:
            current_blob = git("show", f"HEAD:{path}")
            cls.report["files"][path] = {
                "reference_blob_sha256": sha256(old[path]), "current_blob_sha256": sha256(current_blob),
                "working_file_sha256": sha256(new[path]),
                "reference_loaded_text_sha256": sha256(text_bytes(old[path])),
                "working_loaded_text_sha256": sha256(text_bytes(new[path])),
                "git_blob_equal": old[path] == current_blob,
                "working_loaded_text_equal": text_bytes(old[path]) == text_bytes(new[path])}

    @classmethod
    def tearDownClass(cls):
        (cls.directory / "compatibility_evidence.json").write_text(
            json.dumps(cls.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cls.temporary.cleanup()

    def same(self, name, old, new):
        self.assertEqual(old, new, name)
        serialized = json.dumps(old, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        self.report["cases"].append({"case": name, "equal": True, "result_sha256": sha256(serialized),
                                     "result": old})

    def test_01_api_parser_context_and_legacy_prompt_source_locks(self):
        for path in (API, PARSE, CONTEXT, *PROMPTS):
            with self.subTest(path=path):
                record = self.report["files"][path]
                self.assertTrue(record["git_blob_equal"], path)
                self.assertTrue(record["working_loaded_text_equal"], path)

    def test_02_default_and_explicit_legacy_prompt_loaded_utf8_bytes(self):
        for mode in ("func", " FUNC ", "code", " CODE ", None, "", "tool", "unknown", "CodeMode"):
            for explanation in ({}, {"explanation_mode": "legacy_think"}):
                config = {"prompt": explanation}
                old, new = self.old.env(config), self.new.env(config)
                self.same(f"prompt/{mode!r}/{explanation}",
                          old.env_info_mgr._get_system_prompt(action_mode=mode).encode("utf-8").hex(),
                          new.env_info_mgr._get_system_prompt(action_mode=mode).encode("utf-8").hex())

    def test_03_visible_task_prompt_budget_language_and_literal_tags(self):
        raw = ('<system_prompt>payload fallback</system_prompt><task_prompt>'
               '<english_ver>Paint literal <think>paint</think> and <rationale>ink</rationale>.</english_ver>'
               '<chinese_ver>画出中文标签。</chinese_ver><tool_docs>Paint(label,count)</tool_docs></task_prompt>'
               '<code_wrapper>INTERNAL_TEMPLATE</code_wrapper><tool_manifest>{"tools":[]}</tool_manifest>'
               '<reset_context>Canvas is blank.</reset_context>')
        for mode in ("func", "code", "tool", None):
            for language in ("en", "zh"):
                for explicit in (False, True):
                    prompt = {"language": language, "include_code_wrapper": False}
                    if explicit:
                        prompt["explanation_mode"] = "legacy_think"
                    config = {"action_mode": mode, "prompt": prompt,
                              "max_attempts": 2, "max_steps_per_attempt": 3}
                    self.same(f"visible_prompt/{mode!r}/{language}/{explicit}",
                              self.old.env(config)._build_reset_obs_payload(raw),
                              self.new.env(config)._build_reset_obs_payload(raw))

    def test_04_xml_extraction_and_api_priority(self):
        samples = ["", "plain legacy payload", ACTION, "<think>paint</think>\n" + ACTION,
                   '<THINK>paint</THINK><TOOL_CALL id="1">{"name":"Paint","arguments":{}}</TOOL_CALL>',
                   '<think>reason</think><params>{"text":"门","count":2}</params>',
                   '<params>{"a":1}</params><params>{"b":2}</params>',
                   "<think>reason</think><code>\n" + SOURCE + "\n</code>",
                   '<code>first</code><code>second</code>',
                   SOURCE, SOURCE.replace("Context.CompleteStep();", 'string s="<think>literal</think><code>literal</code>";'),
                   "<think>unterminated", '<tool_call>{bad}</tool_call>',
                   "<code>raw</code>" + ACTION + '<params>{"text":"x"}</params>']
        for index, sample in enumerate(samples):
            for tag in ("think", "tool_call", "params", "code"):
                self.same(f"xml/{index}/{tag}", self.old.ns["extract_tag_content"](sample, tag),
                          self.new.ns["extract_tag_content"](sample, tag))
            self.same(f"api_extract/{index}", self.old.Agent._extract_action(sample),
                      self.new.Agent._extract_action(sample))

    def test_05_func_rendering_and_malformed_feedback(self):
        samples = [ACTION, "<think>paint</think>\n" + ACTION,
                   '<tool_call>{"name":"Paint","arguments":{"count":1,"label":"门\\n\\\"A\\\"\\\\"}}</tool_call>',
                   '<params>{"text":"门","flag":true,"none":null,"count":3}</params>',
                   '{"text":"raw JSON","flag":false,"count":0}',
                   '<tool_call>{broken}</tool_call>',
                   '<tool_call>{"name":"Missing","arguments":{}}</tool_call>',
                   '<tool_call>{"name":"Paint","arguments":{"label":"x","count":99}}</tool_call>',
                   '<tool_call>{"name":"Paint","arguments":{"label":"x"}}</tool_call>',
                   '<tool_call>{"name":"Paint","arguments":{"label":"x","count":1,"extra":3}}</tool_call>',
                   ACTION + ACTION, '<params>{}</params><params>{}</params>', "[]", None]
        for explicit in (False, True):
            config = {"action_mode": "func"}
            if explicit:
                config["prompt"] = {"explanation_mode": "legacy_think"}
            for index, sample in enumerate(samples):
                results = []
                for snapshot in (self.old, self.new):
                    env = snapshot.env(config)
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        rendered, errors = env._render_func_code_actions({11: sample, 99: sample}, log_prefix="compat")
                    results.append({"rendered": rendered, "errors": errors, "stdout": output.getvalue()})
                self.same(f"func_render/{explicit}/{index}", *results)
                if index >= 5:
                    self.assertEqual(results[0]["rendered"], {})
                    self.assertIn("no environment action was run", results[0]["errors"][11])

    def test_06_exact_code_envelopes_raw_csharp_and_legacy_mode_passthrough(self):
        samples = [SOURCE, "  " + SOURCE + "\n", "<code>\n" + SOURCE + "\n</code>",
                   "<think>paint carefully</think>\n<code>\n" + SOURCE + "\n</code>",
                   "<THINK>paint</THINK>\n<CODE>" + SOURCE + "</CODE>",
                   "I will paint.\n<code>" + SOURCE + "</code>",
                   "<think>broken\n<code>" + SOURCE + "</code>",
                   "// literal <code> text\n" + SOURCE + "\n// literal </code>",
                   "/* <code> */\n" + SOURCE + "\n/* </code> */",
                   SOURCE.replace("Context.CompleteStep();", 'string s="<code>literal</code><think>literal</think>";'),
                   "using System;\n" + SOURCE + "\n// literal <code> </code>", None, 4]
        configurations = [({"action_mode": "code"}, {}), ({"action_mode": " CODE "}, {}),
                          ({}, {"action_mode": "code"}), ({}, {}),
                          ({"action_mode": ""}, {"action_mode": "code"}),
                          ({"action_mode": "tool"}, {}), ({"action_mode": "unknown"}, {}),
                          ({"action_mode": "CodeMode"}, {})]
        for case, (config, cfg) in enumerate(configurations):
            for explicit in (False, True):
                config = copy.deepcopy(config)
                if explicit:
                    config["prompt"] = {"explanation_mode": "legacy_think"}
                results = []
                for snapshot in (self.old, self.new):
                    env = snapshot.env(config, cfg)
                    actions = dict(enumerate(samples))
                    rendered, errors = env._render_func_code_actions(actions)
                    results.append({"action_mode": env._get_action_mode(), "rendered": rendered,
                                    "errors": errors, "same_mapping": rendered is actions})
                self.same(f"code_and_legacy/{case}/{explicit}", *results)
                if case < 3:
                    self.assertEqual(results[0]["rendered"][3], SOURCE)
                    self.assertEqual(results[0]["rendered"][5], samples[5])
                    self.assertEqual(results[0]["rendered"][9], samples[9])
                else:
                    self.assertTrue(results[0]["same_mapping"])

    def test_07_api_reply_action_assistant_history_and_request_roundtrip(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        reply = "<think>Check the visible canvas.</think>\n" + ACTION
        messages = [{"role": "system", "content": "Use the legacy task contract."},
                    {"role": "user", "content": [{"type": "text", "text": "Blank canvas."}]}]
        results = []
        for snapshot in (self.old, self.new):
            agent = object.__new__(snapshot.Agent)
            calls = []
            def respond(request):
                calls.append(copy.deepcopy(request))
                return reply, copy.deepcopy(usage)
            agent._call_api_with_usage = respond
            obs = {11: {"messages": copy.deepcopy(messages)}, 22: {"skip_infer": True}}
            requests = agent.build_request_messages(obs)
            responses, trace = agent.forward_with_trace(obs)
            history = object.__new__(snapshot.History)
            history.enabled, history._episodes, history._current = True, {}, {}
            history.max_steps_per_episode = 4
            history.record(11, {"step_msg": "Blank canvas."}, responses[11]["action"], 0.25, False,
                           assistant=responses[11]["assistant"])
            step = history._current[11][0]
            context = object.__new__(snapshot.Messages)
            assistant = context._build_assistant_message(step)
            later = messages + [assistant, {"role": "user", "content": "The mark is visible."}]
            later_request = agent.build_request_messages({11: {"messages": later}})
            results.append({"requests": requests, "actual_mock_api_requests": calls, "responses": responses,
                            "trace": trace, "stored_history": step, "assistant_message": assistant,
                            "next_request": later_request})
            self.assertEqual(responses[11], {"action": ACTION, "assistant": reply})
            self.assertEqual(assistant, {"role": "assistant", "content": reply})
            self.assertEqual(obs[11]["messages"], messages)
        self.same("api_think_response_full_roundtrip", *results)
        for step in ({"action": ACTION}, {"assistant": "", "action": ACTION},
                     {"assistant": reply, "action": ACTION}, {"assistant": None, "action": None}):
            self.same(f"assistant_fallback/{step!r}",
                      object.__new__(self.old.Messages)._build_assistant_message(step),
                      object.__new__(self.new.Messages)._build_assistant_message(step))
        for reasoning, content in (("native thought", ACTION), ("", ACTION), ("native thought", ""),
                                   ("<think>existing</think>", ACTION), ("native", reply)):
            self.same(f"reasoning_combine/{reasoning!r}",
                      self.old.Agent._combine_reasoning_and_content(reasoning=reasoning, content=content),
                      self.new.Agent._combine_reasoning_and_content(reasoning=reasoning, content=content))

    def test_08_dispatch_and_invalid_action_clear(self):
        results = []
        for snapshot in (self.old, self.new):
            env = snapshot.env({"action_mode": "func"})
            deliveries = [[], []]
            env.code_act_channels = [SimpleNamespace(send_code_act=lambda flag, text, target=target:
                                                    target.append((flag, text))) for target in deliveries]
            valid, errors = env._render_func_code_actions({11: ACTION, 22: ACTION})
            env.send_code_act(agent_id=[11, 22], code_act=valid)
            with contextlib.redirect_stdout(io.StringIO()):
                invalid, invalid_errors = env._render_func_code_actions({11: '<tool_call>{bad}</tool_call>', 22: ACTION})
            env.send_code_act(agent_id=[11, 22], code_act=invalid)
            results.append({"deliveries": deliveries, "valid_errors": errors, "invalid_errors": invalid_errors})
            self.assertEqual(deliveries[0][-1], (False, ""))
            self.assertTrue(deliveries[1][-1][0])
        self.same("dispatch_valid_then_invalid_clears_only_failed_agent", *results)

    def test_09_simpleagent_native_wording_is_explicitly_not_byte_identical(self):
        replacements = {
            "Task requests for an explicit <think>...</think> reply are superseded here. ":
                "Task requests for separate reasoning blocks are superseded here. ",
            "State the intended action and its purpose, without private reasoning or <think> wrappers. ":
                "State the intended action and its purpose, without private reasoning or explanation tags. ",
            "payload. Do not wrap the rationale in <think> tags. ":
                "payload. Use ordinary prose for the rationale, without explanation tags. ",
        }
        native = []
        for snapshot in (self.old, self.new):
            tree = ast.parse(snapshot.files[SUBMISSION].decode("utf-8"))
            native.append(ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "NATIVE_INSTRUCTIONS"
                                  for target in node.targets))))
        self.assertNotEqual(*native)
        normalized = native[0]
        for before, after in replacements.items():
            normalized = normalized.replace(before, after)
        self.same("simpleagent_native_directive_after_documented_wording_substitutions", normalized, native[1])
        for path in (SUBMISSION, CODEX):
            old_tree = ast.parse(self.old.files[path].decode("utf-8"))
            new_tree = ast.parse(self.new.files[path].decode("utf-8"))
            class ReplaceKnownWording(ast.NodeTransformer):
                def visit_Constant(self, node):
                    if isinstance(node.value, str):
                        for before, after in replacements.items():
                            node.value = node.value.replace(before, after)
                    return node
            old_tree = ReplaceKnownWording().visit(old_tree)
            self.assertEqual(ast.dump(old_tree, include_attributes=False), ast.dump(new_tree, include_attributes=False), path)
        for text in native:
            for invariant in ("exactly one standalone native submit_action", "format and content",
                              "one or two sentences", "submit_action content", "only the task action payload",
                              "without fences or XML wrappers", "does not mean execution succeeded"):
                self.assertIn(invariant, text)
        self.report["simpleagent_native"] = {
            "whole_request_byte_equal": False,
            "observed_difference": list(replacements.items()),
            "other_submission_and_codex_module_ast_equal": True,
            "semantic_assessment": "The same action payload/native submission/rationale separation is retained; "
                                   "the wording broadens the prohibition from literal think tags to explanation tags. "
                                   "Model behavior equivalence is not proved by offline tests."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ref", default=REFERENCE)
    parser.add_argument("--evidence-dir", type=Path)
    OPTIONS, remaining = parser.parse_known_args()
    program = unittest.main(argv=[__file__, *remaining], verbosity=2, exit=False)
    if OPTIONS.evidence_dir and hasattr(LegacyThinkCompatibilityTest, "report"):
        report = LegacyThinkCompatibilityTest.report
        report["test_result"] = {"tests_run": program.result.testsRun,
                                 "failures": len(program.result.failures),
                                 "errors": len(program.result.errors),
                                 "successful": program.result.wasSuccessful()}
        (OPTIONS.evidence_dir / "compatibility_evidence.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(not program.result.wasSuccessful())
