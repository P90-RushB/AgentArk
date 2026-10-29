import asyncio
import json
import threading
import time
import webbrowser
from typing import Any, Dict, List, Optional

from agent_ark.interaction.hooks import to_jsonable


_VIEWER_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AgentArk Interaction Viewer</title>
<style>
:root { color-scheme: light; --bg: #f7f8fa; --panel: #fff; --line: #d9dee7; --text: #151922; --muted: #5b6472; --accent: #2563eb; --env: #0f766e; --agent: #7c3aed; --system: #475569; --error: #b91c1c; }
* { box-sizing: border-box; }
body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
header { position: sticky; top: 0; z-index: 2; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 12px 18px; border-bottom: 1px solid var(--line); background: rgba(255,255,255,.96); }
header nav { display: flex; flex-wrap: wrap; gap: 12px; font-size: 13px; }
header a { color: var(--accent); }
#modelInstructionPanel, #chat, #modelCalls { scroll-margin-top: 80px; }
h1 { margin: 0; font-size: 16px; font-weight: 700; }
.status { color: var(--muted); font-size: 13px; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 14px; padding: 14px; }
#chat { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.conversation { min-width: 0; }
.model-calls-panel { margin-top: 16px; }
.model-decision { border-top: 1px solid var(--line); padding: 10px 0; }
.model-decision:first-child { border-top: 0; }
.model-decision summary { cursor: pointer; font-size: 13px; font-weight: 600; overflow-wrap: anywhere; }
.instruction-card { border-top: 1px solid var(--line); padding: 10px 0; }
.instruction-card summary, .environment-source summary, .action-source summary { cursor: pointer; font-weight: 600; }
.instruction-card pre, .environment-source pre, .action-source pre { max-height: 60vh; overflow: auto; }
.environment-source { border-left: 3px solid #cbd5e1; padding: 6px 10px; }
.action-source { border-left: 3px solid #a78bfa; padding: 6px 10px; }
.assistant-section { margin-top: 8px; }
.model-call { margin: 10px 0 0 12px; border-left: 3px solid #dbeafe; padding: 8px 10px; }
.model-call summary { color: var(--accent); }
.call-section { margin-top: 10px; }
.call-label { color: var(--muted); font-size: 12px; font-weight: 700; margin-bottom: 5px; }
.call-message { border: 1px solid var(--line); border-radius: 5px; padding: 8px; margin-top: 6px; }
.call-message pre, .call-section pre { max-height: 60vh; overflow: auto; }
.missing-image { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
.bubble-row { display: flex; min-width: 0; }
.bubble-row.assistant { justify-content: flex-end; }
.bubble-row.system, .bubble-row.separator { justify-content: center; }
.bubble { width: min(900px, 92%); background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 2px rgba(15, 23, 42, .04); }
.assistant .bubble { border-color: #ddd6fe; background: #fbfaff; }
.user .bubble { border-color: #ccfbf1; background: #f8fffd; }
.system .bubble { border-color: #e2e8f0; background: #f8fafc; }
.error .bubble { border-color: #fecaca; background: #fff7f7; }
.separator .bubble { width: auto; max-width: 92%; color: var(--muted); font-size: 12px; text-align: center; background: transparent; border-style: dashed; box-shadow: none; }
.role { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; margin-bottom: 6px; }
.assistant .role { color: var(--agent); }
.user .role { color: var(--env); }
.system .role { color: var(--system); }
.error .role { color: var(--error); }
.message-content { display: flex; flex-direction: column; gap: 8px; }
pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
img { max-width: min(100%, 760px); border: 1px solid var(--line); border-radius: 6px; background: #fff; }
aside { position: sticky; top: 59px; height: calc(100vh - 73px); min-height: 0; display: flex; flex-direction: column; gap: 12px; overflow: auto; padding-right: 4px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
.panel.raw-trace-panel { display: flex; flex-direction: column; flex: none; }
.panel h2 { margin: 0 0 8px; font-size: 14px; }
.trace-box { min-height: 220px; max-height: 32vh; overflow: auto; border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #f8fafc; }
textarea { width: 100%; min-height: 190px; resize: vertical; border: 1px solid var(--line); border-radius: 6px; padding: 9px; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
input { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px; margin-bottom: 8px; }
button { border: 0; border-radius: 6px; background: var(--accent); color: #fff; font-weight: 700; padding: 9px 12px; cursor: pointer; }
button.secondary { background: #334155; }
.help { color: var(--muted); font-size: 12px; line-height: 1.45; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } aside { position: static; height: auto; overflow: visible; padding-right: 0; } .trace-box { min-height: 180px; max-height: none; } }
</style>
</head>
<body>
<header><h1>AgentArk Chat Viewer</h1><nav><a href="#modelInstructionPanel">Model instructions</a><a href="#chat">Conversation / actions</a><a href="#modelCalls">Actual calls</a></nav><div class="status" id="status">connecting</div></header>
<main>
    <section class="conversation">
      <section class="panel" id="modelInstructionPanel" style="margin-bottom:14px">
        <h2>Model instructions / Harness</h2>
        <p class="help">Recorded system/developer instructions used for action requests, including harness additions. Environment messages below are original task data and may be adapted by the harness. SDK transport instructions are shown separately when recorded; provider-hidden context is unavailable.</p>
        <p class="help" id="instructionContextHelp">Actual model instructions not recorded yet; older records may not have saved them.</p>
        <div id="modelInstructions"></div>
      </section>
      <h2 style="font-size:14px">Environment conversation (original)</h2>
      <p class="help">Environment prompts are source material, before any harness adaptation. Assistant cards show the public reply and selected environment action; execution results follow in environment feedback.</p>
      <section id="chat"></section>
      <section class="panel model-calls-panel">
        <h2>Actual Model Calls</h2>
        <p class="help">Recorded calls in execution order, including summaries and tool continuations. These can differ from the environment conversation above. Provider-hidden instructions and unavailable historical fields cannot be reconstructed.</p>
        <p class="help" id="modelCallLimit"></p>
        <div id="modelCalls"></div>
      </section>
    </section>
  <aside>
    <section class="panel">
      <h2>Human Action</h2>
      <input id="agentId" placeholder="agent id (optional)" />
      <textarea id="actionText" placeholder="<tool_call>{...}</tool_call>"></textarea>
      <div style="display:flex; gap:8px; margin-top:8px;"><button id="submitAction">Submit Action</button><button class="secondary" id="clearAction">Clear</button></div>
      <p class="help">When human mode is enabled, submitted text is sent unchanged as the environment action.</p>
    </section>
        <section class="panel raw-trace-panel">
            <h2>Latest Raw Trace</h2>
            <p class="help">Request previews and recorded response traces. Use Actual Model Calls for the saved per-call inputs, outputs and parameters.</p>
            <div class="help">Request preview (may be before compaction)</div>
            <div class="trace-box"><pre id="rawRequestTrace">(waiting for agent_request)</pre></div>
            <div class="help" style="margin-top:8px;">Response</div>
            <div class="trace-box"><pre id="rawResponseTrace">(waiting for agent_response)</pre></div>
        </section>
        <section class="panel"><h2>Controls</h2><button class="secondary" id="clearEvents">Clear View</button><p class="help" id="eventCount">0 raw events stored</p></section>
  </aside>
</main>
<script>
const chatEl = document.getElementById('chat');
const statusEl = document.getElementById('status');
const eventCountEl = document.getElementById('eventCount');
const rawRequestTraceEl = document.getElementById('rawRequestTrace');
const rawResponseTraceEl = document.getElementById('rawResponseTrace');
const modelCallsEl = document.getElementById('modelCalls');
const modelCallLimitEl = document.getElementById('modelCallLimit');
const modelInstructionsEl = document.getElementById('modelInstructions');
const instructionContextHelpEl = document.getElementById('instructionContextHelp');
const rawEvents = [];
let modelTraceImages = new Map();
let modelCallContext = {};
let loadingState = false;
let stateTraceImages = null;
const maxModelDecisions = 200;
let seenEventSeqs = new Map();
let lastSeenSeq = 0;
let renderedMessagesByAgent = new Map();
let renderedMessageCountsByAgent = new Map();
let renderedObservationsByAgent = new Map();
let legacyResetByAgent = new Map();
let pendingAssistantByAgent = new Map();
let instructionSignatures = new Map();
function esc(text) { return String(text ?? '').replace(/[&<>]/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[s])); }
function formatJson(value, emptyText) {
    if (value === undefined || value === null || value === '') return emptyText;
    try { return JSON.stringify(value, null, 2); } catch (e) { return String(value); }
}
function setTrace(target, value, emptyText) {
    if (!target) return;
    target.textContent = formatJson(value, emptyText);
}
function resetViewerState(options = {}) {
    const clearRawEvents = Boolean(options.clearRawEvents);
    chatEl.innerHTML = '';
    modelCallsEl.innerHTML = '';
    modelInstructionsEl.innerHTML = '';
    instructionContextHelpEl.textContent = 'Actual model instructions not recorded yet; older records may not have saved them.';
    instructionSignatures = new Map();
    modelCallLimitEl.textContent = '';
    if (options.traceImages) modelTraceImages = new Map(options.traceImages);
    else if (!options.preserveTraceImages) modelTraceImages = new Map();
    modelCallContext = {};
    renderedMessagesByAgent = new Map();
    renderedMessageCountsByAgent = new Map();
    renderedObservationsByAgent = new Map();
    legacyResetByAgent = new Map();
    pendingAssistantByAgent = new Map();
    setTrace(rawRequestTraceEl, null, '(waiting for agent_request)');
    setTrace(rawResponseTraceEl, null, '(waiting for agent_response)');
    if (clearRawEvents) {
        rawEvents.length = 0;
        seenEventSeqs = new Map();
        lastSeenSeq = 0;
        eventCountEl.textContent = '0 raw events stored';
    }
}
function renderContent(content) {
  if (Array.isArray(content)) return content.map(part => renderPart(part)).join('');
  return `<pre>${esc(content)}</pre>`;
}
function renderPart(part) {
  if (!part || typeof part !== 'object') return `<pre>${esc(part)}</pre>`;
  if (part.type === 'text') return `<pre>${esc(part.text || '')}</pre>`;
  if (part.type === 'image_url' && part.url) return `<img src="${part.url}" />`;
  return `<pre>${esc(JSON.stringify(part, null, 2))}</pre>`;
}
function roleClass(role) {
    if (role === 'assistant') return 'assistant';
    if (role === 'system') return 'system';
    if (role === 'error') return 'error';
    return 'user';
}
function roleLabel(role) {
    if (role === 'assistant') return 'assistant';
    if (role === 'system') return 'system';
    if (role === 'error') return 'error';
    return 'environment';
}
function extractTurnIndex(ev) {
    const payload = (ev && ev.payload) || {};
    const directTurnIndex = Number(payload.turn_index);
    if (Number.isFinite(directTurnIndex)) return directTurnIndex;

    const nestedTurnIndex = Number(
        payload && payload.info && payload.info.sub_env && payload.info.sub_env.step
            ? payload.info.sub_env.step.turn_index
            : NaN
    );
    if (Number.isFinite(nestedTurnIndex)) return nestedTurnIndex;
    return null;
}
function buildSeparatorText(text, ev) {
    const seq = Number(ev && ev.seq);
    if (Number.isFinite(seq) && seq > 0) {
        return `seq ${seq} · ${text}`;
    }
    return text;
}
function buildBubbleMeta(agentId, ev) {
    const parts = [];
    if (agentId !== undefined && agentId !== null && `${agentId}` !== '') parts.push(`${agentId}`);

    const turnIndex = extractTurnIndex(ev);
    if (turnIndex !== null) parts.push(`turn ${turnIndex + 1}`);

    return parts.join(' · ');
}
function appendBubble(role, content, options = {}) {
    const row = document.createElement('article');
    row.className = `bubble-row ${roleClass(role)}`;
    const meta = options.meta ? ` · ${esc(options.meta)}` : '';
    const body = options.environment ? renderEnvironmentContent(content, role) : renderContent(content);
    row.innerHTML = `<div class="bubble"><div class="role">${roleLabel(role)}${meta}</div><div class="message-content">${body}</div></div>`;
    chatEl.appendChild(row);
}
function renderEnvironmentContent(content, role) {
    if (role === 'system' || role === 'developer') {
        return `<details class="environment-source"><summary>Environment original ${esc(role)} prompt (harness may adapt)</summary>${renderContent(content)}</details>`;
    }
    const sourceText = text => {
        const match = String(text ?? '').match(/<system_prompt>[\s\S]*?<\/system_prompt>/);
        if (!match) return `<pre>${esc(text)}</pre>`;
        const at = match.index;
        return (at ? `<pre>${esc(text.slice(0, at))}</pre>` : '')
            + `<details class="environment-source"><summary>Environment original system prompt (harness may adapt)</summary><pre>${esc(match[0])}</pre></details>`
            + (at + match[0].length < text.length ? `<pre>${esc(text.slice(at + match[0].length))}</pre>` : '');
    };
    if (typeof content === 'string') return sourceText(content);
    if (Array.isArray(content)) return content.map(part => part && part.type === 'text' ? sourceText(part.text || '') : renderPart(part)).join('');
    return renderContent(content);
}
function appendSeparator(text, ev) {
    const row = document.createElement('article');
    row.className = 'bubble-row separator';
    row.innerHTML = `<div class="bubble">${esc(buildSeparatorText(text, ev))}</div>`;
    chatEl.appendChild(row);
    renderedMessagesByAgent = new Map();
    renderedMessageCountsByAgent = new Map();
    renderedObservationsByAgent = new Map();
    legacyResetByAgent = new Map();
    pendingAssistantByAgent = new Map();
}
function obsFallbackContent(obs) {
    const content = [];
    if (obs && obs.step_msg) content.push({type: 'text', text: obs.step_msg});
    if (obs && Array.isArray(obs.images)) {
        for (const image of obs.images) {
            if (image && image.url) content.push({type: 'image_url', url: image.url});
        }
    }
    return content.length ? content : null;
}
function fullMessageStart(previous, current) {
    // Preserve a stable prompt prefix, then match a sliding history window.
    // Counts alone cannot distinguish equal-size or shorter replacement windows.
    let prefix = 0;
    while (prefix < previous.length && prefix < current.length && previous[prefix] === current[prefix]) prefix += 1;
    for (let overlap = Math.min(previous.length - prefix, current.length - prefix); overlap > 0; overlap -= 1) {
        if (current.slice(prefix, prefix + overlap).every((key, idx) => key === previous[previous.length - overlap + idx])) {
            return prefix + overlap;
        }
    }
    return prefix;
}
function assistantEchoText(content) {
    if (typeof content === 'string') return content;
    if (Array.isArray(content) && content.every(part => part && part.type === 'text')) {
        return content.map(part => part.text || '').join('');
    }
    return null;
}
function renderObsMessages(obsMap, ev) {
    if (!obsMap || typeof obsMap !== 'object') return;
    const payload = ev.payload || {};
    const mode = payload.message_mode === 'delta' ? 'delta' : (payload.message_mode === 'full' ? 'full' : 'legacy');
    const observationId = payload.observation_id;
    for (const [agentId, obs] of Object.entries(obsMap)) {
        if (!obs || typeof obs !== 'object') continue;
        if (ev.event === 'env_reset' && observationId == null) {
            const previousReset = legacyResetByAgent.get(agentId);
            const resetKey = JSON.stringify(obs);
            // Older hook streams have no observation IDs. Only coalesce the
            // adjacent cross-source reset mirror, never a later identical reset.
            if (previousReset && previousReset.source === 'ArkEnv' && ev.source === 'EnvRuntime' && resetKey === previousReset.key) {
                legacyResetByAgent.delete(agentId);
                continue;
            }
            legacyResetByAgent.set(agentId, {key: resetKey, source: ev.source});
        } else {
            legacyResetByAgent.delete(agentId);
        }
        // One observation can be mirrored by ArkEnv, EnvRuntime, and the request
        // hook. Content is not an identity: identical frames on later steps count.
        if (observationId !== undefined && observationId !== null) {
            const seen = renderedObservationsByAgent.get(agentId) || new Set();
            if (seen.has(String(observationId))) continue;
            seen.add(String(observationId));
            renderedObservationsByAgent.set(agentId, seen);
        }
        if (ev.event === 'env_reset') {
            renderedMessagesByAgent.delete(agentId);
            renderedMessageCountsByAgent.delete(agentId);
            pendingAssistantByAgent.delete(agentId);
        }
        const messages = Array.isArray(obs.messages) ? obs.messages : [];
        const fallback = messages.length ? null : obsFallbackContent(obs);
        const visible = messages.length ? messages : (fallback ? [{role: 'user', content: fallback}] : []);
        const keys = visible.map(msg => JSON.stringify(msg));
        const previous = renderedMessagesByAgent.get(agentId) || [];
        const isRequest = ev.event === 'agent_request' || ev.event === 'human_request';
        // Raw fallback is a current frame, never a full message history. Retain
        // repeated frames on separate steps; legacy request mirrors still overlap.
        let startIndex = 0;
        if (mode !== 'delta' && (messages.length || isRequest)) {
            // Unmarked archived snapshots retain the original count contract.
            // New emitters explicitly mark full and can use overlap/boundaries.
            const previousCount = renderedMessageCountsByAgent.get(agentId) || 0;
            startIndex = mode === 'legacy' && messages.length
                ? (messages.length < previousCount ? 0 : previousCount)
                : fullMessageStart(previous, keys);
        }
        const newStarts = payload.new_message_start_by_agent;
        const authoredStart = newStarts && newStarts[agentId];
        if (mode !== 'delta' && messages.length && renderedMessageCountsByAgent.has(agentId) && Number.isInteger(authoredStart) && authoredStart >= 0 && authoredStart <= messages.length) {
            startIndex = authoredStart;
        }
        for (let idx = startIndex; idx < visible.length; idx += 1) {
            const msg = visible[idx];
            if (!msg || typeof msg !== 'object') continue;
            const role = String(msg.role || 'user');
            const content = msg.content || '';
            const pendingAssistant = pendingAssistantByAgent.get(agentId);
            if (role === 'assistant' && pendingAssistant !== undefined && assistantEchoText(content) === pendingAssistant) {
                pendingAssistantByAgent.delete(agentId);
                continue;
            }
            appendBubble(role, content, {meta: buildBubbleMeta(agentId, ev), environment: role !== 'assistant'});
        }
        renderedMessagesByAgent.set(agentId, keys);
        if (messages.length) renderedMessageCountsByAgent.set(agentId, messages.length);
        // An unreturned echo must not suppress the same text on an unrelated turn.
        pendingAssistantByAgent.delete(agentId);
    }
}
function renderAssistantResponse(ev) {
    const payload = ev.payload || {};
    const actions = payload.actions;
    if (!actions || typeof actions !== 'object') return;
    for (const [agentId, action] of Object.entries(actions)) {
        const trace = (payload.raw_trace_by_agent || payload.raw_response_by_agent || {})[agentId] || {};
        const calls = Array.isArray(trace.model_requests) ? trace.model_requests : [];
        const call = [...calls].reverse().find(item => item && item.purpose === 'action');
        const forwarded = action && typeof action === 'object' ? action.assistant : undefined;
        // Echo identity follows the original environment-facing response, even
        // when the public model content is different from that response.
        if (typeof forwarded === 'string' && forwarded) pendingAssistantByAgent.set(String(agentId), forwarded);
        const native = trace.simple_agent && trace.simple_agent.native_action_submission === true;
        const publicText = native && call && typeof call.completion === 'string' ? call.completion
            : native && typeof trace.assistant_raw === 'string' ? trace.assistant_raw : forwarded;
        const hasAction = Boolean(action && typeof action === 'object' && Object.prototype.hasOwnProperty.call(action, 'action'));
        let selected = hasAction ? action.action : undefined;
        let actionNote = '';
        if (typeof selected === 'string' && selected.endsWith('\n...[truncated]')) {
            const prefix = selected.slice(0, -'\n...[truncated]'.length);
            if (typeof trace.action_extracted === 'string' && trace.action_extracted.startsWith(prefix)) {
                selected = trace.action_extracted;
                actionNote = 'Full action from the saved action trace; the event preview was truncated.';
            } else actionNote = 'Truncated event preview; the complete action was not recorded here.';
        }
        if (!hasAction && !publicText) continue;
        renderAppliedSummary(trace, call, agentId, ev);
        let body = '';
        // Native assistant content is prose even when it happens to contain a
        // short raw/params action. Keep that submitted value explicit.
        const combined = !native && typeof selected === 'string' && selected && typeof publicText === 'string' && publicText.includes(selected);
        if (combined) {
            const combinedBody = publicText.length > 2000
                ? `<details class="action-source"><summary>Full assistant reply and environment action · ${publicText.length} characters</summary><pre>${esc(publicText)}</pre></details>`
                : `<pre>${esc(publicText)}</pre>`;
            body += `<div class="call-label">Assistant reply / Environment action (combined)</div>${combinedBody}`;
        } else {
            if (typeof publicText === 'string' && publicText) body += `<div class="assistant-section"><div class="call-label">Assistant content</div><pre>${esc(publicText)}</pre></div>`;
            if (hasAction) {
                const value = selected === null ? '(no environment action selected)' : selected === '' ? '(empty environment action)' : typeof selected === 'string' ? selected : formatJson(selected, '(action unavailable)');
                const display = typeof selected === 'string' && selected.length <= 2000 ? `<pre>${esc(value)}</pre>`
                    : typeof selected === 'string' ? `<details class="action-source"><summary>Full environment action · ${selected.length} characters</summary><pre>${esc(selected)}</pre></details>`
                    : `<pre>${esc(value)}</pre>`;
                body += `<div class="assistant-section"><div class="call-label">Environment action</div>${display}</div>`;
            }
        }
        if (hasAction) body += `<p class="help">Selected action; execution result is reported by the next environment feedback.${actionNote ? ' ' + esc(actionNote) : ''}</p>`;
        const row = document.createElement('article');
        row.className = 'bubble-row assistant';
        row.innerHTML = `<div class="bubble"><div class="role">assistant · ${esc(buildBubbleMeta(agentId, ev))}</div><div class="message-content">${body}</div></div>`;
        chatEl.appendChild(row);
    }
}
function renderAppliedSummary(trace, call, agentId, ev) {
    if (!trace.simple_agent || !call || !Array.isArray(call.request_messages)) return;
    const summaries = (trace.model_requests || []).filter(item => item && item.purpose === 'summary');
    if (!summaries.length) return;
    const notes = call.request_messages.filter(message => message && message.role === 'user').flatMap(message =>
        typeof message.content === 'string' ? [message.content] : Array.isArray(message.content)
            ? message.content.filter(part => part && part.type === 'text').map(part => part.text || '') : []
    ).filter(text => text.startsWith('Model notes; not environment facts.'));
    if (!notes.length) return;
    const row = document.createElement('article');
    row.className = 'bubble-row system';
    row.innerHTML = `<div class="bubble" data-kind="harness-context"><div class="role">SimpleAgent context · ${esc(buildBubbleMeta(agentId, ev))}</div><details><summary>Context notes used · ${summaries.length} summary call(s) this decision</summary><p class="help">This is the memory in the actual action request. A summary attempt alone does not prove new notes were committed.</p>${notes.map(note => `<pre>${esc(note)}</pre>`).join('')}</details></div>`;
    chatEl.appendChild(row);
}
function updateRawTrace(ev) {
    const payload = ev.payload || {};
    if (ev.event === 'agent_request' && payload.raw_trace_by_agent) {
        setTrace(rawRequestTraceEl, payload.raw_trace_by_agent, '(waiting for agent_request)');
    } else if (ev.event === 'agent_response' && payload.raw_trace_by_agent) {
        setTrace(rawResponseTraceEl, payload.raw_trace_by_agent, '(waiting for agent_response)');
    } else if (ev.event === 'human_response' && payload.action) {
        setTrace(rawResponseTraceEl, {'__default__': {'assistant_raw': payload.action, 'action_extracted': payload.action}}, '(waiting for agent_response)');
    }
}
function traceJsonSection(label, value, emptyText = '(not recorded)') {
    const text = typeof value === 'string' && value ? value : formatJson(value, emptyText);
    return `<div class="call-section"><div class="call-label">${esc(label)}</div><pre>${esc(text)}</pre></div>`;
}
function traceImage(url) {
    let reference = '';
    const match = typeof url === 'string' && url.match(/^agentark-trace-image:([a-f0-9]{64})$/);
    if (match) {
        reference = match[1];
        const asset = modelTraceImages.get(reference);
        url = asset && asset.data_url;
    }
    // Model text and archived traces are untrusted. Only local inline raster
    // images can become HTML attributes; never fetch a model-supplied URL.
    if (typeof url === 'string' && /^data:image\/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/=\r\n]+$/.test(url)) {
        return `<img src="${url.replace(/[\r\n]/g, '')}" alt="Recorded model request image${reference ? ' ' + reference : ''}" />`;
    }
    return `<div class="missing-image">Request image unavailable${reference ? ' · SHA-256 ' + reference : ''}${!reference && url ? ' · ' + esc(url) : ''}</div>`;
}
function traceContent(content) {
    if (!Array.isArray(content)) return `<pre>${esc(typeof content === 'string' ? content : formatJson(content, '(empty content)'))}</pre>`;
    return content.map(part => {
        if (part && part.type === 'text') return `<pre>${esc(part.text || '')}</pre>`;
        if (part && part.type === 'image_url') return traceImage(part.url || (part.image_url && part.image_url.url));
        return `<pre>${esc(formatJson(part, '(empty part)'))}</pre>`;
    }).join('');
}
function traceMessages(messages) {
    if (!Array.isArray(messages) || !messages.length) return '<p class="help">Request messages not recorded.</p>';
    return messages.map((message, index) => {
        if (!message || typeof message !== 'object') return traceJsonSection(`Message ${index + 1}`, message);
        const extras = Object.fromEntries(Object.entries(message).filter(([key]) => key !== 'role' && key !== 'content'));
        return `<div class="call-message"><div class="call-label">${index + 1} · ${esc(message.role || 'unknown role')}</div>${traceContent(message.content)}${Object.keys(extras).length ? traceJsonSection('Message fields (tool calls, IDs or native continuation)', extras) : ''}</div>`;
    }).join('');
}
function traceTransport(transport) {
    if (!transport || typeof transport !== 'object' || !Array.isArray(transport.inputs)) {
        return traceJsonSection('SDK transport request', transport);
    }
    const metadata = Object.fromEntries(Object.entries(transport).filter(([key]) => key !== 'inputs'));
    const inputs = transport.inputs.map((input, index) => {
        let content;
        if (input && input.type === 'image') content = traceImage(input.url);
        else if (input && input.type === 'text') content = `<pre>${esc(input.text || '')}</pre>`;
        else content = `<pre>${esc(formatJson(input, '(empty input)'))}</pre>`;
        return `<div class="call-message"><div class="call-label">SDK input ${index + 1}</div>${content}</div>`;
    }).join('');
    return `<div class="call-section"><div class="call-label">SDK transport request</div>${inputs}</div>` + traceJsonSection('SDK boundary and output schema (SDK-added context may be unavailable)', metadata);
}
function renderModelInstructions(ev) {
    if (ev.event !== 'agent_response') return;
    const payload = ev.payload || {};
    const traces = payload.raw_trace_by_agent || payload.raw_response_by_agent || {};
    for (const [agentId, trace] of Object.entries(traces)) {
        const calls = Array.isArray(trace.model_requests) ? trace.model_requests : [];
        const call = [...calls].reverse().find(item => item && item.purpose === 'action' && Array.isArray(item.request_messages));
        if (!call) continue;
        const instructions = [];
        for (const message of call.request_messages) {
            if (!message || !['system', 'developer'].includes(message.role)) break;
            instructions.push(message);
        }
        if (!instructions.length) continue;
        const scope = [payload.case_id ?? modelCallContext.case_id ?? '', payload.model_name ?? modelCallContext.model_name ?? '', agentId];
        const transport = call.request_transport;
        const codex = transport && transport.transport === 'codex_sdk_structured_json';
        // The Codex adapter's first input is its outer protocol; subsequent
        // inputs carry the portable messages already shown separately below.
        const outer = codex && Array.isArray(transport.inputs) ? transport.inputs.slice(0, 1) : [];
        const metadata = transport ? Object.fromEntries(Object.entries(transport).filter(([key]) => key !== 'inputs')) : null;
        const config = trace.simple_agent ? Object.fromEntries(Object.entries(trace.simple_agent).filter(([key]) => !['turn', 'attempt'].includes(key))) : null;
        const signature = JSON.stringify([instructions, outer, metadata, call.tools, config]);
        const key = JSON.stringify(scope);
        if (instructionSignatures.get(key) === signature) continue;
        instructionSignatures.set(key, signature);
        const harness = trace.simple_agent ? 'SimpleAgent' : 'Model';
        const protocol = codex ? 'Codex SDK structured JSON ({text, tool_calls})'
            : config && config.native_action_submission === true ? 'Declared function tools' : 'Recorded model messages';
        const title = [...scope.filter(Boolean), harness, protocol, 'Recorded model instructions'].join(' · ');
        const card = document.createElement('details');
        card.className = 'instruction-card';
        card.innerHTML = `<summary>${esc(title)}</summary>`;
        let populated = false;
        card.ontoggle = () => {
            if (!card.open || populated) return;
            populated = true;
            let body = '<p class="help">From the saved action request, after harness adaptation. These instructions supersede conflicting output-format examples in the original environment prompt. Summary requests use their own instructions.</p>';
            if (codex) {
                body += '<p class="help">Codex SDK structured JSON transport: the outer response is {text, tool_calls}. Supplied-message function instructions are encoded by this transport; this is not an SDK native-function-tools request.</p>';
                body += traceJsonSection('Outer SDK transport instructions', outer);
                body += traceJsonSection('SDK output schema and recorded thread instructions / boundary (SDK-added context may be unavailable)', metadata);
            } else if (transport) {
                body += traceJsonSection('Recorded transport metadata', metadata);
            }
            body += `<div class="call-section"><div class="call-label">Actual action request · system / developer messages</div>${traceMessages(instructions)}</div>`;
            if (call.tools !== undefined) body += traceJsonSection('Declared function tools', call.tools, '(no tools recorded)');
            if (config) body += traceJsonSection('Saved harness settings', config);
            card.innerHTML = `<summary>${esc(title)}</summary>` + body;
        };
        modelInstructionsEl.appendChild(card);
        instructionContextHelpEl.textContent = `${harness} · ${protocol}. Saved action-request instructions are available below. Environment original output-format examples may be adapted; context notes and per-call details appear in the conversation and Actual Model Calls.`;
        while (modelInstructionsEl.children.length > maxModelDecisions) modelInstructionsEl.removeChild(modelInstructionsEl.firstChild);
        while (instructionSignatures.size > maxModelDecisions) instructionSignatures.delete(instructionSignatures.keys().next().value);
    }
}
function renderModelCall(call, index) {
    if (!call || typeof call !== 'object') return traceJsonSection(`Call ${index + 1}`, call);
    const title = `Call ${index + 1} · ${call.purpose || 'unknown purpose'} · ${call.status || 'status not recorded'}`;
    const displayed = new Set(['purpose', 'status', 'request_messages', 'tools', 'parameters', 'completion', 'tool_calls', 'response_metadata', 'native_assistant', 'usage', 'request_images', 'request_transport']);
    const extra = Object.fromEntries(Object.entries(call).filter(([key]) => !displayed.has(key)));
    let html = `<details class="model-call"><summary>${esc(title)}</summary><div class="call-section"><div class="call-label">Actual request messages</div>${traceMessages(call.request_messages)}</div>`;
    if (call.request_transport !== undefined) html += traceTransport(call.request_transport);
    html += traceJsonSection('Declared tools', call.tools, '(no tools recorded)');
    html += traceJsonSection('Request parameters', call.parameters);
    html += traceJsonSection('Assistant content', call.completion, '(empty assistant content)');
    html += traceJsonSection('Assistant function calls', call.tool_calls, '(no function calls recorded)');
    if (call.response_metadata !== undefined) html += traceJsonSection('Response metadata / native reasoning (encrypted fields are opaque)', call.response_metadata);
    if (call.native_assistant !== undefined) html += traceJsonSection('Native assistant continuation (encrypted fields are opaque)', call.native_assistant);
    html += traceJsonSection('Usage', call.usage);
    if (call.request_images !== undefined) html += traceJsonSection('Request image fingerprints', call.request_images);
    if (Object.keys(extra).length) html += traceJsonSection('Additional recorded call fields', extra);
    return html + '</details>';
}
function renderModelTrace(ev) {
    const payload = ev.payload || {};
    if (ev.event === 'case_start') modelCallContext.case_id = payload.case_id;
    if (ev.event === 'model_start') modelCallContext.model_name = payload.model_name;
    if (ev.event !== 'agent_response') return;
    const images = payload.model_trace_images;
    if (images && typeof images === 'object') {
        for (const [key, value] of Object.entries(images)) {
            if (/^[a-f0-9]{64}$/.test(key) && value && typeof value === 'object') modelTraceImages.set(key, value);
        }
    }
    const traces = payload.raw_trace_by_agent || payload.raw_response_by_agent || {};
    const agents = new Set([...Object.keys(traces), ...Object.keys(payload.actions || {})]);
    for (const agentId of agents) {
        const trace = traces[agentId] || {};
        const calls = Array.isArray(trace.model_requests) ? trace.model_requests : [];
        const parts = [payload.case_id ?? modelCallContext.case_id,
            payload.model_name ?? modelCallContext.model_name, buildBubbleMeta(agentId, ev)].filter(Boolean);
        parts.push(calls.length ? `${calls.length} model call${calls.length === 1 ? '' : 's'}` : (trace.skipped ? 'inference skipped' : 'model calls unavailable'));
        if (trace.replay) parts.push('recorded replay');
        const card = document.createElement('details');
        card.className = 'model-decision';
        const summary = `<summary>${esc(parts.join(' · '))}</summary>`;
        card.innerHTML = summary;
        // Full contexts can be large and are repeated across model calls.
        // Materialize them only when this decision is opened by the reader.
        let populated = false;
        card.ontoggle = () => {
            if (!card.open || populated) return;
            populated = true;
            let body = calls.length ? calls.map(renderModelCall).join('')
                : (trace.skipped ? '<p class="help">Inference was skipped; no model request was made for this agent.</p>'
                    : '<p class="help">This legacy record did not save actual model calls. The environment conversation and saved action remain available; missing harness inputs cannot be reconstructed.</p>');
            if (trace.agent_events !== undefined) body += traceJsonSection('Agent events / tool results', trace.agent_events);
            if (trace.agent_usage !== undefined) body += traceJsonSection('Decision usage', trace.agent_usage);
            card.innerHTML = summary + body;
        };
        modelCallsEl.appendChild(card);
        while (modelCallsEl.children.length > maxModelDecisions) {
            modelCallsEl.removeChild(modelCallsEl.firstChild);
            modelCallLimitEl.textContent = `Showing the latest ${maxModelDecisions} decisions. Older calls are outside this view; saved records are unchanged.`;
        }
    }
}
function addEvent(ev) {
    const seq = Number(ev && ev.seq);
    const eventIdentity = JSON.stringify([ev.time ?? null, ev.event, ev.source ?? null]);
    // SSE reconnects replay buffered events, including run_start. Check them
    // before treating a backwards sequence as a fresh stream.
    if (Number.isFinite(seq) && seq > 0 && seenEventSeqs.get(seq) === eventIdentity) return;
    if (ev && ev.event === 'run_start') {
        resetViewerState({clearRawEvents: true, traceImages: loadingState ? stateTraceImages : null});
    } else if (Number.isFinite(seq) && seq > 0 && lastSeenSeq > 0 && seq <= lastSeenSeq) {
        resetViewerState({clearRawEvents: true, traceImages: loadingState ? stateTraceImages : null});
    }
    if (Number.isFinite(seq) && seq > 0) {
        seenEventSeqs.set(seq, eventIdentity);
        lastSeenSeq = seq;
    }
    rawEvents.push(ev);
    eventCountEl.textContent = `${rawEvents.length} raw events stored`;
    const payload = ev.payload || {};
    updateRawTrace(ev);
    renderModelTrace(ev);
    renderModelInstructions(ev);
    if (ev.event === 'run_start') appendSeparator('new run', ev);
    else if (ev.event === 'case_start') appendSeparator(`case ${payload.case_id || ''} · ${payload.task_name || ''}`.trim(), ev);
    else if (ev.event === 'model_start') appendSeparator(`model ${payload.model_name || ''}`.trim(), ev);
    else if (ev.event === 'agent_request' || ev.event === 'human_request' || ev.event === 'env_reset') renderObsMessages(payload.obs, ev);
    else if (ev.event === 'env_step') renderObsMessages(payload.next_obs, ev);
    else if (ev.event === 'agent_response') renderAssistantResponse(ev);
    else if (ev.event === 'human_response' && payload.action) {
        // human_response already feeds raw trace; the visible assistant bubble is rendered from agent_response.
    }
    else if (ev.event === 'error') appendBubble('error', payload.error || JSON.stringify(payload, null, 2), {force: true});
    while (chatEl.children.length > 500) chatEl.removeChild(chatEl.firstChild);
  window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
}
async function loadState() {
  const res = await fetch('/state');
  const data = await res.json();
  resetViewerState({clearRawEvents: true});
  for (const [key, value] of Object.entries(data.model_trace_images || {})) {
      if (/^[a-f0-9]{64}$/.test(key) && value && typeof value === 'object') modelTraceImages.set(key, value);
  }
  stateTraceImages = new Map(modelTraceImages);
  loadingState = true;
  try { for (const ev of data.events || []) addEvent(ev); } finally { loadingState = false; stateTraceImages = null; }
}
async function connectViewer() {
    try { await loadState(); } catch (e) {}
    const es = new EventSource('/events');
    es.onopen = () => statusEl.textContent = 'connected';
    es.onerror = () => statusEl.textContent = 'reconnecting';
    es.onmessage = msg => { try { addEvent(JSON.parse(msg.data)); } catch (e) {} };
}
connectViewer();
document.getElementById('submitAction').onclick = async () => {
  const action = document.getElementById('actionText').value;
  const agent_id = document.getElementById('agentId').value || null;
  await fetch('/human/actions', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({agent_id, action})});
};
document.getElementById('clearAction').onclick = () => document.getElementById('actionText').value = '';
document.getElementById('clearEvents').onclick = () => {
    resetViewerState({clearRawEvents: true, preserveTraceImages: true});
};
window.AgentArkRawEvents = rawEvents;
</script>
</body>
</html>
"""


class HumanActionBroker:
    def __init__(self):
        self._condition = threading.Condition()
        self._actions: Dict[str, List[str]] = {}

    @staticmethod
    def _key(agent_id: Any = None) -> str:
        if agent_id is None or str(agent_id).strip() == '':
            return '__default__'
        return str(agent_id)

    def submit(self, action: Any, *, agent_id: Any = None) -> None:
        action_text = '' if action is None else str(action)
        key = self._key(agent_id)
        with self._condition:
            self._actions.setdefault(key, []).append(action_text)
            self._condition.notify_all()

    def wait_for_action(self, *, agent_id: Any = None, timeout: Optional[float] = None) -> str:
        key = self._key(agent_id)
        deadline = None if timeout is None else time.time() + float(timeout)
        with self._condition:
            while True:
                for candidate in (key, '__default__'):
                    queue = self._actions.get(candidate) or []
                    if queue:
                        return queue.pop(0)
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError(f'No human action submitted for agent_id={agent_id!r}')
                    self._condition.wait(timeout=remaining)


class LocalViewerHook:
    def __init__(
        self,
        *,
        host: str = '127.0.0.1',
        port: int = 18181,
        event_buffer_size: int = 500,
        open_browser: bool = False,
        action_broker: Optional[HumanActionBroker] = None,
    ):
        self.host = host
        self.port = int(port)
        self.event_buffer_size = max(1, int(event_buffer_size))
        self.open_browser = bool(open_browser)
        self.action_broker = action_broker or HumanActionBroker()
        self._events: List[Dict[str, Any]] = []
        self._model_trace_images: Dict[str, Dict[str, Any]] = {}
        self._event_trace_image_refs: List[set[str]] = []
        self._active_trace_case: Optional[tuple[str, str]] = None
        self._active_trace_image_keys: set[str] = set()
        self._unscoped_trace_image_keys: set[str] = set()
        self._lock = threading.RLock()
        self._server = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f'http://{self.host}:{self.port}'

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            import uvicorn
            from fastapi import FastAPI, Request
            from fastapi.responses import HTMLResponse, StreamingResponse
        except Exception as exc:  # pragma: no cover
            raise RuntimeError('fastapi and uvicorn are required for LocalViewerHook') from exc

        app = FastAPI(title='AgentArk Interaction Viewer', version='0.1.0')

        @app.get('/', response_class=HTMLResponse)
        def index() -> str:
            return _VIEWER_HTML

        @app.get('/health')
        def health() -> Dict[str, Any]:
            return {'ok': True, 'event_count': len(self._snapshot_events())}

        @app.get('/state')
        def state() -> Dict[str, Any]:
            return self._snapshot_state()

        @app.get('/events')
        async def events() -> StreamingResponse:
            async def stream():
                cursor = 0
                while True:
                    snapshot = self._snapshot_events()
                    for item in snapshot:
                        seq = int(item.get('seq', 0) or 0)
                        if seq <= cursor:
                            continue
                        cursor = seq
                        yield 'data: ' + json.dumps(item, ensure_ascii=False) + '\n\n'
                    await asyncio.sleep(0.25)
            return StreamingResponse(stream(), media_type='text/event-stream')

        @app.post('/human/actions')
        async def submit_action(request: Request) -> Dict[str, Any]:
            body = await request.json()
            action = body.get('action', '') if isinstance(body, dict) else ''
            agent_id = body.get('agent_id') if isinstance(body, dict) else None
            self.action_broker.submit(action, agent_id=agent_id)
            return {'ok': True}

        config = uvicorn.Config(app, host=self.host, port=self.port, log_level='warning')
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name='agentark-viewer', daemon=True)
        self._thread.start()
        print(f'[AgentArk viewer] {self.url}')
        if self.open_browser:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass

    def handle_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            saved = to_jsonable(event)
            if saved.get('event') == 'run_start':
                self._model_trace_images.clear()
                self._active_trace_case = None
                self._active_trace_image_keys.clear()
                self._unscoped_trace_image_keys.clear()
            payload = saved.get('payload') or {}
            payload = payload if isinstance(payload, dict) else {}
            scope = self._trace_scope(payload)
            if saved.get('event') == 'case_start' and payload.get('case_id') not in (None, ''):
                case_id = str(payload['case_id'])
                if self._active_trace_case is None or self._active_trace_case[0] != case_id:
                    self._active_trace_case = (case_id, '')
                    self._active_trace_image_keys.clear()
            elif saved.get('event') == 'agent_request' and scope is not None and scope != self._active_trace_case:
                self._active_trace_case = scope
                self._active_trace_image_keys.clear()
            images = payload.get('model_trace_images') if isinstance(payload, dict) else None
            image_keys = set()
            if isinstance(images, dict):
                assets = {key: value for key, value in images.items() if self._image_key(key) and isinstance(value, dict)}
                self._model_trace_images.update(assets)
                image_keys.update(assets)
            references = set(self._image_references(saved))
            if saved.get('event') == 'agent_response':
                if scope is not None and self._active_trace_case is None:
                    self._active_trace_case = scope
                if scope is not None and scope == self._active_trace_case:
                    self._active_trace_image_keys.update(references | image_keys)
                else:
                    # Older/custom streams may omit scope or interleave cases.
                    # Keep their assets for this run instead of guessing expiry.
                    self._unscoped_trace_image_keys.update(references | image_keys)
            self._events.append(saved)
            self._event_trace_image_refs.append(references)
            if len(self._events) > self.event_buffer_size:
                self._events = self._events[-self.event_buffer_size:]
                self._event_trace_image_refs = self._event_trace_image_refs[-self.event_buffer_size:]
            needed = self._active_trace_image_keys | self._unscoped_trace_image_keys
            for keys in self._event_trace_image_refs:
                needed.update(keys)
            self._model_trace_images = {key: value for key, value in self._model_trace_images.items() if key in needed}

    def close(self) -> None:
        server = self._server
        if server is not None:
            server.should_exit = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def _snapshot_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    @staticmethod
    def _image_key(value: Any) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)

    @staticmethod
    def _trace_scope(payload: Dict[str, Any]) -> Optional[tuple[str, str]]:
        values = (payload.get('case_id'), payload.get('model_name'))
        return tuple(str(value) for value in values) if all(value not in (None, '') for value in values) else None

    @classmethod
    def _image_references(cls, value: Any):
        if isinstance(value, str) and value.startswith('agentark-trace-image:'):
            key = value.removeprefix('agentark-trace-image:')
            if cls._image_key(key):
                yield key
        elif isinstance(value, dict):
            for key, item in value.items():
                if key != 'model_trace_images':
                    yield from cls._image_references(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from cls._image_references(item)

    def _snapshot_state(self) -> Dict[str, Any]:
        with self._lock:
            # Assets arrive once per case. Keep the active case's full image
            # set for later tool rereads, plus references in buffered old calls.
            # Unscoped legacy streams conservatively retain a per-run cache.
            return {'events': list(self._events), 'model_trace_images': dict(self._model_trace_images)}
