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
h1 { margin: 0; font-size: 16px; font-weight: 700; }
.status { color: var(--muted); font-size: 13px; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 14px; padding: 14px; }
#chat { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
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
<header><h1>AgentArk Chat Viewer</h1><div class="status" id="status">connecting</div></header>
<main>
    <section id="chat"></section>
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
            <p class="help">Strict request and response payloads captured at the model boundary.</p>
            <div class="help">Request</div>
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
const rawEvents = [];
let seenEventSeqs = new Set();
let lastSeenSeq = 0;
let renderedMessageCounts = new Map();
let pendingAssistantByAgent = new Map();
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
    renderedMessageCounts = new Map();
    pendingAssistantByAgent = new Map();
    setTrace(rawRequestTraceEl, null, '(waiting for agent_request)');
    setTrace(rawResponseTraceEl, null, '(waiting for agent_response)');
    if (clearRawEvents) {
        rawEvents.length = 0;
        seenEventSeqs = new Set();
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
    row.innerHTML = `<div class="bubble"><div class="role">${roleLabel(role)}${meta}</div><div class="message-content">${renderContent(content)}</div></div>`;
    chatEl.appendChild(row);
}
function appendSeparator(text, ev) {
    const row = document.createElement('article');
    row.className = 'bubble-row separator';
    row.innerHTML = `<div class="bubble">${esc(buildSeparatorText(text, ev))}</div>`;
    chatEl.appendChild(row);
    renderedMessageCounts = new Map();
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
function renderObsMessages(obsMap, ev) {
    if (!obsMap || typeof obsMap !== 'object') return;
    for (const [agentId, obs] of Object.entries(obsMap)) {
        if (!obs || typeof obs !== 'object') continue;
        const messages = Array.isArray(obs.messages) ? obs.messages : [];
        if (messages.length) {
            let startIndex = renderedMessageCounts.get(agentId) || 0;
            if (messages.length < startIndex) startIndex = 0;
            for (let idx = startIndex; idx < messages.length; idx += 1) {
                const msg = messages[idx];
                if (!msg || typeof msg !== 'object') continue;
                const role = String(msg.role || 'user');
                const content = msg.content || '';
                const pendingAssistant = pendingAssistantByAgent.get(agentId);
                if (role === 'assistant' && pendingAssistant !== undefined && String(content) === pendingAssistant) {
                    pendingAssistantByAgent.delete(agentId);
                    continue;
                }
                appendBubble(role, content, {meta: buildBubbleMeta(agentId, ev)});
            }
            renderedMessageCounts.set(agentId, messages.length);
            continue;
        }
        const fallback = obsFallbackContent(obs);
        if (fallback) appendBubble('user', fallback, {meta: buildBubbleMeta(agentId, ev)});
    }
}
function firstActionPayload(actions) {
    if (!actions || typeof actions !== 'object') return null;
    const keys = Object.keys(actions).sort((a, b) => Number(a) - Number(b));
    for (const key of keys) {
        const payload = actions[key];
        if (payload && typeof payload === 'object') return {agentId: key, payload};
    }
    return null;
}
function renderAssistantResponse(ev) {
    const item = firstActionPayload(ev.payload && ev.payload.actions);
    if (!item) return;
    const text = item.payload.assistant || '';
    if (text) {
        pendingAssistantByAgent.set(String(item.agentId), String(text));
        appendBubble('assistant', text, {meta: buildBubbleMeta(item.agentId, ev)});
    }
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
function addEvent(ev) {
    const seq = Number(ev && ev.seq);
    if (ev && ev.event === 'run_start') {
        resetViewerState({clearRawEvents: true});
    } else if (Number.isFinite(seq) && seq > 0 && lastSeenSeq > 0 && seq <= lastSeenSeq) {
        resetViewerState({clearRawEvents: true});
    }
    if (Number.isFinite(seq) && seq > 0) {
        if (seenEventSeqs.has(seq)) return;
        seenEventSeqs.add(seq);
        lastSeenSeq = seq;
    }
    rawEvents.push(ev);
    eventCountEl.textContent = `${rawEvents.length} raw events stored`;
    const payload = ev.payload || {};
    updateRawTrace(ev);
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
  for (const ev of data.events || []) addEvent(ev);
}
loadState().catch(() => {});
const es = new EventSource('/events');
es.onopen = () => statusEl.textContent = 'connected';
es.onerror = () => statusEl.textContent = 'reconnecting';
es.onmessage = msg => { try { addEvent(JSON.parse(msg.data)); } catch (e) {} };
document.getElementById('submitAction').onclick = async () => {
  const action = document.getElementById('actionText').value;
  const agent_id = document.getElementById('agentId').value || null;
  await fetch('/human/actions', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({agent_id, action})});
};
document.getElementById('clearAction').onclick = () => document.getElementById('actionText').value = '';
document.getElementById('clearEvents').onclick = () => {
    resetViewerState({clearRawEvents: true});
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
            return {'events': self._snapshot_events()}

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
            self._events.append(to_jsonable(event))
            if len(self._events) > self.event_buffer_size:
                self._events = self._events[-self.event_buffer_size:]

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
