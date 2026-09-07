(function () {
  const data = window.AGENTARK_REVIEW_DATA;
  const taskList = document.getElementById('taskList');
  const workspace = document.getElementById('workspace');
  const searchInput = document.getElementById('searchInput');
  const familyFilters = document.getElementById('familyFilters');
  const topStats = document.getElementById('topStats');
  let selectedId = Number(location.hash.replace('#task-', '')) || (data?.tasks?.[0]?.id ?? null);
  let selectedFamily = '全部';
  let playbackTimer = null;
  let reviewNotes = {};
  let noteSaveTimer = null;

  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const fmt = value => Number.isFinite(Number(value)) ? Number(value).toFixed(4).replace(/0+$/, '').replace(/\.$/, '') : String(value ?? '—');
  const taskById = id => data.tasks.find(task => task.id === Number(id));
  const familySet = ['全部', ...new Set(data.tasks.map(task => task.family))];

  topStats.textContent = `${data.task_count} TASKS · HIGH/LOW REPLAY · HF VERIFIED`;

  async function loadReviewNotes() {
    try {
      const response = await fetch('/api/review-notes', {cache: 'no-store'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      reviewNotes = payload.notes && typeof payload.notes === 'object' ? payload.notes : {};
    } catch (error) {
      console.warn('Review note API unavailable; using browser storage fallback.', error);
      reviewNotes = {};
    }
  }

  async function saveReviewNotes(statusElement) {
    try {
      const response = await fetch('/api/review-notes', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({notes: reviewNotes})
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      statusElement.textContent = '已保存到工作台目录的 review_notes.json。';
    } catch (error) {
      statusElement.textContent = '服务端保存失败；当前内容仍保存在本浏览器 localStorage。';
      console.warn('Unable to persist review notes to disk.', error);
    }
  }

  function renderFamilies() {
    familyFilters.innerHTML = familySet.map(family => `<button class="filter-chip ${family === selectedFamily ? 'active' : ''}" data-family="${esc(family)}">${esc(family)}</button>`).join('');
    familyFilters.querySelectorAll('button').forEach(button => button.onclick = () => {
      selectedFamily = button.dataset.family;
      renderFamilies();
      renderTaskList();
    });
  }

  function filteredTasks() {
    const query = searchInput.value.trim().toLowerCase();
    return data.tasks.filter(task => {
      const familyOk = selectedFamily === '全部' || task.family === selectedFamily;
      const haystack = `${task.id} ${task.name} ${task.title_zh} ${task.family} ${task.summary_zh}`.toLowerCase();
      return familyOk && (!query || haystack.includes(query));
    });
  }

  function renderTaskList() {
    const tasks = filteredTasks();
    taskList.innerHTML = tasks.map(task => `<button class="task-item ${task.id === selectedId ? 'active' : ''}" data-id="${task.id}">
      <span class="task-id">T${task.id}</span>
      <span class="task-label"><strong>${esc(task.title_zh)}</strong><span>${esc(task.name)}</span></span>
      <span class="modality">${task.semantic_modality === 'image' ? 'IMG' : 'TXT'}</span>
    </button>`).join('') || '<div class="empty-state">没有匹配任务</div>';
    taskList.querySelectorAll('.task-item').forEach(button => button.onclick = () => selectTask(Number(button.dataset.id)));
  }

  function chips(config) {
    const pairs = [
      ['观测', `${config.obs_mode} · ${config.width}×${config.height}`],
      ['动作', config.action_mode],
      ['轮次', `${config.max_attempts}×${config.max_steps_per_attempt}`],
      ['决策间隔', `${config.time_between_decisions}s`],
      ['时间倍速', config.time_scale],
      ['帧策略', config.video_frame_selection || '—']
    ];
    return pairs.map(([key, value]) => `<span class="chip">${esc(key)}: ${esc(value)}</span>`).join('');
  }

  function replayTab(replay, active) {
    const label = replay.kind === 'high' ? '高分 Replay' : '低分 Replay';
    return `<button class="tab ${active ? 'active' : ''}" data-replay="${replay.kind}">${label}<span class="score">${fmt(replay.score_reward)}</span></button>`;
  }

  function stepCards(replay) {
    return replay.steps.map(step => {
      const rewardClass = Number(step.reward) < 0 ? 'reward neg' : 'reward';
      return `<details class="step-card" data-step="${step.number}">
        <summary><span class="step-name">A${step.attempt} · S${step.attempt_step} · ${esc(step.tool.name)}</span><span class="${rewardClass}">${fmt(step.reward)}</span></summary>
        <div class="step-content"><p>${esc(step.explanation_zh)}</p><pre>${esc(step.tool.raw)}</pre>${step.after_text ? `<pre style="margin-top:8px">${esc(step.after_text)}</pre>` : ''}</div>
      </details>`;
    }).join('');
  }

  function renderReplay(task, kind) {
    const replay = task.replays.find(item => item.kind === kind) || task.replays[0];
    const unityClip = (task.diagnostic_clips || []).find(item => item.kind === replay.kind);
    const host = document.getElementById('replayBody');
    const hasFrames = replay.frames.length > 0;
    const frameNote = replay.has_agent_transition_video
      ? '这段播放的是该 agent 实际收到的连续帧；可拖动逐帧检查。'
      : task.semantic_modality === 'text'
        ? '该任务的语义观测是文本；兼容相机不作为评估证据，因此这里展示逐步文本。'
        : '该 replay 每个决策只保留一张 agent 可见帧；播放表现为真实决策状态切换，不伪造中间运动。';
    host.innerHTML = `
      <div class="replay-meta">
        <span>seed <b>${replay.seed}</b></span><span>score <b>${fmt(replay.score_reward)}</b></span>
        <span>attempts <b>${replay.max_attempts}</b></span><span>steps <b>${replay.steps.length}</b></span>
        <span>success <b>${replay.rollout_success}</b></span><span>truncated <b>${replay.rollout_truncated}</b></span>
        <span>frames <b>${replay.frame_count}</b></span>
      </div>
      ${hasFrames ? `<div class="frame-player">
        <div class="stage"><img id="frameImage" alt="Agent 实际观察帧"><span class="stage-label" id="stageLabel"></span></div>
        <div class="player-side">
          ${unityClip ? '<div class="source-toggle"><button class="source-button active" data-source="agent">Agent 实际观察</button><button class="source-button" data-source="unity">Unity 连续捕捉</button></div>' : ''}
          <div class="controls"><button class="play" id="playButton" aria-label="播放或暂停">▶</button><input type="range" id="frameSlider" min="0" max="${Math.max(0, replay.frames.length - 1)}" value="0"><span class="frame-count" id="frameCount"></span></div>
          <p class="frame-note" id="frameNote">${esc(frameNote)}</p>
          <div class="step-list">${stepCards(replay)}</div>
        </div>
      </div>` : `<div class="frame-player"><div class="text-only"><pre id="textStage"></pre></div><div class="player-side"><p class="frame-note">${esc(frameNote)}</p><div class="step-list">${stepCards(replay)}</div></div></div>`}`;
    if (hasFrames) {
      wireFramePlayer(replay.frames, replay.has_agent_transition_video, frameNote);
      document.querySelectorAll('.source-button').forEach(button => button.onclick = () => {
        document.querySelectorAll('.source-button').forEach(item => item.classList.toggle('active', item === button));
        if (button.dataset.source === 'unity') {
          wireFramePlayer(unityClip.frames, true, unityClip.note_zh);
        } else {
          wireFramePlayer(replay.frames, replay.has_agent_transition_video, frameNote);
        }
      });
    }
    else {
      const text = replay.steps.map(step => `${step.explanation_zh}\n\n${step.after_text}`).join('\n\n────────\n\n');
      document.getElementById('textStage').textContent = text;
    }
  }

  function wireFramePlayer(frames, isContinuous, note) {
    if (playbackTimer) { clearInterval(playbackTimer); playbackTimer = null; }
    const image = document.getElementById('frameImage');
    const slider = document.getElementById('frameSlider');
    const count = document.getElementById('frameCount');
    const label = document.getElementById('stageLabel');
    const playButton = document.getElementById('playButton');
    const noteElement = document.getElementById('frameNote');
    let index = 0;
    slider.max = Math.max(0, frames.length - 1);
    noteElement.textContent = note;
    const show = next => {
      index = Math.max(0, Math.min(frames.length - 1, Number(next)));
      const frame = frames[index];
      image.src = frame.path;
      slider.value = index;
      count.textContent = `${index + 1}/${frames.length}`;
      label.textContent = `A${frame.attempt} · S${frame.step} · ${frame.phase} · cam ${frame.camera} · ${frame.width}×${frame.height}`;
      document.querySelectorAll('.step-card').forEach(card => card.open = Number(card.dataset.step) === frame.step);
    };
    const stop = () => { if (playbackTimer) clearInterval(playbackTimer); playbackTimer = null; playButton.textContent = '▶'; };
    playButton.onclick = () => {
      if (playbackTimer) return stop();
      playButton.textContent = '❚❚';
      playbackTimer = setInterval(() => {
        if (index >= frames.length - 1) { stop(); return; }
        show(index + 1);
      }, isContinuous ? 80 : 650);
    };
    slider.oninput = () => { stop(); show(slider.value); };
    show(0);
  }

  function standardSection(task) {
    const standard = task.standard_trajectory;
    if (!standard.exists) return '<p class="frame-note">开发目录未留下 action_trajectories.md。</p>';
    const actions = standard.actions.map(action => `<div class="standard-action"><strong>${esc(action.section)} · ${action.number}. ${esc(action.tool.name)}</strong><br>${esc(action.explanation_zh)}<pre style="margin-top:6px">${esc(action.tool.raw)}</pre></div>`).join('');
    return `<div class="standard-actions">${actions || '<p class="frame-note">文档存在，但没有可解析的 tool_call。</p>'}</div><pre>${esc(standard.markdown)}</pre>`;
  }

  function selectTask(id) {
    if (playbackTimer) { clearInterval(playbackTimer); playbackTimer = null; }
    selectedId = id;
    location.hash = `task-${id}`;
    renderTaskList();
    const task = taskById(id);
    workspace.innerHTML = `
      <div class="task-head">
        <div><div class="eyebrow">TASK ${task.id} · ${esc(task.family)}</div><h1>${esc(task.title_zh)}</h1><p class="summary">${esc(task.summary_zh)}</p></div>
        <a class="source-link" href="${esc(task.hf.dataset_url)}" target="_blank" rel="noreferrer">HF SOURCE ↗</a>
      </div>
      <div class="grid two">
        <section class="card"><div class="card-head"><h2>怎么玩</h2></div><div class="card-body"><ol class="clean-list">${task.play_zh.map(item => `<li>${esc(item)}</li>`).join('')}</ol></div></section>
        <section class="card"><div class="card-head"><h2>人工审核重点</h2></div><div class="card-body"><ul class="clean-list audit-list">${task.audit_zh.map(item => `<li>${esc(item)}</li>`).join('')}</ul></div></section>
      </div>
      <div class="grid" style="margin-top:14px"><section class="card"><div class="card-head"><h2>运行契约</h2><span class="modality">${task.semantic_modality.toUpperCase()}</span></div><div class="card-body"><div class="chips">${chips(task.config)}</div></div></section></div>
      <section class="card replay-section" style="max-width:1220px;margin-left:auto;margin-right:auto">
        <div class="card-head"><h2>真实 Replay 对照</h2><span class="modality">MODEL ${esc(task.hf.model || '')}</span></div>
        <div class="replay-tabs">${task.replays.map((replay, idx) => replayTab(replay, idx === 0)).join('')}</div>
        <div id="replayBody"></div>
      </section>
      <details class="doc"><summary>标准 action 轨迹（开发时留存）</summary><div>${standardSection(task)}</div></details>
      <details class="doc"><summary>Agent 原始英文任务说明</summary><div><pre>${esc(task.prompt_en)}</pre></div></details>
      <details class="doc"><summary>审核笔记（自动持久化）</summary><div><textarea class="review-note" id="reviewNote" placeholder="记录画面、机制、难度或价值判断…"></textarea><div class="note-status" id="noteStatus">输入会自动保存到工作台目录；浏览器 localStorage 同时作为后备。</div></div></details>`;
    document.querySelectorAll('.tab').forEach(button => button.onclick = () => {
      document.querySelectorAll('.tab').forEach(item => item.classList.toggle('active', item === button));
      if (playbackTimer) { clearInterval(playbackTimer); playbackTimer = null; }
      renderReplay(task, button.dataset.replay);
    });
    renderReplay(task, 'high');
    const note = document.getElementById('reviewNote');
    const status = document.getElementById('noteStatus');
    const noteKey = String(task.id);
    const browserKey = `agentark-review-note-${task.id}`;
    const browserValue = localStorage.getItem(browserKey) || '';
    if (reviewNotes[noteKey] === undefined && browserValue) {
      reviewNotes[noteKey] = browserValue;
      saveReviewNotes(status);
    }
    note.value = reviewNotes[noteKey] ?? browserValue;
    note.oninput = () => {
      reviewNotes[noteKey] = note.value;
      localStorage.setItem(browserKey, note.value);
      status.textContent = '正在保存…';
      if (noteSaveTimer) clearTimeout(noteSaveTimer);
      noteSaveTimer = setTimeout(() => saveReviewNotes(status), 300);
    };
    workspace.scrollTop = 0;
  }

  searchInput.oninput = renderTaskList;
  window.addEventListener('hashchange', () => {
    const id = Number(location.hash.replace('#task-', ''));
    if (taskById(id) && id !== selectedId) selectTask(id);
  });
  loadReviewNotes().finally(() => {
    renderFamilies();
    renderTaskList();
    if (selectedId) selectTask(selectedId);
  });
})();
