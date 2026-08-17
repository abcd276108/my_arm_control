const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

let recording = false;
let startedAt = 0;
let timerHandle = null;
let cameraOnline = false;
let initializerReady = false;
let initialized = false;
let currentStage = 1;
let maxStageReached = 1;
let initializing = false;
let initializationCommandSent = false;
let recordingBusy = false;
let activeTask = '';
let latestFusion = null;
let executionPolling = null;

const executionPanel = document.querySelector('[data-panel="6"]');
executionPanel.querySelector('.heading').innerHTML = '<p class="eyebrow">STEP 06</p><h2>實機執行</h2><p>確認工作區安全後執行已核准的軌跡。</p>';
executionPanel.querySelector('.workflow-card').innerHTML = `
  <strong id="executionTitle">等待操作員啟動</strong>
  <p id="executionMessage">請確認焊槍未通電、工作區無人員與障礙物，並準備隨時關閉手臂電源。</p>
  <label class="execution-check"><input type="checkbox" id="executionSafetyCheck"><span>我已確認工作區安全，並同意執行本次軌跡。</span></label>
  <label class="save-trajectory-option"><input type="checkbox" id="saveTrajectoryCheck"><span>執行完成後儲存路徑</span></label>
  <input class="trajectory-name-input" id="savedTrajectoryName" placeholder="路徑名稱（例如：標準點焊任務）" disabled>
  <p class="saved-trajectory-result" id="savedTrajectoryResult" hidden></p>
  <button class="record-button" id="executeTrajectoryButton" disabled>開始執行軌跡</button>
  <div id="executionProgress" hidden></div>
  <button class="ghost-button" id="executionBackButton">返回人員確認</button>`;

document.querySelector('[data-panel="4"] h2').textContent = '模擬驗證';
document.querySelector('[data-panel="5"] .workflow-card').innerHTML = `
  <strong>請確認模擬結果與預期路徑</strong>
  <label class="operator-note"><span>需要修改的部分（選填）</span><textarea id="operatorNote" rows="5" placeholder="例如：忽略第 3 點、調整第 2 點位置。"></textarea></label>
  <button class="record-button" id="approveExecutionButton">確認並前往實機執行</button>
  <button class="ghost-button" id="reviewBackButton">返回模擬驗證</button>`;

$('#savedTrajectoryName').insertAdjacentHTML(
  'afterend',
  '<button class="ghost-button" id="saveTrajectoryNowButton">儲存路徑</button>',
);
$('#executionBackButton').insertAdjacentHTML(
  'afterend',
  '<button class="ghost-button" id="executionHomeButton">返回主頁</button>',
);

const connectionPanel = document.querySelector('.connection');
connectionPanel.insertAdjacentHTML('beforeend', '<span class="dot execution-header-dot"></span><b>實機狀態</b><small id="headerExecutionState">待命</small>');

const arrivalConfirmButton = document.createElement('button');
arrivalConfirmButton.id = 'arrivalConfirmButton';
arrivalConfirmButton.textContent = '確認手臂已到位';
arrivalConfirmButton.hidden = true;
$('#initializeButton').insertAdjacentElement('afterend', arrivalConfirmButton);

const reinitializeButton = document.createElement('button');
reinitializeButton.className = 'ghost-button';
reinitializeButton.id = 'reinitializeButton';
reinitializeButton.textContent = '重新初始化手臂';
reinitializeButton.hidden = true;
$('#recordButton').insertAdjacentElement('afterend', reinitializeButton);

function showStage(stage) {
  currentStage = stage;
  maxStageReached = Math.max(maxStageReached, stage);
  $$('.stage').forEach((panel) => panel.classList.toggle('active', +panel.dataset.panel === stage));
  $$('.steps button').forEach((button) => {
    const number = +button.dataset.stage;
    button.classList.toggle('active', number === stage);
    button.classList.toggle('visited', number <= maxStageReached && number !== stage);
  });
  $('#globalBack').hidden = stage === 1;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function renderTaskPreview(result) {
  const preview = document.querySelector('[data-panel="3"] .empty-stage');
  preview.innerHTML = `<strong>RGB-D 示教資料已建立</strong><p>${result.task}</p><p>RGB ${result.rgb_count} 張 · Depth ${result.depth_count} 張 · ${result.duration_seconds} 秒</p><p id="inferenceStatus">下一步：使用手臂端訓練權重進行尖端推論並產生 XYZ 軌跡。</p><div class="preview-actions"><button class="record-button" id="runInferenceButton">執行推論並產生 XYZ</button><button class="ghost-button" id="recordAgainButton">重新示教</button></div>`;
  document.querySelector('#runInferenceButton').addEventListener('click', () => void runTaskInference(result.task));
  document.querySelector('#recordAgainButton').addEventListener('click', () => showStage(2));
  maxStageReached = Math.max(maxStageReached, 3);
}

function showRecordingDecision(result) {
  let panel = document.querySelector('#recordingDecision');
  if (!panel) {
    panel = document.createElement('div');
    panel.id = 'recordingDecision';
    panel.className = 'recording-decision';
    document.querySelector('#captureLayout .control-card').appendChild(panel);
  }
  panel.innerHTML = `<strong>示教錄製完成</strong><p>RGB ${result.rgb_count} 張 · Depth ${result.depth_count} 張 · ${result.duration_seconds} 秒</p><button class="record-button" id="continueToInferenceButton">前往推論</button><button class="ghost-button" id="retryRecordingButton">重新示教</button>`;
  panel.hidden = false;
  $('#continueToInferenceButton').addEventListener('click', () => showStage(3));
  $('#retryRecordingButton').addEventListener('click', () => {
    panel.hidden = true;
    $('#recordState').textContent = '尚未開始';
    $('#recordLabel').textContent = '準備完成';
    $('#timer').textContent = '00:00.0';
  });
}

async function addDownPointGallery(runName) {
  const preview = document.querySelector('[data-panel="3"] .preview-video');
  if (!preview) return;
  try {
    const response = await fetch(`/api/fusion-down-points?run=${encodeURIComponent(runName)}`, { cache: 'no-store' });
    const result = await response.json();
    if (!response.ok || !result.count) return;
    let index = 0;
    const gallery = document.createElement('section');
    gallery.className = 'down-point-gallery';
    gallery.innerHTML = `<div class="down-point-gallery-title" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px"><strong>辨識落下點</strong><span style="color:#676b6e;font-size:13px">推論完成後逐張確認</span></div><div class="down-point-image"><img id="downPointImage" alt="辨識落下點"><strong id="downPointCounter"></strong></div><div class="down-point-controls"><button class="ghost-button" id="previousDownPoint">上一張</button><button class="ghost-button" id="nextDownPoint">下一張</button></div>`;
    preview.insertAdjacentElement('afterbegin', gallery);
    const update = () => {
      $('#downPointImage').src = `/api/fusion-down-frame?run=${encodeURIComponent(runName)}&index=${index}&v=${Date.now()}`;
      $('#downPointCounter').textContent = `落下點 ${index + 1}/${result.count}`;
      $('#previousDownPoint').disabled = index === 0;
      $('#nextDownPoint').disabled = index === result.count - 1;
    };
    $('#previousDownPoint').addEventListener('click', () => { if (index > 0) { index -= 1; update(); } });
    $('#nextDownPoint').addEventListener('click', () => { if (index < result.count - 1) { index += 1; update(); } });
    update();
  } catch {}
}

async function restoreLatestTask() {
  try {
    const response = await fetch('/api/tasks/latest-runtime');
    const result = await response.json();
    if (response.ok && result.task) {
      localStorage.setItem('jetarmRuntimeLastTask', JSON.stringify(result));
      renderTaskPreview(result);
    }
  } catch {}
}

function restoreFusionPreview(result) {
  if (!result?.run_dir) return false;
  latestFusion = result;
  const runName = result.run_dir.split('/').filter(Boolean).pop();
  const preview = document.querySelector('[data-panel="3"] .empty-stage');
  if (!preview) return false;
  preview.className = 'trajectory-preview';
  preview.innerHTML = `<div class="preview-video preview-video-full"><video controls playsinline preload="metadata" src="/api/fusion-video?run=${encodeURIComponent(runName)}"></video><p>深紅為 DOWN、深藍為 UP。</p><button class="record-button" id="goSimulationButton">確認軌跡，進入模擬驗證</button><button class="ghost-button" id="rerunInferenceButton">重新推論</button></div>`;
  $('#goSimulationButton').addEventListener('click', () => {
    $('#simulationArchive').textContent = result.simulation_archive || '模擬資料已產生';
    showStage(4);
  });
  $('#rerunInferenceButton').addEventListener('click', () => {
    if (result.task) void runTaskInference(result.task);
  });
  maxStageReached = Math.max(maxStageReached, 4);
  showStage(3);
  return true;
}

function refreshControls() {
  const readyToRecord = cameraOnline && initialized && !initializing;
  $('#recordButton').disabled = !readyToRecord;
  $('#recordButton').textContent = readyToRecord
    ? (recording ? '結束示教' : '開始視覺示教')
    : (initialized ? '等待相機連線' : '等待初始化');
  $('#initializeButton').disabled = !initializerReady || initializing || initialized;
  $('#initializeButton').hidden = initializationCommandSent;
  arrivalConfirmButton.hidden = !initializationCommandSent || initialized;
  arrivalConfirmButton.disabled = initializing;
  reinitializeButton.hidden = !initialized && !initializationCommandSent;
  reinitializeButton.textContent = initialized ? '重新初始化手臂' : '手臂未移動，重新送出';
  reinitializeButton.disabled = !initializerReady || initializing || recording;
}

async function setRecording(active, cancelled = false) {
  if (!cameraOnline || !initialized || recordingBusy || active === recording) return false;
  recordingBusy = true;
  $('#recordButton').disabled = true;
  try {
    const response = await fetch(active ? '/api/recording/start' : '/api/recording/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(active ? { category: 'runtime_capture' } : { status: cancelled ? 'cancelled' : 'completed' }),
    });
    const result = await response.json();
    if (!response.ok || !result.success) throw new Error(result.detail || '錄製處理失敗');
    recording = active;
    if (active) {
      activeTask = result.task || '';
      latestFusion = null;
      localStorage.removeItem('jetarmRuntimeFusion');
    }
    if (!active && !cancelled) {
      localStorage.setItem('jetarmRuntimeLastTask', JSON.stringify(result));
      renderTaskPreview(result);
      showRecordingDecision(result);
    }
  } catch (error) {
    $('#recordState').textContent = `錄製失敗：${error.message}`;
    window.alert(`錄製失敗：${error.message}`);
    if (!active) {
      recording = false;
      clearInterval(timerHandle);
      timerHandle = null;
      $('.status-light').classList.remove('recording');
      $('#recordLabel').textContent = cancelled ? '已取消' : '錄製已停止';
      $('#cancelButton').disabled = true;
      if (!cancelled) {
        try {
          const latestResponse = await fetch('/api/tasks/latest-runtime', { cache: 'no-store' });
          const latest = await latestResponse.json();
          if (latestResponse.ok && latest.task && latest.rgb_count > 0) {
            localStorage.setItem('jetarmRuntimeLastTask', JSON.stringify(latest));
            renderTaskPreview(latest);
            showRecordingDecision(latest);
          }
        } catch {}
      }
      refreshControls();
    }
    return false;
  } finally {
    recordingBusy = false;
  }
  $('#recordState').textContent = active ? '錄製中' : '錄製完成';
  $('#recordLabel').textContent = active ? 'REC 視覺示教中' : '錄製完成';
  $('.status-light').classList.toggle('recording', active);
  $('#cancelButton').disabled = !active;
  if (active) {
    startedAt = Date.now();
    timerHandle = setInterval(() => {
      const elapsed = (Date.now() - startedAt) / 1000;
      const minutes = Math.floor(elapsed / 60);
      const seconds = (elapsed % 60).toFixed(1).padStart(4, '0');
      $('#timer').textContent = `${String(minutes).padStart(2, '0')}:${seconds}`;
    }, 100);
  } else {
    clearInterval(timerHandle);
  }
  refreshControls();
  return true;
}

async function runTaskInference(task) {
  const button = document.querySelector('#runInferenceButton');
  const status = document.querySelector('#inferenceStatus');
  button.disabled = true;
  button.textContent = '推論中…';
  status.innerHTML = '<span class="spinner inference-spinner"></span><strong>正在準備推論...</strong><br>請勿關閉頁面或停止手臂服務。';
  try {
    const modelsResponse = await fetch('/api/models');
    const modelsResult = await modelsResponse.json();
    const model = modelsResult.items?.[0];
    if (!model) {
      throw new Error('手臂端尚無 best.pt。請先將本地訓練權重上傳到 runs/smart_teaching_tip/模型名稱/weights/best.pt');
    }
    status.innerHTML = `<span class="spinner inference-spinner"></span><strong>尖端推論與 XYZ 分析中...</strong><br>使用權重：${model}<br>請勿關閉頁面或停止手臂服務。`;
    const response = await fetch('/api/fusion', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task, model }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '推論失敗');
    latestFusion = { ...result, task };
    localStorage.setItem('jetarmRuntimeFusion', JSON.stringify(latestFusion));
    const runName = result.run_dir.split('/').filter(Boolean).pop();
    const preview = document.querySelector('[data-panel="3"] .empty-stage');
    preview.className = 'trajectory-preview';
    preview.innerHTML = `<div class="preview-video preview-video-full"><video controls playsinline preload="metadata" src="/api/fusion-video?run=${encodeURIComponent(runName)}"></video><p>深紅為 DOWN、深藍為 UP。</p><button class="record-button" id="goSimulationButton">確認軌跡，進入模擬驗證</button><button class="ghost-button" id="rerunInferenceButton">重新推論</button></div>`;
    document.querySelector('#goSimulationButton').addEventListener('click', () => {
      $('#simulationArchive').textContent = result.simulation_archive;
      showStage(4);
    });
    document.querySelector('#rerunInferenceButton').addEventListener('click', () => void runTaskInference(task));
    status.innerHTML = `XYZ 軌跡已完成：<br>${result.run_dir}<br>Gazebo 交付檔：${result.simulation_archive}<br><strong>正在開啟軌跡回放…</strong>`;
    button.textContent = 'XYZ 已產生';
    maxStageReached = Math.max(maxStageReached, 4);
    showStage(3);
  } catch (error) {
    status.textContent = `尚未完成：${error.message}`;
    button.textContent = '重新執行推論並產生 XYZ';
    button.disabled = false;
  }
}

$('#simulationPassedButton').addEventListener('click', () => {
  showStage(5);
});
$('#simulationBackButton').addEventListener('click', () => showStage(3));
$('#reviewBackButton').addEventListener('click', () => showStage(4));
$('#executionBackButton').addEventListener('click', () => showStage(5));
$('#executionHomeButton').addEventListener('click', () => showStage(1));
$('#approveExecutionButton').addEventListener('click', () => {
  $('#executionSafetyCheck').checked = false;
  $('#executeTrajectoryButton').disabled = true;
  showStage(6);
});

$('#executionSafetyCheck').addEventListener('change', (event) => {
  $('#executeTrajectoryButton').disabled = !event.target.checked;
});

$('#saveTrajectoryCheck').addEventListener('change', (event) => {
  $('#savedTrajectoryName').disabled = !event.target.checked;
  if (event.target.checked) $('#savedTrajectoryName').focus();
});

function setExecutionHeader(running, failed = false) {
  const dot = document.querySelector('.execution-header-dot');
  dot.classList.toggle('executing', running);
  dot.classList.toggle('failed', failed);
  $('#headerExecutionState').textContent = running ? '手臂執行中' : (failed ? '執行異常' : '待命');
}

async function saveCurrentTrajectory(runName, force = false) {
  if (!force && !$('#saveTrajectoryCheck').checked) return;
  const response = await fetch('/api/robot/save-trajectory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run: runName, name: $('#savedTrajectoryName').value.trim() }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || '路徑儲存失敗');
  $('#savedTrajectoryResult').hidden = false;
  $('#savedTrajectoryResult').textContent = `路徑已儲存：${result.path}`;
}

$('#saveTrajectoryNowButton').addEventListener('click', async () => {
  const runName = latestFusion?.run_dir?.split('/').filter(Boolean).pop();
  if (!runName) {
    window.alert('目前沒有可儲存的路徑。');
    return;
  }
  const nameInput = $('#savedTrajectoryName');
  const name = nameInput.value.trim();
  if (!name) {
    window.alert('請先輸入路徑名稱。');
    nameInput.disabled = false;
    nameInput.focus();
    return;
  }
  try {
    await saveCurrentTrajectory(runName, true);
    $('#savedTrajectoryResult').dataset.saved = 'true';
  } catch (error) {
    $('#savedTrajectoryResult').hidden = false;
    $('#savedTrajectoryResult').textContent = error.message;
  }
});

async function pollExecutionStatus() {
  try {
    const response = await fetch('/api/robot/execution-status', { cache: 'no-store' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '無法取得實機執行狀態');
    const running = result.state === 'running';
    $('#executionProgress').hidden = !running;
    setExecutionHeader(running, result.state === 'failed');
    $('#executionTitle').textContent = running ? '執行中' : (result.state === 'completed' ? '執行完成' : (result.message || '等待操作員啟動'));
    $('#executionMessage').textContent = running
      ? '請勿進入工作區；若發現異常，立即關閉手臂電源。'
      : (result.state === 'completed' ? '' : (result.message || '等待執行'));
    if (!running) {
      clearInterval(executionPolling);
      executionPolling = null;
      if (result.state === 'completed' && !$('#savedTrajectoryResult').dataset.saved) {
        const runName = latestFusion?.run_dir?.split('/').filter(Boolean).pop();
        if (runName) {
          try {
            await saveCurrentTrajectory(runName);
            $('#savedTrajectoryResult').dataset.saved = 'true';
          } catch (error) {
            $('#savedTrajectoryResult').hidden = false;
            $('#savedTrajectoryResult').textContent = error.message;
          }
        }
      }
      $('#executeTrajectoryButton').disabled = result.state === 'completed' || !$('#executionSafetyCheck').checked;
      $('#executeTrajectoryButton').textContent = result.state === 'completed' ? '執行完成' : '重新執行軌跡';
    }
  } catch (error) {
    setExecutionHeader(false, true);
    $('#executionTitle').textContent = '狀態讀取失敗';
    $('#executionMessage').textContent = error.message;
  }
}

$('#executeTrajectoryButton').addEventListener('click', async () => {
  const runName = latestFusion?.run_dir?.split('/').filter(Boolean).pop();
  if (!runName) {
    window.alert('找不到本次推論軌跡，請返回軌跡預覽重新推論。');
    return;
  }
  if (!window.confirm('手臂將依序執行全部落下點。確認焊槍未通電、工作區已淨空，是否開始？')) return;
  const button = $('#executeTrajectoryButton');
  button.disabled = true;
  button.textContent = '啟動中…';
  $('#executionProgress').hidden = false;
  $('#savedTrajectoryResult').hidden = true;
  delete $('#savedTrajectoryResult').dataset.saved;
  setExecutionHeader(true);
  $('#executionTitle').textContent = '正在啟動軌跡';
  try {
    const response = await fetch('/api/robot/execute-trajectory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run: runName, operator_confirmed: true }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '實機軌跡啟動失敗');
    button.textContent = '軌跡執行中';
    await pollExecutionStatus();
    executionPolling = setInterval(() => void pollExecutionStatus(), 1500);
  } catch (error) {
    setExecutionHeader(false, true);
    $('#executionProgress').hidden = true;
    $('#executionTitle').textContent = '實機軌跡啟動失敗';
    $('#executionMessage').textContent = error.message;
    button.textContent = '重新執行軌跡';
    button.disabled = false;
  }
});

async function initializeRobot(force = false) {
  if (!initializerReady || initializing || (!force && initialized)) return;
  if (force && !window.confirm('手臂將再次移動至初始化姿態。請確認周圍無人員與障礙物，是否繼續？')) return;

  initializing = true;
  initialized = false;
  initializationCommandSent = false;
  $('#initializationOverlay').classList.remove('completed');
  $('#cameraView').classList.add('needs-initialization');
  $('#initializeSpinner').hidden = false;
  $('#initializeStatus').textContent = force ? '重新初始化中…' : '初始化中…';
  $('#initializeButton').textContent = '請稍候';
  $('#recordState').textContent = '尚未開始';
  refreshControls();

  try {
    const response = await fetch('/api/robot/initialize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });
    const result = await response.json();
    if (!response.ok || !result.success) {
      throw new Error(result.detail || result.message || '初始化失敗');
    }
    initializationCommandSent = true;
    $('#initializeStatus').textContent = '命令已送出，請確認手臂是否到位';
    $('#initializeSpinner').hidden = true;
  } catch (error) {
    initialized = false;
    initializationCommandSent = false;
    $('#initializationOverlay').classList.remove('completed');
    $('#cameraView').classList.add('needs-initialization');
    $('#initializeSpinner').hidden = true;
    $('#initializeStatus').textContent = `初始化失敗：${error.message}`;
    $('#initializeButton').textContent = '重新初始化';
  } finally {
    initializing = false;
    refreshControls();
  }
}

async function checkHealth() {
  try {
    const health = await fetch('/api/health').then((response) => response.json());
    $('#serviceState').textContent = health.status === 'ready' ? '已連線' : '異常';
    cameraOnline = health.camera_stream === 'ready';
    initializerReady = health.robot_initializer === 'ready';
    $('#cameraState').textContent = cameraOnline ? '已連線' : '等待串流橋接';
    if (cameraOnline && !$('#liveImage').src) {
      $('#liveImage').src = '/api/camera/stream';
      $('.camera-empty').hidden = true;
    }
    if (!initialized && !initializing && !initializationCommandSent) {
      $('#initializeButton').textContent = initializerReady ? '初始化手臂' : '控制服務未啟用';
      $('#initializeStatus').textContent = initializerReady
        ? '初始化後才能開始示教'
        : '實機初始化目前為安全鎖定';
    }
    refreshControls();
  } catch {
    $('#serviceState').textContent = '未連線';
    initializerReady = false;
    refreshControls();
  }
}

function loadSavedTasks() {
  let records = [];
  try {
    records = JSON.parse(localStorage.getItem('jetarmSavedTasks') || '[]');
  } catch {}
  const container = $('#runtimeSavedList');
  if (!records.length) {
    container.className = 'empty-stage';
    container.innerHTML = '<strong>尚無已儲存任務</strong><p>請返回選擇「視覺示教」，完成並儲存第一筆標準任務。</p><button class="back-button" data-return-home>返回選擇模式</button>';
    container.querySelector('[data-return-home]').addEventListener('click', () => showStage(1));
    return;
  }
  container.className = 'runtime-task-list';
  container.innerHTML = records.map((record, index) => `<button class="runtime-task" data-runtime-task="${index}"><strong>${record.name}</strong><small>${record.date} · ${record.valid} 個軌跡點</small></button>`).join('');
  container.querySelectorAll('[data-runtime-task]').forEach((item) => item.addEventListener('click', () => showStage(3)));
}

async function loadSavedTasksFromServer() {
  const container = $('#runtimeSavedList');
  container.className = 'empty-stage';
  container.innerHTML = '<strong>正在讀取已儲存任務…</strong>';
  try {
    const response = await fetch('/api/robot/saved-trajectories', { cache: 'no-store' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '無法讀取已儲存任務');
    const records = result.items || [];
    if (!records.length) {
      container.innerHTML = '<strong>尚無已儲存任務</strong><p>完成軌跡後，輸入名稱並按「儲存路徑」。</p>';
      return;
    }
    container.className = 'runtime-task-list';
    container.innerHTML = records.map((record, index) => `
      <button class="runtime-task" data-runtime-task="${index}">
        <strong>${record.name}</strong>
        <small>${record.saved_at || '已儲存'}</small>
      </button>
    `).join('');
    container.querySelectorAll('[data-runtime-task]').forEach((item) => {
      item.addEventListener('click', () => {
        const record = records[Number(item.dataset.runtimeTask)];
        if (record?.source_run) {
          latestFusion = {
            run_dir: `runs/infer_yolo_depth_fusion/${record.source_run}`,
            saved_trajectory: record.path,
          };
        }
        showStage(3);
      });
    });
  } catch (error) {
    container.innerHTML = `<strong>讀取失敗</strong><p>${error.message}</p>`;
  }
}

$$('.mode[data-mode]').forEach((button) => button.addEventListener('click', () => {
  const history = button.dataset.mode === 'history';
  $('#captureTitle').textContent = history ? '選擇已儲存任務' : '即時視覺示教';
  $('#captureLayout').hidden = history;
  $('#historyLayout').hidden = !history;
  if (history) void loadSavedTasksFromServer();
  showStage(2);
}));
$('#globalBack').addEventListener('click', () => showStage(Math.max(1, currentStage - 1)));
$$('.steps button').forEach((button) => button.addEventListener('click', () => {
  const stage = +button.dataset.stage;
  if (stage <= maxStageReached) showStage(stage);
}));
$('#initializeButton').addEventListener('click', () => initializeRobot(false));
reinitializeButton.addEventListener('click', () => initializeRobot(true));
arrivalConfirmButton.addEventListener('click', () => {
  initialized = true;
  initializationCommandSent = false;
  $('#initializationOverlay').classList.add('completed');
  $('#cameraView').classList.remove('needs-initialization');
  $('#recordLabel').textContent = cameraOnline ? '準備完成' : '等待相機';
  refreshControls();
});
$('#recordButton').addEventListener('click', () => void setRecording(!recording));
$('#cancelButton').addEventListener('click', async () => {
  const stopped = await setRecording(false, true);
  if (stopped) {
    $('#recordState').textContent = '已取消';
    $('#timer').textContent = '00:00.0';
  }
});
document.addEventListener('keydown', (event) => {
  if (event.code !== 'Space' || event.repeat || ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName) || !document.querySelector('[data-panel="2"]').classList.contains('active') || !cameraOnline || !initialized) return;
  event.preventDefault();
  void setRecording(!recording);
});

try {
  const savedFusion = JSON.parse(localStorage.getItem('jetarmRuntimeFusion') || 'null');
  const savedTask = JSON.parse(localStorage.getItem('jetarmRuntimeLastTask') || 'null');
  if (restoreFusionPreview(savedFusion)) {
    // Keep the completed inference and preview across browser refreshes.
  } else if (savedTask?.task) renderTaskPreview(savedTask);
  else void restoreLatestTask();
} catch {
  void restoreLatestTask();
}
checkHealth();
setInterval(checkHealth, 5000);
