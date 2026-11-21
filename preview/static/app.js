'use strict';
const $ = id => document.getElementById(id);
const names = {q1: 'Q1', bumi: 'Bumi', g1: 'G1', all: 'Q1 / Bumi / G1'};
let state = {}, searching = 0, searchTimer, statusTimer, statsTimer;
let peer, peerGeneration, receivedStream, reconnectTimer, handshakeTimer, cancelIce, offerAbort, retryCount = 0;
let stopped = false, frameReady = false, streamDecoded = false, frameCallback, lastStats, pendingLoad, controlError = '';
let connectWhenReady = true;
let picoUploading = false, picoRecords = [], picoEnabled = false;
let connectionMessage = '正在连接…';

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
function selectedRobot() { return $('mode').value === 'compare' ? 'all' : $('robot').value; }
function selectedRelease() { return (state.releases || []).find(r => r.id===$('release').value); }
function motionDecodeRelease() {
  const releases = state.releases || [];
  return releases.find(r => r.interface==='motion' && r.joint_training) || releases.find(r => r.id==='motiondecode-30000');
}
function standupRelease() {
  const releases = state.releases || [];
  const selected = selectedRelease();
  if (selected?.interface==='human' && selected.neutral_standing) return selected;
  return releases.find(r => r.interface==='motion' && r.neutral_standing) || releases.find(r => r.id==='joint-motion-6000') || releases.find(r => r.id==='joint-motion-5000');
}
function flipRelease() {
  const selected = selectedRelease();
  return selected?.flips ? selected : (state.releases || []).find(r => r.flips && r.interface==='motion');
}
function loading() { return state.loading || !!pendingLoad; }
function updateVideo() {
  const visible = !loading() && state.frame_available && frameReady;
  $('frame').style.display = visible ? 'block' : 'none';
  $('empty').hidden = !!visible;
  $('empty').textContent = loading() ? '正在载入策略与物理环境…' : !state.robot ? '选择模式和动作，再点击「载入动作」。' : state.error ? '仿真载入失败，请查看错误信息。' : !state.frame_available ? '等待当前会话画面…' : '等待 WebRTC 视频画面；连接失败时可点击「重连视频」。';
  const rtc = peer ? `${peer.connectionState} / ICE ${peer.iceConnectionState || '—'}` : '未连接';
  $('connection').textContent = `WebRTC · ${rtc} · ${connectionMessage}`;
}
function markVideoReady(pc, stream, decoded = false) {
  const video = $('frame');
  if (!stopped && pc && pc === peer && peerGeneration === (state.session_generation ?? 0) && stream && stream === receivedStream && video.srcObject === stream && video.readyState >= 2) {
    if (decoded) streamDecoded = true;
    if (streamDecoded && !loading() && state.frame_available) frameReady = true;
  }
  updateVideo();
}
function updateButtons() {
  $('pico-upload').disabled = !picoEnabled || picoUploading;
  $('pico-load').disabled = loading() || picoUploading || selectedRelease()?.interface !== 'human' || !$('pico-recordings').value;
  const recovery = standupRelease();
  $('standup').disabled=loading() || !recovery;
  $('standup-pose').disabled=loading();
  $('standup').textContent='站起来：载入姿态'+(recovery ? ' · '+recovery.id.replace(/^joint-(motion|human)-/, '')+(recovery.interface==='human' ? ' · 人体五点' : '') : '');
  const hasMotionDecode=!!motionDecodeRelease();
  $('motiondecode-actions').hidden=!hasMotionDecode;
  for (const id of ['motiondecode-baji','motiondecode-dance']) $(id).disabled=loading() || !hasMotionDecode;
  $('flip-actions').hidden=!flipRelease();
  $('flips').disabled=loading() || !flipRelease();
  const checkpoints = state.checkpoints || {};
  const configured = selectedRobot() === 'all' ? ['q1', 'bumi', 'g1'].every(robot => checkpoints[robot]) : checkpoints[selectedRobot()];
  $('select').disabled = loading() || !configured || !$('motions').value;
  $('start').disabled = loading() || !state.robot || state.playing || state.failed || state.completed || !!state.error;
  $('pause').disabled = !state.playing || loading();
  $('reset').disabled = loading() || !state.robot || !!state.error;
}
async function control(action, values = {}) {
  if (action === 'select' || action === 'standup') {
    pendingLoad = {baseGeneration: state.session_generation ?? 0};
    frameReady = false;
    updateVideo(); updateButtons();
  }
  try {
    await request('/api/control', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action, ...values})});
    controlError = '';
  } catch (error) {
    controlError = error.message;
    if (action === 'select' || action === 'standup') pendingLoad = null;
    updateButtons(); updateVideo();
  }
  $('error').textContent = state.error || controlError;
}
async function motions() {
  const sequence = ++searching;
  $('motions').replaceChildren(); updateButtons();
  try {
    const release = $('release').value;
    const result = await request(`/api/motions?robot=${encodeURIComponent(selectedRobot())}&q=${encodeURIComponent($('search').value)}${release ? '&release='+encodeURIComponent(release) : ''}`);
    if (stopped || sequence !== searching) return;
    $('motions').replaceChildren(...result.motions.map(name => new Option(name, name)));
    $('count').textContent = `${result.total.toLocaleString()} 个匹配 · 最多显示 200 个`;
    if (result.motions.length) $('motions').selectedIndex = 0;
    updateButtons();
  } catch (error) { $('error').textContent = error.message; }
}
function chooseMotionDecode(category) {
  const release=motionDecodeRelease();
  if (loading() || !release) return;
  $('release').value=release.id;
  $('search').value='motiondecode_'+category;
  return motions();
}
$('standup').addEventListener('click', () => {
  const release=standupRelease();
  if (loading() || !release) return;
  const pose=$('standup-pose').value || 'lying';
  $('release').value=release.id;
  $('search').value=pose==='half_squat' ? 'fallAndGetUp1_subject5' : 'fallAndGetUp2_subject2';
  motions();
  return control('standup', {robot:selectedRobot(), release:release.id, pose});
});
$('flips').addEventListener('click',()=>{
  const release=flipRelease();
  if (loading() || !release) return;
  $('release').value=release.id;
  $('search').value='seed_flip_';
  return motions();
});
$('motiondecode-baji').addEventListener('click',()=>chooseMotionDecode('baji'));
$('motiondecode-dance').addEventListener('click',()=>chooseMotionDecode('dance'));
$('mode').addEventListener('change', () => { $('robot').disabled = $('mode').value === 'compare'; return motions(); });
$('robot').addEventListener('change', motions);
$('release').addEventListener('change', motions);
$('motions').addEventListener('change', updateButtons);
$('search').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(motions, 180); });
$('select').addEventListener('click', () => control('select', {robot: selectedRobot(), motion: $('motions').value,...($('release').value ? {release:$('release').value} : {}),...(selectedRelease()?.interface==='human' ? {human_points:$('human-points').value} : {})}));
async function recordings(selectId) {
  const result = await request('/api/recordings');
  picoEnabled = result.enabled;
  picoRecords = result.recordings;
  const selected = selectId || $('pico-recordings').value;
  $('pico-recordings').replaceChildren(...picoRecords.map(r => new Option(r.filename+' · '+r.duration_seconds.toFixed(2)+'秒',r.id)));
  if (picoRecords.some(r => r.id===selected)) $('pico-recordings').value=selected;
  else if (picoRecords.length) $('pico-recordings').selectedIndex=0;
  updateButtons();
}
$('pico-upload').addEventListener('click', () => $('pico-file').click());
$('pico-recordings').addEventListener('change', updateButtons);
$('pico-file').addEventListener('change', async () => {
  const file = $('pico-file').files[0];
  if (!file) return;
  picoUploading = true; updateButtons();
  $('pico-status').textContent = '正在上传并校验 '+file.name+'…';
  try {
    if (file.size > 32*1024*1024) throw new Error('PICO 文件不能超过 32 MB');
    const result = await request('/api/recordings', {method:'POST',
      headers:{'Content-Type':'application/x-ndjson','X-Filename':encodeURIComponent(file.name)}, body:file});
    await recordings(result.recording.id);
    $('pico-status').textContent = '已保存 '+result.recording.filename+'，选中后点击「载入选中 PICO 录制」。';
  } catch (error) { $('pico-status').textContent = '上传失败：'+error.message; }
  finally { picoUploading = false; $('pico-file').value=''; updateButtons(); }
});
$('pico-load').addEventListener('click', () => {
  const record = picoRecords.find(r => r.id===$('pico-recordings').value);
  if (!record || selectedRelease()?.interface!=='human') return;
  $('human-points').value='five';
  return control('select', {robot:selectedRobot(),motion:'pico',release:$('release').value,
    human_points:'five',recording:record.id});
});
for (const action of ['start', 'pause', 'reset']) $(action).addEventListener('click', () => control(action));
function number(value, digits = 1) { return Number.isFinite(value) ? value.toFixed(digits) : '—'; }
function terminal(participant) {
  return participant.failed ? `物理失败：${(participant.termination_terms || []).join(', ') || participant.error || '已停止'}` : participant.completed ? '动作完成' : (participant.playing ?? state.playing) ? '运行中' : '已暂停';
}
function renderStatus(next) {
  if (JSON.stringify(next.releases) !== JSON.stringify(state.releases)) {
    const selected = $('release').value;
    $('release').replaceChildren(...(next.releases || []).map(r => new Option(r.label,r.id)));
    if ((next.releases || []).some(r => r.id===selected)) $('release').value=selected;
    else if (next.release_id) $('release').value=next.release_id;
  }
  const release = (next.releases || []).find(r => r.id===$('release').value);
  $('human-points').disabled = release?.interface!=='human';
  $('release-note').textContent = release?.interface==='human' ? (release.joint_training ? '动作与人体联合训练；人体当前帧输入。MotionDecode 暂缺人体原始数据。' : '原 motion 30000 教师 + 独立人体更新；人体当前帧输入。MotionDecode 暂缺人体原始数据。') : release?.id==='motiondecode-30000' ? 'MotionDecode 使用原 Motion 30000 权重；暂无人体配对数据。点击下方八极拳或舞蹈，再载入动作。' : release?.joint_training ? '动作与人体联合训练；Motion 输入包含已准备的机器人参考动作，MotionDecode 暂缺人体配对数据。' : 'Motion 输入：包含已准备的机器人参考动作。';
  const generation = next.session_generation ?? 0;
  if (generation !== (state.session_generation ?? 0)) {
    closePeer(); retryCount = 0; connectWhenReady = true;
    connectionMessage = '等待当前会话准备完成…';
  }
  if (pendingLoad && !next.loading && generation > pendingLoad.baseGeneration) pendingLoad = null;
  if (next.loading || !next.frame_available || next.robot !== state.robot || next.motion !== state.motion || next.mode !== state.mode) frameReady = false;
  state = next;
  const compare = state.mode === 'compare' || state.robot === 'all';
  const participants = state.participants || [];
  const standingHold = state.standing_hold_frame != null || participants.some(p => p.standing_hold_frame != null);
  const activeRelease = (state.releases || []).find(r => r.id===state.release_id);
  const inputMode = state.input_mode || participants[0]?.input_mode;
  $('active').textContent = state.robot ? `当前运行：${compare ? '三本体同屏' : '单本体'} · ${names[state.robot] || state.robot} · ${picoRecords.find(r => r.id===state.recording_id)?.filename || state.motion || '—'}${activeRelease ? ' · '+activeRelease.label : ''}${inputMode ? ' · '+inputMode : ''}` : '当前运行：尚未载入';
  $('runtime').textContent = loading() ? '正在载入固定策略与物理环境…' : state.error ? '仿真错误，请重新载入动作' : state.failed ? '物理失败，整个会话已暂停；点击「重置」后才能继续' : state.completed ? '动作完成；点击「重置」可重新播放' : state.playing ? '策略驱动物理仿真 · 播放中' : state.robot ? '仿真已暂停' : '尚未载入仿真';
  $('legend').textContent = standingHold ? '实色：实际策略与物理。半透明：已审核的自然垂手站立目标；目标保持时物理继续运行。' : compare ? 'Q1 / Bumi / G1 均为实际策略驱动的物理本体，保留真实尺寸；同源舞蹈主体同步。' : '实色：实际策略与物理。半透明：参考目标。';
  $('progress').textContent = `${state.steps || 0} 控制步 / ${standingHold ? '持续站立目标' : (state.clip_frames || '—')+' 帧'} · ${number(state.simulated_seconds || 0, 2)} 仿真秒`;
  $('physics-stats').textContent = `实测物理 ${number(state.simulation_hz)} Hz · 渲染 ${number(state.render_fps)} fps · 仿真速度 ${number(state.sim_speed, 2)}×`;
  const failures = participants.filter(participant => participant.failed).map(participant => `${names[participant.robot] || participant.robot} ${terminal(participant)}`);
  $('failure').textContent = failures.length ? failures.join('；') : terminal(state);
  $('failure').className = state.failed ? 'failed' : '';
  $('participants').textContent = participants.map(participant => `${names[participant.robot] || participant.robot} · ${terminal(participant)} · ${participant.steps || 0} 步`).join('\n');
  const checkpoints = compare ? ['q1', 'bumi', 'g1'].map(robot => [robot, (state.checkpoints || {})[robot]]) : [[state.robot, state.checkpoint || (state.checkpoints || {})[state.robot]]];
  $('checkpoint').textContent = checkpoints.filter(([, checkpoint]) => checkpoint).map(([robot, checkpoint]) => `${names[robot] || robot}: ${checkpoint.path || '—'}`).join('\n') || '尚未载入固定检查点';
  $('iteration').textContent = checkpoints.filter(([, checkpoint]) => checkpoint).map(([robot, checkpoint]) => `${names[robot] || robot}: ${checkpoint.iteration == null ? '—' : checkpoint.iteration.toLocaleString()}`).join(' · ') || '—';
  $('source').textContent = compare ? [
    standingHold ? '各本体从审核姿态恢复，最终站立目标持续保持；物理继续运行。' : `共同主体：t=0 同步，在共同最短时长 ${number(state.source_duration_seconds, 2)} 秒停止。`,
    ...participants.map(participant => {
      const window = participant.source_window || {};
      const start = window.start_frame ?? participant.source_start_frame;
      const stop = window.stop_frame ?? participant.source_stop_frame;
      const path = participant.source_metadata?.source_id || participant.source_path || state.source?.[participant.robot] || '—';
      const tail = state.tail_frames?.[participant.robot];
      return `${names[participant.robot] || participant.robot}：源帧 [${start ?? '—'}, ${stop ?? '—'}) / ${window.source_frames ?? participant.source_frames ?? '—'}；起点偏移 ${number(window.offset_seconds, 2)} 秒${tail ? `；尾部未播放 ${tail} 帧` : ''}\n${path}`;
    }),
  ].join('\n') : state.source || '—';
  $('identity').textContent = state.data_manifest_sha256 || '—';
  $('error').textContent = state.error || controlError;
  updateButtons(); updateVideo();
  if (connectWhenReady && !loading() && (state.frame_available || (generation === 0 && !state.robot))) connectVideo();
}
async function refresh() {
  try {
    const next = await request('/api/status');
    if (!stopped) renderStatus(next);
  } catch (error) { if (!stopped) $('runtime').textContent = `状态服务连接失败：${error.message}`; }
  if (!stopped) statusTimer = setTimeout(refresh, 500);
}
function closePeer() {
  clearTimeout(reconnectTimer); clearTimeout(handshakeTimer);
  reconnectTimer = handshakeTimer = null;
  const old = peer; peer = null; peerGeneration = null; receivedStream = null;
  if (frameCallback != null && typeof $('frame').cancelVideoFrameCallback === 'function') $('frame').cancelVideoFrameCallback(frameCallback);
  frameCallback = null;
  if (cancelIce) cancelIce();
  if (offerAbort) offerAbort.abort();
  offerAbort = null;
  if (old) old.close();
  const stream = $('frame').srcObject;
  if (stream) stream.getTracks().forEach(track => track.stop());
  $('frame').srcObject = null;
  frameReady = streamDecoded = false; lastStats = null;
  $('stream-stats').textContent = '接收 / 解码帧率：等待视频统计';
}
function retryVideo(pc, message) {
  if (stopped || peer !== pc) return;
  closePeer();
  if (retryCount < 3) {
    retryCount++;
    connectionMessage = `${message}；正在重试 ${retryCount}/3`;
    reconnectTimer = setTimeout(connectVideo, retryCount * 1500);
  } else connectionMessage = `${message}；自动重试已停止，请点击「重连视频」`;
  updateVideo();
}
async function gatherIce(pc) {
  if (pc.iceGatheringState === 'complete') return;
  await new Promise((resolve, reject) => {
    const finish = error => {
      clearTimeout(timer); pc.removeEventListener('icegatheringstatechange', changed); cancelIce = null;
      if (error) reject(error); else resolve();
    };
    const changed = () => { if (pc.iceGatheringState === 'complete') finish(); };
    const timer = setTimeout(() => finish(new Error('ICE 候选收集超时')), 8000);
    cancelIce = () => finish(new Error('连接已关闭'));
    pc.addEventListener('icegatheringstatechange', changed);
  });
}
function receiveOffer(offer) {
  if (typeof MediaSource === 'undefined' || typeof MediaSource.isTypeSupported !== 'function') return offer;
  const sdp = offer.sdp.replace(/^a=fmtp:[^\r\n]+/gm, line => {
    const match = line.match(/profile-level-id=([0-9a-f]{6})/i);
    if (!match || parseInt(match[1].slice(-2), 16) >= 40) return line;
    const profile = match[1].toLowerCase();
    // RFC6184 max-recv-level extends reception without changing the native default profile/level.
    if (!MediaSource.isTypeSupported(`video/mp4; codecs="avc1.${profile.slice(0, 4)}28"`)) return line;
    return line.replace(/;?\s*max-recv-level=[^;\s]+/gi, '') + `;max-recv-level=${profile.slice(2, 4)}28`;
  });
  return {...offer, sdp};
}
async function connectVideo() {
  if (stopped) return;
  closePeer();
  if (loading() || (state.robot && !state.frame_available)) {
    connectWhenReady = true; connectionMessage = '等待当前会话准备完成…'; updateVideo(); return;
  }
  connectWhenReady = false;
  if (typeof RTCPeerConnection !== 'function') {
    connectionMessage = '此浏览器不支持 WebRTC，请使用支持 WebRTC 的浏览器'; updateVideo(); return;
  }
  let pc;
  try {
    pc = new RTCPeerConnection({iceServers: []}); peer = pc; peerGeneration = state.session_generation ?? 0;
    connectionMessage = '正在协商视频连接…'; updateVideo();
    pc.addTransceiver('video', {direction: 'recvonly'});
    pc.addEventListener('track', event => {
      if (stopped || peer !== pc) return;
      const video = $('frame'), stream = event.streams[0] || new MediaStream([event.track]);
      receivedStream = stream; streamDecoded = false; video.srcObject = stream;
      if (typeof video.requestVideoFrameCallback === 'function') {
        const decodedFrame = () => {
          if (stopped || peer !== pc || receivedStream !== stream) return;
          markVideoReady(pc, stream, true); frameCallback = video.requestVideoFrameCallback(decodedFrame);
        };
        frameCallback = video.requestVideoFrameCallback(decodedFrame);
      }
      $('frame').play().catch(error => { if (peer === pc) { connectionMessage = `视频播放失败：${error.message}`; updateVideo(); } });
    });
    pc.addEventListener('connectionstatechange', () => {
      if (stopped || peer !== pc) return;
      if (pc.connectionState === 'connected') {
        clearTimeout(handshakeTimer); retryCount = 0; connectionMessage = '视频已连接';
      } else if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
        retryVideo(pc, `连接${pc.connectionState === 'failed' ? '失败' : '中断'}`); return;
      }
      updateVideo();
    });
    pc.addEventListener('iceconnectionstatechange', updateVideo);
    handshakeTimer = setTimeout(() => retryVideo(pc, '连接超时'), 18000);
    await pc.setLocalDescription(receiveOffer(await pc.createOffer()));
    await gatherIce(pc);
    if (stopped || peer !== pc) return;
    offerAbort = typeof AbortController === 'function' ? new AbortController() : null;
    const answer = await request('/api/offer', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({sdp: pc.localDescription.sdp, type: 'offer'}), ...(offerAbort ? {signal: offerAbort.signal} : {})});
    if (stopped || peer !== pc) return;
    await pc.setRemoteDescription(answer);
  } catch (error) {
    if (pc) retryVideo(pc, error.message);
    else { connectionMessage = `无法创建 WebRTC 连接：${error.message}`; updateVideo(); }
  }
}
async function refreshStats() {
  const pc = peer;
  if (pc && typeof pc.getStats === 'function') {
    try {
      const reports = await pc.getStats();
      if (stopped) return;
      if (peer !== pc) { statsTimer = setTimeout(refreshStats, 1000); return; }
      let inbound, pair, selectedPair;
      reports.forEach(report => { if (report.type === 'transport') selectedPair = report.selectedCandidatePairId; });
      reports.forEach(report => {
        if (report.type === 'inbound-rtp' && (report.kind === 'video' || report.mediaType === 'video')) inbound = report;
        if (report.type === 'candidate-pair' && (report.id === selectedPair || (report.nominated && report.state === 'succeeded'))) pair = report;
      });
      if (inbound) {
        const seconds = lastStats ? (inbound.timestamp - lastStats.timestamp) / 1000 : 0;
        const received = seconds > 0 ? (inbound.framesReceived - lastStats.framesReceived) / seconds : undefined;
        const decoded = seconds > 0 ? (inbound.framesDecoded - lastStats.framesDecoded) / seconds : inbound.framesPerSecond;
        const bitrate = seconds > 0 && Number.isFinite(lastStats.bytesReceived) ? (inbound.bytesReceived - lastStats.bytesReceived) * 8 / seconds / 1e6 : undefined;
        if (inbound.framesDecoded > 0) markVideoReady(pc, receivedStream, true);
        lastStats = {timestamp: inbound.timestamp, framesReceived: inbound.framesReceived, framesDecoded: inbound.framesDecoded, bytesReceived: inbound.bytesReceived};
        const detail = inbound.frameWidth && inbound.frameHeight ? ` · ${inbound.frameWidth}×${inbound.frameHeight} · ${number(bitrate, 2)} Mbps` : '';
        $('stream-stats').textContent = `接收 ${number(received)} fps · 解码 ${number(decoded)} fps · RTT ${pair ? number(pair.currentRoundTripTime * 1000, 0) : '—'} ms${detail}`;
      }
    } catch (_) { $('stream-stats').textContent = '浏览器暂未提供视频接收统计'; }
  }
  if (!stopped) statsTimer = setTimeout(refreshStats, 1000);
}
$('reconnect').addEventListener('click', () => { retryCount = 0; return connectVideo(); });
$('frame').addEventListener('loadeddata', () => markVideoReady(peer, receivedStream));
$('frame').addEventListener('timeupdate', () => markVideoReady(peer, receivedStream));
if (typeof globalThis.addEventListener === 'function') globalThis.addEventListener('pagehide', () => {
  stopped = true; searching++;
  clearTimeout(searchTimer); clearTimeout(statusTimer); clearTimeout(statsTimer);
  closePeer();
});
recordings().catch(error => { $('pico-status').textContent = error.message; });
motions(); refresh(); refreshStats();
