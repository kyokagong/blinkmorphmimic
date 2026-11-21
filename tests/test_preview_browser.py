"""Run the browser script at its HTTP and WebRTC boundaries without extra packages."""
import pathlib
import shutil
import subprocess
import unittest


class PreviewBrowserTests(unittest.TestCase):
    def test_standup_shortcut_pins_checkpoint_and_selected_robot(self):
        node=shutil.which('node')
        if not node:self.skipTest('Node unavailable')
        script=r"""
const assert=require('assert/strict'),fs=require('fs'),vm=require('vm'),elements={},requests=[];
const release=process.argv[4],pose=process.argv[3];
const server={session_generation:3,releases:[{id:'joint-motion-5000',interface:'motion',joint_training:true},
 ...(release!=='joint-motion-5000'?[{id:release,interface:release.includes('human')?'human':'motion',joint_training:true,neutral_standing:release.endsWith('20000')||release.endsWith('7000')||release.includes('human')}]:[])],checkpoints:{q1:{},bumi:{},g1:{}}};
function element(id){return elements[id]??={value:id==='mode'?process.argv[2]:id==='robot'?process.argv[5]:id==='standup-pose'?pose:id==='release'?release:'',style:{},listeners:{},
 addEventListener(n,fn){this.listeners[n]=fn;},replaceChildren(...children){this.value=children[0]?.value||'';}};}
const context={document:{getElementById:element},Option:function(label,value){this.value=value;},setTimeout(){},clearTimeout(){},
 fetch:async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>url==='/api/recordings'?{enabled:true,recordings:[] }:url==='/api/status'?server:{total:1,motions:['fallAndGetUp2_subject2']}};}};
async function flush(){for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));}
(async()=>{
 vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);await flush();
 assert.equal(element('standup').disabled,false);
 assert(element('standup').textContent.includes(release.split('-').at(-1)));
 await element('standup').listeners.click();await flush();
 const call=requests.find(r=>r.url==='/api/control');
 assert.deepEqual(JSON.parse(call.options.body),{action:'standup',robot:process.argv[2]==='compare'?'all':process.argv[5],release,pose});
 assert.equal(element('search').value,pose==='half_squat'?'fallAndGetUp1_subject5':'fallAndGetUp2_subject2');
 assert(!requests.some(r=>r.options?.body && JSON.parse(r.options.body).action==='start'),'loading a pose must stay paused');
 assert.equal(element('standup').disabled,true,'duplicate click while loading must be disabled');
 server.session_generation=4;server.robot='bumi';server.frame_available=true;context.renderStatus(server);
 assert.equal(element('standup').disabled,false);
 server.releases=[];context.renderStatus(server);assert.equal(element('standup').disabled,true);
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
        app=pathlib.Path(__file__).resolve().parents[1]/'preview/static/app.js'
        for mode in ('single','compare'):
            for pose in ('lying', 'seated', 'half_squat'):
                for release in ('joint-motion-5000', 'joint-motion-6000', 'joint-human-6000', 'joint-motion-7000', 'joint-human-7000', 'joint-motion-20000', 'joint-human-20000'):
                    for robot in ('q1', 'bumi', 'g1'):
                        with self.subTest(mode=mode, pose=pose, release=release, robot=robot):
                            result=subprocess.run([node,'-e',script,str(app),mode,pose,release,robot],capture_output=True,text=True)
                            self.assertEqual(result.returncode,0,result.stderr)

    def test_h264_receive_level_is_extended_only_after_browser_support_probe(self):
        node=shutil.which('node')
        if not node:self.skipTest('Node unavailable')
        script=r"""
const assert=require('assert/strict'),fs=require('fs'),vm=require('vm');
const context={document:{getElementById:()=>({value:'',style:{},addEventListener(){},replaceChildren(){}})},
  fetch:()=>new Promise(()=>{}),setTimeout(){},clearTimeout(){},MediaSource:{isTypeSupported:()=>true}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);
const offer={type:'offer',sdp:'v=0\r\na=rtpmap:96 H264/90000\r\na=fmtp:96 profile-level-id=42e01f;packetization-mode=1\r\n'};
const enhanced=context.receiveOffer(offer);
assert(enhanced.sdp.includes('profile-level-id=42e01f'));
assert(enhanced.sdp.includes('max-recv-level=e028'));
assert(!offer.sdp.includes('max-recv-level'),'native offer must remain unchanged');
context.MediaSource.isTypeSupported=()=>false;
assert.equal(context.receiveOffer(offer).sdp,offer.sdp);
"""
        result=subprocess.run([node,'-e',script,str(pathlib.Path(__file__).resolve().parents[1]/'preview/static/app.js')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_shared_session_generation_releases_pending_and_rejects_old_video(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node unavailable")
        script = r"""
const assert=require('assert/strict'),fs=require('fs'),vm=require('vm');
const elements={},timers=new Map(),callbacks=new Map(),peers=[];let timer=0,callback=0,stops=0;
let state={mode:'single',robot:'q1',motion:'dance',session_generation:1,frame_id:1,
  frame_available:true,loading:false,playing:false,steps:1,checkpoints:{q1:{},bumi:{},g1:{}}};
function element(id){
  if(!elements[id])elements[id]={value:id==='mode'?'single':id==='robot'?'q1':'',style:{},listeners:{},readyState:0,
    addEventListener(name,fn){this.listeners[name]=fn;},replaceChildren(...children){this.value=children[0]?.value||'';},
    requestVideoFrameCallback(fn){callbacks.set(++callback,fn);return callback;},cancelVideoFrameCallback(id){callbacks.delete(id);},play:async()=>{}};
  return elements[id];
}
class Peer{
  constructor(){this.listeners={};this.iceGatheringState='complete';this.connectionState='new';peers.push(this);}
  addTransceiver(){}addEventListener(name,fn){this.listeners[name]=fn;}removeEventListener(){}
  async createOffer(){return {type:'offer',sdp:'offer'};}async setLocalDescription(value){this.localDescription=value;}
  async setRemoteDescription(){}close(){this.closed=true;}
}
const context={document:{getElementById:element},Option:function(name,value){this.value=value;},RTCPeerConnection:Peer,
  setTimeout(fn,delay){timers.set(++timer,{fn,delay});return timer;},clearTimeout(id){timers.delete(id);},
  fetch:async(url)=>({ok:true,json:async()=>url==='/api/recordings'?{enabled:true,recordings:[] }:url.startsWith('/api/motions')?{motions:['dance'],total:1}:url==='/api/status'?state:{type:'answer',sdp:'answer'}})};
async function flush(){for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));}
async function poll(){const entry=[...timers].find(([,value])=>value.delay===500);assert(entry);timers.delete(entry[0]);await entry[1].fn();await flush();}
(async()=>{
  vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);await flush();
  const old=peers.at(-1),video=element('frame');
  old.listeners.track({streams:[{getTracks:()=>[{stop(){stops++;}}]}]});video.readyState=2;
  const oldDecoded=[...callbacks.values()][0];oldDecoded();assert.equal(video.style.display,'block');
  if(process.argv[2]==='superseded'){
    element('mode').value='compare';await element('mode').listeners.change();await element('select').listeners.click();
    state={...state,mode:'single',robot:'g1',motion:'another',session_generation:3,frame_id:8};await poll();
    assert(!element('runtime').textContent.includes('载入'),'another viewer completing a newer generation must release local pending state');
    assert(element('active').textContent.includes('G1'));assert.equal(element('reset').disabled,false);
  }else{
    state={...state,robot:'bumi',session_generation:2,frame_id:7};await poll();
    video.listeners.timeupdate();oldDecoded();
    assert.equal(video.style.display,'none','previous stream events cannot reveal old robot under new session label');
    assert.equal(old.closed,true);assert.equal(stops,1);assert.equal(video.srcObject,null);
    const fresh=peers.at(-1);assert.notEqual(fresh,old);
    fresh.listeners.track({streams:[{getTracks:()=>[]}]});video.readyState=0;video.listeners.timeupdate();
    assert.equal(video.style.display,'none','fresh stream must decode before becoming visible');
    video.readyState=2;video.listeners.timeupdate();oldDecoded();
    assert.equal(video.style.display,'none','queued old events still cannot prove a newly attached stream decoded');
    [...callbacks.values()].at(-1)();assert.equal(video.style.display,'block');
    const count=peers.length;await poll();assert.equal(peers.length,count,'steady session polls must not create more peers');
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        app = pathlib.Path(__file__).resolve().parents[1] / "preview/static/app.js"
        for scenario in ("superseded", "stale-video"):
            with self.subTest(scenario=scenario):
                result = subprocess.run([node, "-e", script, str(app), scenario], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_video_comparison_controls_and_connection_cleanup(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node unavailable")
        script = r"""
const assert = require('assert/strict'), fs = require('fs'), vm = require('vm');
const elements = {}, requests = [], timers = new Map(), events = {}, peers = [];
let timerId = 0, stopped = 0, state = {
  mode:'single',robot:'q1',motion:'dance',session_generation:1,loading:false,playing:false,
  frame_available:true,frame_id:1,steps:10,simulated_seconds:.2,clip_frames:50,
  checkpoints:{q1:{path:'q1.pt',iteration:30000},bumi:{path:'b.pt'},g1:{path:'g.pt'}},
  participants:[],simulation_hz:48,render_fps:29,sim_speed:.96
};
function element(id) {
  if (!elements[id]) elements[id] = {
    value:id==='mode'?'single':id==='robot'?'q1':id==='motions'?'dance':'',
    style:{},listeners:{},hidden:false,srcObject:null,
    addEventListener(name, callback){ this.listeners[name] = callback; },
    replaceChildren(...children){this.children=children;this.value=children[0]?.value||'';},
    play:async()=>{},
  };
  return elements[id];
}
class Peer {
  constructor(){this.listeners={};this.iceGatheringState='complete';this.connectionState='new';peers.push(this);}
  addTransceiver(kind, options){assert.equal(kind,'video');assert.equal(options.direction,'recvonly');}
  addEventListener(name, callback){this.listeners[name]=callback;}
  removeEventListener(name){delete this.listeners[name];}
  async createOffer(){return {type:'offer',sdp:'offer-sdp'};}
  async setLocalDescription(offer){this.localDescription=offer;}
  async setRemoteDescription(answer){assert.equal(answer.type,'answer');this.remoteDescription=answer;}
  async getStats(){this.samples=(this.samples||0)+1;const second=this.samples>1;return new Map([
    ['video',{type:'inbound-rtp',kind:'video',framesDecoded:second?50:20,framesReceived:second?52:22,timestamp:second?3000:2000,framesPerSecond:24}],
    ['transport',{type:'transport',selectedCandidatePairId:'pair'}],
    ['pair',{id:'pair',type:'candidate-pair',state:'succeeded',currentRoundTripTime:.032}],
  ]);}
  close(){this.closed=true;this.connectionState='closed';}
}
const context = {
  document:{getElementById:element},Option:function(name,value){this.text=name;this.value=value;},
  RTCPeerConnection:Peer,MediaStream:function(tracks){this.getTracks=()=>tracks;},
  setTimeout(callback,delay){const id=++timerId;timers.set(id,{callback,delay});return id;},
  clearTimeout(id){timers.delete(id);},
  addEventListener(name,callback){events[name]=callback;},
  fetch:async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>url==='/api/recordings'?{enabled:true,recordings:[] }:
    url.startsWith('/api/motions')?{motions:['dance'],total:1}:url==='/api/offer'?{type:'answer',sdp:'answer-sdp'}:
    url==='/api/control'?{ok:true}:state};},
};
async function flush(){for(let i=0;i<10;i++)await new Promise(resolve=>setImmediate(resolve));}
async function runTimer(delay){const entry=[...timers].find(([,timer])=>timer.delay===delay);assert(entry,`missing ${delay}ms timer`);timers.delete(entry[0]);await entry[1].callback();await flush();}
(async()=>{
  vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);
  await flush();
  assert.equal(peers.length,1,'video peer should connect on page load');
  assert.deepEqual(JSON.parse(requests.find(r=>r.url==='/api/offer').options.body),{type:'offer',sdp:'offer-sdp'});
  const first=peers[0];first.connectionState='connected';first.listeners.connectionstatechange();
  first.listeners.track({streams:[{getTracks:()=>[{stop(){stopped++;}}]}]});
  await flush();
  const originalFetch=context.fetch;let deliverCatalog;
  context.fetch=(url,options)=>url.startsWith('/api/motions?robot=all')?
    new Promise(resolve=>{deliverCatalog=()=>resolve(originalFetch(url,options));}):originalFetch(url,options);
  element('mode').value='compare';const changedMode=element('mode').listeners.change();await flush();
  assert.equal(element('select').disabled,true,'old catalog must not be loadable while changing mode');
  context.fetch=originalFetch;deliverCatalog();await changedMode;await flush();
  assert.equal(element('robot').disabled,true);
  assert(requests.some(r=>r.url==='/api/motions?robot=all&q='));
  assert(!requests.some(r=>r.url==='/api/control'),'dropdown must not switch active session');
  assert(element('active').textContent.includes('Q1'),'active label must still identify running robot');
  element('search').value='a & b';element('search').listeners.input();await runTimer(180);
  assert(requests.some(r=>r.url==='/api/motions?robot=all&q=a%20%26%20b'));
  await element('select').listeners.click();await flush();
  assert.deepEqual(JSON.parse(requests.find(r=>r.url==='/api/control').options.body),{action:'select',robot:'all',motion:'dance'});
  assert.equal(element('frame').style.display,'none','loading must hide previous video');
  state={...state,mode:'compare',robot:'all',session_generation:2,frame_id:2,source:{q1:'q1/dance.npz'},source_duration_seconds:2,
    participants:[{robot:'q1',steps:10,failed:true,termination_terms:['fallen'],source_window:{start_frame:150,stop_frame:250,source_frames:500,offset_seconds:3},source_metadata:{source_id:'shared/dance'}},{robot:'bumi',steps:10},{robot:'g1',steps:10}],failed:true};
  await runTimer(500);
  assert(element('active').textContent.includes('同屏'));
  assert(element('participants').textContent.includes('Q1'));
  assert(element('failure').textContent.includes('fallen'));
  assert(element('source').textContent.includes('[150, 250)'),'comparison must disclose actual crop window');
  assert(element('source').textContent.includes('shared/dance'));
  const current=peers.at(-1);current.connectionState='connected';current.listeners.connectionstatechange();
  current.listeners.track({streams:[{getTracks:()=>[{stop(){stopped++;}}]}]});await runTimer(1000);
  state={...state,failed:false,playing:true,participants:[{robot:'q1',steps:11},{robot:'bumi',steps:11},{robot:'g1',steps:11}]};
  await runTimer(500);
  assert(element('participants').textContent.includes('Q1 · 运行中'),'participant status must reflect shared playback');
  await runTimer(1000);
  assert.equal(element('stream-stats').textContent,'接收 30.0 fps · 解码 30.0 fps · RTT 32 ms');
  assert(!element('stream-stats').textContent.includes('延迟'));
  let releaseStats;
  current.getStats=()=>new Promise(resolve=>{releaseStats=resolve;});
  const inFlightStats=runTimer(1000);await flush();
  await element('reconnect').listeners.click();await flush();
  releaseStats(new Map());await inFlightStats;
  assert([...timers.values()].some(timer=>timer.delay===1000),'stats must continue after reconnect during getStats');
  assert.equal(current.closed,true,'reconnect must release previous peer');
  assert(!element('stream-stats').textContent.includes('RTT 32 ms'),'new connection must not show old statistics');
  assert.equal(stopped,2,'generation change and reconnect must stop received tracks');
  assert.equal(peers.length,3);
  events.pagehide();await flush();
  assert.equal(peers.at(-1).closed,true,'leaving page must release peer');
  assert.equal(timers.size,0,'leaving page must cancel polling and retries');
  assert(!requests.some(r=>r.url.startsWith('/api/frame')),'video must never poll JPEG endpoint');
  const beforeFailures=peers.length;
  vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),{...context,
    RTCPeerConnection:class extends Peer{async createOffer(){throw new Error('unreachable');}}});
  await flush();await runTimer(1500);await runTimer(3000);await runTimer(4500);
  assert.equal(peers.length-beforeFailures,4,'initial attempt plus only three automatic retries');
  assert(element('connection').textContent.includes('自动重试已停止'));
  events.pagehide();assert.equal(timers.size,0);
  vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),{...context,RTCPeerConnection:undefined});
  await flush();
  assert(element('connection').textContent.includes('不支持 WebRTC'));
  events.pagehide();assert.equal(timers.size,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        app = pathlib.Path(__file__).resolve().parents[1] / "preview/static/app.js"
        result = subprocess.run([node, "-e", script, str(app)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


    def test_joint_release_note_and_shortcuts_prefer_joint_then_fall_back(self):
        node=shutil.which('node')
        if not node:self.skipTest('Node unavailable')
        script=r"""
const assert=require('assert/strict'),fs=require('fs'),vm=require('vm'),elements={},requests=[];
const human={id:'joint-human',label:'Joint human',interface:'human',joint_training:true};
const motion={id:'joint-motion',label:'Joint motion',interface:'motion',joint_training:true,flips:true};
const legacy={id:'motiondecode-30000',label:'Legacy motion',interface:'motion',joint_training:false};
const server={release_id:human.id,releases:[human,legacy,motion],checkpoints:{q1:{}},loading:false,frame_available:false};
function element(id){return elements[id]??={value:id==='mode'?'single':id==='robot'?'q1':id==='release'?'joint-human':'',style:{},listeners:{},
 addEventListener(n,fn){this.listeners[n]=fn;},replaceChildren(...children){this.value=children[0]?.value||'';}};}
const context={document:{getElementById:element},Option:function(label,value){this.value=value;},setTimeout(){},clearTimeout(){},
 fetch:async(url)=>{requests.push(url);return {ok:true,json:async()=>url==='/api/recordings'?{enabled:true,recordings:[] }:url==='/api/status'?server:{total:1,motions:['motiondecode_baji_example']}};}};
async function flush(){for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));}
(async()=>{
 vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);await flush();
 assert(element('release-note').textContent.includes('联合训练'));
 assert(!element('release-note').textContent.includes('独立人体更新'));
 await element('motiondecode-baji').listeners.click();await flush();
 assert.equal(element('release').value,'joint-motion');
 assert.equal(new URL(requests.findLast(u=>u.startsWith('/api/motions')),'http://local').searchParams.get('release'),'joint-motion');
 server.releases=[human,legacy];context.renderStatus(server);
 await element('motiondecode-dance').listeners.click();await flush();
 assert.equal(element('release').value,'motiondecode-30000');
 server.releases=[human,motion];context.renderStatus(server);
 assert.equal(element('motiondecode-actions').hidden,false);
 await element('motiondecode-baji').listeners.click();await flush();
 assert.equal(element('release').value,'joint-motion');
 await element('flips').listeners.click();await flush();
 assert.equal(element('search').value,'seed_flip_');
 assert.equal(element('release').value,'joint-motion');
 assert.equal(element('flip-actions').hidden,false);
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
        result=subprocess.run([node,'-e',script,str(pathlib.Path(__file__).resolve().parents[1]/'preview/static/app.js')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_motiondecode_shortcuts_expose_baji_and_dance_from_a_human_release(self):
        node=shutil.which('node')
        if not node:self.skipTest('Node unavailable')
        script=r"""
const assert=require('assert/strict'),fs=require('fs'),vm=require('vm'),elements={},requests=[];
const server={mode:'compare',robot:'all',motion:'dance1_subject2',session_generation:4,frame_id:8,frame_available:true,
  loading:false,playing:false,steps:12,checkpoints:{q1:{},bumi:{},g1:{}},release_id:'human-1000',input_mode:'human-five-h1',
  releases:[{id:'human-1000',label:'Human +1000',interface:'human'},{id:'motiondecode-30000',label:'MotionDecode',interface:'motion'}]};
function element(id){return elements[id]??=( {value:id==='mode'?'compare':id==='robot'?'q1':id==='release'?'human-1000':'',style:{},listeners:{},
 addEventListener(n,fn){this.listeners[n]=fn;},replaceChildren(...children){this.value=children[0]?.value||'';}} );}
const context={document:{getElementById:element},Option:function(label,value){this.value=value;},setTimeout(){},clearTimeout(){},
 fetch:async(url,options)=>{requests.push({url,options});let data={};if(url==='/api/status')data=server;
 else if(url.startsWith('/api/motions')){const q=new URL(url,'http://local').searchParams.get('q');data={total:1,motions:[q==='motiondecode_baji'?'motiondecode_baji_example':q==='motiondecode_dance'?'motiondecode_dance_example':'dance1_subject2']};}
 return {ok:true,json:async()=>url==='/api/recordings'?{enabled:true,recordings:[] }:data};}};
async function flush(){for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));}
(async()=>{
 vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);await flush();
 assert.equal(element('motiondecode-actions').hidden,false);
 assert.equal(typeof element('motiondecode-baji').listeners.click,'function','baji entry must be visible and wired');
 await element('motiondecode-baji').listeners.click();await flush();
 assert.equal(element('release').value,'motiondecode-30000');assert.equal(element('search').value,'motiondecode_baji');
 assert.equal(element('motions').value,'motiondecode_baji_example');assert(!requests.some(r=>r.url==='/api/control'));
 assert(element('active').textContent.includes('Human +1000'),'preselection must keep the actual running model label');
 context.renderStatus(server);assert(element('release-note').textContent.includes('MotionDecode'));assert.equal(element('human-points').disabled,true);
 await element('select').listeners.click();await flush();
 const body=JSON.parse(requests.findLast(r=>r.url==='/api/control').options.body);
 assert.equal(body.release,'motiondecode-30000');assert.equal(body.robot,'all');assert.equal(body.motion,'motiondecode_baji_example');assert(!('human_points' in body));
 assert.equal(element('motiondecode-dance').disabled,true,'switching sources must wait for the pending load');
 Object.assign(server,{session_generation:5,release_id:'motiondecode-30000',motion:'motiondecode_baji_example',input_mode:'motion'});context.renderStatus(server);
 await element('motiondecode-dance').listeners.click();await flush();
 assert.equal(element('motions').value,'motiondecode_dance_example');
 const query=new URL(requests.findLast(r=>r.url.startsWith('/api/motions')).url,'http://local').searchParams;
 assert.equal(query.get('release'),'motiondecode-30000');assert.equal(query.get('q'),'motiondecode_dance');
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
        result=subprocess.run([node,'-e',script,str(pathlib.Path(__file__).resolve().parents[1]/'preview/static/app.js')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == "__main__":
    unittest.main()
