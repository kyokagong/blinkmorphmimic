"""CPU-only preview boundary/HTTP tests; no trained-policy claims or GPU imports."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import shutil
import subprocess
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MODULE = Path(__file__).resolve().parents[1] / "preview" / "server.py"
spec = importlib.util.spec_from_file_location("preview_server", MODULE)
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.checkpoint_path = self.root / "runs" / "one" / "q1" / "ckpt_000020.pt"
        self.checkpoint_path.parent.mkdir(parents=True)
        self.checkpoint_path.write_bytes(b"fixed checkpoint boundary fixture")
        self.checkpoint = server.FixedCheckpoint.pin(self.checkpoint_path, "q1")
        for robot in server.ROBOTS:
            folder = self.root / "data" / robot
            folder.mkdir(parents=True)
            (folder / "dance.npz").write_bytes(b"motion fixture")
        self.catalog = server.Catalog(self.root / "data", {"q1": self.checkpoint})
        self.preview = server.Preview(self.catalog)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_checkpoint_refuses_playback(self):
        with self.assertRaisesRegex(ValueError, "playback refused"):
            self.preview.control(dict(action="select", robot="bumi", motion="dance"))
        with self.assertRaises(ValueError):
            self.preview.control(dict(action="start"))

    def test_release_selection_accepts_only_startup_whitelist(self):
        releases={'baseline':dict(catalog=self.catalog,label='Motion 30000'),
                  'human-1000':dict(catalog=self.catalog,label='30000 + Human 1000')}
        preview=server.Preview(self.catalog,releases=releases)
        preview.control(dict(action='select',robot='q1',motion='dance',release='human-1000'))
        self.assertEqual(preview.requests.get_nowait()['release'],'human-1000')
        for value in ('latest','/etc/passwd','../runs'):
            with self.assertRaisesRegex(ValueError,'release'):
                preview.control(dict(action='select',robot='q1',motion='dance',release=value))
        self.assertEqual([r['id'] for r in preview.snapshot()['releases']],['baseline','human-1000'])

    def test_snapshot_marks_joint_training_only_for_bound_manifest(self):
        import copy
        joint=copy.copy(self.catalog);joint.training_manifest=self.root/'training.json'
        preview=server.Preview(self.catalog,releases={'old':dict(catalog=self.catalog,label='Old'),
                                                       'joint':dict(catalog=joint,label='Joint')})
        rows={row['id']:row for row in preview.snapshot()['releases']}
        self.assertFalse(rows['old']['joint_training'])
        self.assertTrue(rows['joint']['joint_training'])

    def test_release_manifest_pins_three_checkpoints_and_rejects_sha_changes(self):
        import torch
        from morphmimic.train.human_adapt import PARENT_SHA256
        manifest=self.root/'data/manifest.json'
        manifest.write_text(json.dumps(dict(complete=True,human_data=True)))
        data_hash=server.sha256(manifest)
        checks={}
        for robot in server.ROBOTS:
            path=self.root/'runs/release'/robot/'ckpt_01000.pt';path.parent.mkdir(parents=True)
            torch.save(dict(iteration=1000,cfg=dict(robot=robot,kp_interface=True,kp_fixed_horizon=1),
                human_adaptation=dict(parent=dict(iteration=30000,sha256=PARENT_SHA256[robot]),human_updates=1000,paired_manifest_sha256=data_hash)),path)
            checks[robot]=dict(path=str(path),sha256=server.sha256(path))
        row=dict(id='human-1000',label='30000 + Human 1000',data_root=str(self.root/'data'),
                 checkpoints=checks,interface='human',parent_iteration=30000,human_updates=1000)
        path=self.root/'releases.json';path.write_text(json.dumps([row]))
        releases=server.load_release_catalogs(path)
        self.assertEqual(releases['human-1000']['catalog'].interface,'human')
        checks['g1']['sha256']='0'*64;path.write_text(json.dumps([row]))
        with self.assertRaisesRegex(ValueError,'SHA'):server.load_release_catalogs(path)

    def test_catalog_search_and_path_boundaries(self):
        self.assertEqual(self.catalog.search("q1", "DANCE")["motions"], ["dance"])
        for robot, clip in [("x2", "dance"), ("q1", "../dance"), ("q1", "/etc/passwd")]:
            with self.assertRaises(ValueError):
                self.catalog.select(robot, clip)
        (self.root / "data" / "q1" / "escape.npz").symlink_to(self.checkpoint_path)
        catalog = server.Catalog(self.root / "data", {"q1": self.checkpoint})
        self.assertNotIn("escape", catalog.motions["q1"])

    def test_motion_catalog_rejects_modified_reference_with_unchanged_manifest(self):
        path=self.catalog.motions['q1']['dance']
        (self.root/'data/manifest.json').write_text(json.dumps(dict(clips=[dict(robot='q1',name='dance',output_sha256=server.sha256(path))])))
        catalog=server.Catalog(self.root/'data',{'q1':self.checkpoint})
        self.assertEqual(catalog.select('q1','dance'),path)
        path.write_bytes(b'changed reference')
        with self.assertRaisesRegex(ValueError,'SHA'):catalog.select('q1','dance')

    def test_fixed_checkpoint_rejects_latest_wrong_robot_symlink_and_changes(self):
        for name in ("latest.pt", "last.pt"):
            path = self.checkpoint_path.with_name(name)
            path.write_bytes(b"mutable")
            with self.assertRaises(ValueError):
                server.FixedCheckpoint.pin(path, "q1")
        with self.assertRaises(ValueError):
            server.FixedCheckpoint.pin(self.checkpoint_path, "g1")
        alias = self.checkpoint_path.with_name("ckpt_000021.pt")
        alias.symlink_to(self.checkpoint_path)
        with self.assertRaises(ValueError):
            server.FixedCheckpoint.pin(alias, "q1")
        self.checkpoint_path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.checkpoint.verify()

    def test_standup_profiles_validate_and_prefer_fixed_6000_with_5000_fallback(self):
        for robot in server.ROBOTS:
            self.catalog.checkpoints[robot] = self.checkpoint
            for motion in ('fallAndGetUp2_subject2', 'fallAndGetUp1_subject5'):
                self.catalog.motions[robot][motion] = self.catalog.motions[robot]['dance']
        self.catalog.common.update(('fallAndGetUp2_subject2', 'fallAndGetUp1_subject5'))
        releases = {name: dict(catalog=self.catalog, label=name)
                    for name in ('joint-motion-5000', 'joint-motion-6000')}
        preview = server.Preview(self.catalog, releases=releases)
        for robot in (*server.ROBOTS, 'all'):
            for pose, motion in (('lying', 'fallAndGetUp2_subject2'),
                                 ('seated', 'fallAndGetUp2_subject2'),
                                 ('half_squat', 'fallAndGetUp1_subject5')):
                with self.subTest(robot=robot, pose=pose):
                    preview.control(dict(action='standup', robot=robot, pose=pose))
                    request = preview.requests.get_nowait()
                    self.assertEqual((request['pose'], request['motion'], request['release']),
                                     (pose, motion, 'joint-motion-6000'))
        for pose in ('../lying', 'standing', '', None, [], {}):
            with self.subTest(invalid_pose=pose), self.assertRaises(ValueError):
                preview.control(dict(action='standup', robot='q1', pose=pose))
        preview.control(dict(action='standup', robot='q1', release='joint-motion-5000'))
        request = preview.requests.get_nowait()
        self.assertEqual((request['pose'], request['release']), ('lying', 'joint-motion-5000'))
        del releases['joint-motion-6000']
        preview.control(dict(action='standup', robot='q1'))
        self.assertEqual(preview.requests.get_nowait()['release'], 'joint-motion-5000')

    def test_standup_uses_fixed_fallen_frame_and_preserves_policy_failure(self):
        for pose, motion, frame in (('lying', 'fallAndGetUp2_subject2', 650),
                                   ('seated', 'fallAndGetUp2_subject2', 700),
                                   ('half_squat', 'fallAndGetUp1_subject5', 8298)):
            with self.subTest(pose=pose):
                self.check_standup_worker(pose, motion, frame)

    def check_standup_worker(self, pose, motion, frame):
        self.catalog.motions['q1'][motion] = self.catalog.motions['q1']['dance']
        preview = server.Preview(self.catalog, releases={
            'joint-motion-5000': dict(catalog=self.catalog, label='Joint 5000')})
        starts = []
        class Session:
            def __init__(self, *args, start_frame=0):
                starts.append(start_frame)
                self.reset()
            def reset(self):
                self.steps, self.failed, self.completed = 0, False, False
            def step(self):
                self.steps += 1
                self.failed = True
            def status(self):
                return dict(steps=self.steps, failed=self.failed, completed=self.completed)
            def frame(self): return b'frame'
            def close(self): pass
        preview.factory = Session
        for extra in ({'start_frame': 0}, {'motion': 'dance'}, {'release': 'missing'}):
            with self.assertRaises(ValueError):
                preview.control(dict(action='standup', robot='q1', **extra))
        preview.control(dict(action='standup', robot='q1', release='joint-motion-5000', pose=pose))
        worker = threading.Thread(target=preview.run)
        worker.start()
        try:
            deadline = time.monotonic() + 2
            while not preview.snapshot()['frame_available'] and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(preview.snapshot()['steps'], 0)
            self.assertFalse(preview.snapshot()['playing'])
            preview.control(dict(action='start'))
            while not preview.snapshot()['failed'] and time.monotonic() < deadline:
                time.sleep(.005)
            status = preview.snapshot()
            self.assertEqual(starts, [frame])
            self.assertEqual(status['motion'], motion)
            self.assertEqual(status['steps'], 1)
            self.assertTrue(status['failed'])
            self.assertFalse(status['playing'])
            self.assertIsNone(status['error'])
            preview.control(dict(action='reset'))
            deadline = time.monotonic() + 2
            while preview.snapshot()['frame_id'] <= status['frame_id'] and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(preview.snapshot()['steps'], 0)
            self.assertFalse(preview.snapshot()['playing'])
            self.assertEqual(starts, [frame])
        finally:
            preview.stop.set()
            worker.join(timeout=2)

    def test_controls_cannot_supply_paths_or_extra_arguments(self):
        for data in [None, [], {}, dict(action="load", path="/tmp/a"),
                     dict(action="select", robot="q1", motion="dance", checkpoint="/tmp/model"),
                     dict(action="pause", robot="q1")]:
            with self.assertRaises(ValueError):
                self.preview.control(data)
        self.preview.control(dict(action="select", robot="q1", motion="dance"))
        self.assertEqual(self.preview.requests.qsize(), 1)

    def test_http_static_status_missing_frame_and_rejected_control(self):
        http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.handler_for(self.preview))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{http.server_port}"
        try:
            with urlopen(base + "/api/status") as response:
                status = json.load(response)
                self.assertEqual(status["control_hz"], 50)
                self.assertFalse(status["qualified"])
                self.assertEqual(status["checkpoints"]["q1"]["iteration"], 20)
            with urlopen(base + "/api/motions?robot=q1&q=dan") as response:
                self.assertEqual(json.load(response)["motions"], ["dance"])
            with urlopen(base + "/") as response:
                self.assertIn(b"Physical motion preview", response.read())
            for path, code in [("/api/frame", 409), ("/../server.py", 404), ("/api/motions?robot=human", 400)]:
                with self.assertRaises(HTTPError) as caught:
                    urlopen(base + path)
                self.assertEqual(caught.exception.code, code)
            request = Request(base + "/api/control", data=b'{"action":"start"}', headers={"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request)
            self.assertEqual(caught.exception.code, 400)
            request = Request(base + "/api/control", data=b'{}', headers={"Origin": "https://other.example"})
            with self.assertRaises(HTTPError):
                urlopen(request)
        finally:
            http.shutdown()
            http.server_close()
            thread.join()

    def test_checkpoint_goal_anchor_reaches_policy_and_environment(self):
        for anchor in ("reference", "robot"):
            torch = MagicMock()
            torch.load.return_value = {"cfg": {"goal_anchor": anchor}}
            load_policy = MagicMock(return_value=(MagicMock(), SimpleNamespace(robot="q1", iteration=20, hist_len=1,interface='motion')))
            cfg = MagicMock()
            environment = MagicMock(step_dt=.02, device="cpu")
            env_cfg = MagicMock(return_value=cfg)
            fake_modules = {
                "torch": torch,
                "yaml": SimpleNamespace(safe_dump=lambda value: json.dumps(value)),
                "mjlab.viewer": SimpleNamespace(OffscreenRenderer=MagicMock()),
                "mjlab.utils.torch": SimpleNamespace(configure_torch_backends=lambda **kwargs: None),
                "morphmimic.constants": SimpleNamespace(PROPRIO_DIM=93),
                "morphmimic.env.env_cfg": SimpleNamespace(COMMAND_NAME="motion", morphmimic_env_cfg=env_cfg),
                "morphmimic.env.morphmimic_env": SimpleNamespace(MorphMimicEnv=MagicMock(return_value=environment)),
                "morphmimic.eval.policy_loader": SimpleNamespace(load_policy=load_policy),
                "morphmimic.train.obs_history": SimpleNamespace(ObsHistory=MagicMock()),
            }
            with patch.dict(sys.modules, fake_modules), patch.object(server.PolicySession, "reset"):
                session = server.PolicySession("q1", self.catalog.motions["q1"]["dance"], self.checkpoint, "cpu")
                try:
                    self.assertEqual(load_policy.call_args.kwargs.get("goal_anchor"), anchor)
                    self.assertEqual(env_cfg.call_args.kwargs.get("goal_anchor"), anchor)
                finally:
                    session.close()

    @unittest.skipUnless(importlib.util.find_spec('mjlab'), 'requires frozen mjlab runtime')
    def test_session_uses_saved_reference_dtype_and_eval_backend_settings(self):
        import torch
        matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
        before = matmul.fp32_precision, cudnn.fp32_precision, cudnn.benchmark, cudnn.deterministic
        try:
            for saved_dtype in ('float16', 'float32', None):
                with self.subTest(saved_dtype=saved_dtype):
                    matmul.fp32_precision = cudnn.fp32_precision = 'ieee'
                    cudnn.benchmark, cudnn.deterministic = False, True
                    saved_cfg = {'goal_anchor': 'robot'}
                    if saved_dtype is not None:
                        saved_cfg['library_dtype'] = saved_dtype
                    torch.save({'cfg': saved_cfg}, self.checkpoint_path)
                    checkpoint = server.FixedCheckpoint.pin(self.checkpoint_path, 'q1')
                    cfg = SimpleNamespace(commands={'motion': SimpleNamespace(library_dtype='float32')}, viewer=MagicMock())
                    received = []
                    environment = MagicMock(step_dt=.02, device='cpu')
                    def make_env(actual_cfg, device):
                        received.append((actual_cfg.commands['motion'].library_dtype,
                            matmul.fp32_precision, cudnn.fp32_precision, cudnn.benchmark, cudnn.deterministic))
                        return environment
                    meta = SimpleNamespace(robot='q1', iteration=20, hist_len=1,interface='motion')
                    modules = {
                        'mjlab.viewer': SimpleNamespace(OffscreenRenderer=MagicMock()),
                        'morphmimic.constants': SimpleNamespace(PROPRIO_DIM=93),
                        'morphmimic.env.env_cfg': SimpleNamespace(COMMAND_NAME='motion', morphmimic_env_cfg=lambda *args, **kwargs: cfg),
                        'morphmimic.env.morphmimic_env': SimpleNamespace(MorphMimicEnv=make_env),
                        'morphmimic.eval.policy_loader': SimpleNamespace(load_policy=lambda *args, **kwargs: (MagicMock(), meta)),
                        'morphmimic.train.obs_history': SimpleNamespace(ObsHistory=MagicMock()),
                    }
                    with patch.dict(sys.modules, modules), patch.object(server.PolicySession, 'reset'):
                        session = server.PolicySession('q1', self.catalog.motions['q1']['dance'], checkpoint, 'cpu')
                        try:
                            self.assertEqual(received, [(saved_dtype or 'float32', 'ieee', 'ieee', True, False)])
                        finally:
                            session.close()
        finally:
            matmul.fp32_precision, cudnn.fp32_precision, cudnn.benchmark, cudnn.deterministic = before

    def test_help_keeps_torch_and_simulator_imports_lazy(self):
        script = """
import runpy, sys
path = sys.argv[1]
sys.argv = [path, '--help']
try:
    runpy.run_path(path, run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules and 'mjlab' not in sys.modules
"""
        result = subprocess.run([sys.executable, '-I', '-c', script, str(MODULE)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_policy_step_keeps_environment_state_mutable_for_reset(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed; run reset regression in the project runtime")
        class Environment:
            device = "cpu"
            physics_diagnostics = SimpleNamespace(report=lambda: {"overflow_bits": 0})
            def step(self, action):
                self.state = torch.zeros(1)
                obs = {key: torch.zeros(1, 1) for key in ("policy", "goal", "goal_kp")}
                return obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}
            def reset(self, env_ids):
                self.state.zero_()
                return {key: torch.zeros(1, 1) for key in ("policy", "goal", "goal_kp")}, {}
        session = server.PolicySession.__new__(server.PolicySession)
        session.start_frame = 0
        session.torch, session.env = torch, Environment()
        session.command, session.history = MagicMock(), MagicMock()
        session.policy = lambda *_: torch.zeros(1, 1)
        session.goals = {}
        session.steps, session.total_reward = 0, 0.0
        session.step()
        session.reset()
        self.assertEqual(session.steps, 0)
        self.assertFalse(torch.is_inference(session.env.state))
        session.start_frame = 650
        session.reset()
        self.assertEqual(session.command.assign_clips.call_args.kwargs['start_frames'].tolist(), [650])
        session.env.physics_diagnostics = SimpleNamespace(report=lambda: {"overflow_bits": 4})
        with self.assertRaisesRegex(RuntimeError, "physics overflow"):
            session.step()

    def test_browser_does_not_request_cleared_frame_during_loading(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node unavailable for browser-script boundary regression")
        script = r"""
const fs = require('fs'), vm = require('vm');
const requestedFrames = [];
const elements = {};
function element(id) {
  if (!elements[id]) {
    elements[id] = {value:'q1',style:{},addEventListener(){},replaceChildren(){}};
    if (id === 'frame') Object.defineProperty(elements[id], 'src', {set(url){requestedFrames.push(url);}});
  }
  return elements[id];
}
const state = {frame_id:2,frame_available:false,loading:true,steps:0,checkpoints:{}};
const context = {document:{getElementById:element},Option:function(){},
  setTimeout(){},clearTimeout(){},
  fetch:async url=>({ok:true,json:async()=>url.startsWith('/api/motions') ? {motions:[],total:0} : state})};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'), context);
setImmediate(()=>{if(requestedFrames.length) throw new Error('Loading state requested missing frame: '+requestedFrames);});
"""
        result = subprocess.run([node, "-e", script, str(MODULE.parent / "static" / "app.js")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_policy_step_failure_is_preserved_and_reset_is_explicit(self):
        class FakeSession:
            def __init__(self, *_):
                self.reset()
            def reset(self):
                self.steps, self.failed, self.completed = 0, False, False
            def step(self):
                self.steps += 1
                self.failed = True
            def status(self):
                return dict(steps=self.steps, failed=self.failed, completed=self.completed)
            def frame(self):
                return b"fake test image"
            def close(self):
                pass
        preview = server.Preview(self.catalog, factory=FakeSession)
        worker = threading.Thread(target=preview.run)
        worker.start()
        def until(predicate):
            deadline = time.monotonic() + 2
            while not predicate():
                if time.monotonic() >= deadline:
                    self.fail("Worker timed out")
                time.sleep(.005)
        try:
            preview.control(dict(action="select", robot="q1", motion="dance"))
            until(lambda: preview.snapshot()["frame_id"] > 0)
            preview.control(dict(action="start"))
            until(lambda: preview.snapshot()["failed"])
            self.assertEqual(preview.snapshot()["steps"], 1)
            self.assertFalse(preview.snapshot()["playing"])
            with self.assertRaisesRegex(ValueError, "Reset"):
                preview.control(dict(action="start"))
            preview.control(dict(action="reset"))
            until(lambda: not preview.snapshot()["failed"])
            self.assertEqual(preview.snapshot()["steps"], 0)
        finally:
            preview.stop.set()
            worker.join(timeout=2)

    def test_inflight_frame_survives_selection_until_new_session_frame_is_ready(self):
        next_motion = self.root / 'data/q1/next.npz'
        next_motion.write_bytes(b'motion fixture')
        catalog = server.Catalog(self.root / 'data', {'q1': self.checkpoint})
        constructing, allow_construct = threading.Event(), threading.Event()
        rendering, allow_render = threading.Event(), threading.Event()
        def factory(robot, path, checkpoint, device):
            if path.stem == 'next':
                constructing.set()
                if not allow_construct.wait(3):
                    raise RuntimeError('test construction barrier timed out')
            def frame():
                if path.stem == 'next':
                    rendering.set()
                    if not allow_render.wait(3):
                        raise RuntimeError('test rendering barrier timed out')
                return path.stem.encode()
            return SimpleNamespace(steps=0, failed=False, completed=False,
                status=lambda: dict(steps=0, failed=False, completed=False),
                frame=frame, close=lambda: None)
        preview = server.Preview(catalog, factory=factory)
        http = server.ThreadingHTTPServer(('127.0.0.1', 0), server.handler_for(preview))
        http_thread = threading.Thread(target=http.serve_forever, daemon=True)
        worker = threading.Thread(target=preview.run)
        http_thread.start()
        worker.start()
        base = f'http://127.0.0.1:{http.server_port}'
        def get_frame(frame_id):
            try:
                with urlopen(f'{base}/api/frame?id={frame_id}', timeout=2) as response:
                    return response.status, response.read()
            except HTTPError as error:
                return error.code, error.read()
        def until(predicate):
            deadline = time.monotonic() + 2
            while not predicate():
                if time.monotonic() >= deadline:
                    self.fail('Worker timed out')
                time.sleep(.005)
        try:
            self.assertEqual(get_frame(0)[0], 409)
            preview.control(dict(action='select', robot='q1', motion='dance'))
            until(lambda: preview.snapshot()['frame_id'] == 1)
            with urlopen(base + '/api/status', timeout=2) as response:
                old_status = json.load(response)
            self.assertTrue(old_status['frame_available'])
            self.assertFalse(old_status['loading'])
            preview.control(dict(action='select', robot='q1', motion='next'))
            self.assertTrue(constructing.wait(2))
            self.assertTrue(preview.snapshot()['loading'])
            self.assertFalse(preview.snapshot()['frame_available'])
            self.assertEqual(get_frame(old_status['frame_id']), (200, b'dance'))
            allow_construct.set()
            self.assertTrue(rendering.wait(2))
            self.assertTrue(preview.snapshot()['loading'])
            self.assertFalse(preview.snapshot()['frame_available'])
            self.assertEqual(get_frame(old_status['frame_id']), (200, b'dance'))
            allow_render.set()
            until(lambda: preview.snapshot()['frame_id'] == 2)
            self.assertFalse(preview.snapshot()['loading'])
            self.assertTrue(preview.snapshot()['frame_available'])
            self.assertEqual(get_frame(2), (200, b'next'))
        finally:
            preview.stop.set()
            allow_construct.set()
            allow_render.set()
            worker.join(timeout=2)
            http.shutdown()
            http.server_close()
            http_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()


class NeutralStandingPreviewTests(unittest.TestCase):
    def test_pinned_robot_specific_recovery_reference_is_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);catalog_root=root/'catalog';catalog_root.mkdir()
            for robot in server.ROBOTS:(catalog_root/robot).mkdir()
            catalog=server.Catalog(catalog_root,{})
            rows=[];poses={'lying':{}}
            for robot in server.ROBOTS:
                path=root/f'{robot}.npz';path.write_bytes(robot.encode())
                name=f'{robot}_neutral_lying'
                rows.append(dict(robot=robot,name=name,output=str(path),output_sha256=server.sha256(path)))
                poses['lying'][robot]=dict(clip=name,start_frame=0,hold_frame=120)
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(clips=rows)))
            windows=root/'windows.json';windows.write_text(json.dumps(dict(training_manifest_sha256=server.sha256(manifest),preview_poses=poses)))
            catalog.training_manifest=manifest;catalog.training_manifest_sha256=server.sha256(manifest)
            catalog.set_recovery_windows(dict(path=str(windows),sha256=server.sha256(windows)))
            selected=catalog.select_recovery('all','lying')
            assert set(selected)==set(server.ROBOTS)
            assert selected['q1']['hold_frame']==120
            assert selected['bumi']['path']==root/'bumi.npz'
            (root/'q1.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA'):catalog.select_recovery('q1','lying')

    def test_policy_continues_after_final_reference_without_replaying_old_clip(self):
        import torch
        session=server.PolicySession.__new__(server.PolicySession)
        session.torch=torch;session.hold_frame=120
        session.command=SimpleNamespace(global_frame=torch.tensor([120]),library=SimpleNamespace(has_human=False))
        def step(action):
            assert session.command.global_frame.item()==120
            session.command.global_frame+=1
            return dict(policy=torch.zeros(1,1),goal=torch.zeros(1,1),goal_kp=torch.zeros(1,1)),torch.zeros(1),torch.tensor([False]),torch.tensor([False]),{}
        session.env=SimpleNamespace(step=step,physics_diagnostics=SimpleNamespace(report=lambda:dict(overflow_bits=0)))
        session.history=SimpleNamespace(push=lambda obs:None)
        session.steps=0;session.total_reward=0.
        for _ in range(300):session.step(torch.zeros(1,2))
        assert session.steps==300
        assert not session.completed and not session.failed


def test_default_standup_uses_first_reviewed_motion_release():
    catalog = SimpleNamespace(
        interface='motion', recovery_windows=True, reference_manifest=True,
        select_recovery=lambda robot, pose: {},
    )
    releases = {
        'joint-motion-20000': dict(catalog=catalog, label='Joint 20000'),
        'joint-motion-7000': dict(catalog=catalog, label='Joint 7000'),
    }
    preview = server.Preview(catalog, releases=releases)
    preview.control(dict(action='standup', robot='q1'))
    assert preview.requests.get_nowait()['release'] == 'joint-motion-20000'


def test_new_standup_releases_require_reviewed_neutral_references():
    for release,interface in (('joint-human-6000','human'),('joint-motion-7000','motion'),('joint-human-7000','human')):
        catalog=SimpleNamespace(interface=interface,recovery_windows=None)
        preview=server.Preview(catalog,releases={release:dict(catalog=catalog,label='Neutral standing')})
        with unittest.TestCase().assertRaisesRegex(ValueError,'reviewed neutral'):
            preview.control(dict(action='standup',robot='q1',release=release,pose='lying'))
        catalog.recovery_windows=True
        catalog.select_recovery=lambda robot,pose:{}
        preview.control(dict(action='standup',robot='q1',release=release,pose='lying'))
        request=preview.requests.get_nowait()
        assert request['release']==release
        assert request['motion']=='neutral_lying'


def test_flip_catalog_requires_common_sources_and_pins_authentic_files(tmp_path):
    root=tmp_path/'original';root.mkdir()
    for robot in server.ROBOTS: (root/robot).mkdir()
    catalog=server.Catalog(root,{r:object() for r in server.ROBOTS},interface='human')
    rows=[];clips=[]
    for robot in server.ROBOTS:
        output=tmp_path/(robot+'.npz');output.write_bytes(robot.encode())
        source=tmp_path/(robot+'.source');source.write_bytes(b'original robot source')
        human=tmp_path/(robot+'.human');human.write_bytes(b'original human source')
        sidecar=tmp_path/(robot+'.sidecar');sidecar.write_bytes(b'authentic resampled human')
        row=dict(robot=robot,name='seed_flip_test',source_id='seed/flip/test',split='train',frames=100,
            output=str(output),output_sha256=server.sha256(output),human_available=True,
            human_sidecar=str(sidecar),human_sha256=server.sha256(sidecar))
        rows.append(row)
        clips.append(dict(robot=robot,clip=row['name'],source_id=row['source_id'],split='train',frames=100,
            source=dict(path=str(source),sha256=server.sha256(source)),human_source=dict(path=str(human),sha256=server.sha256(human))))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(clips=rows)))
    catalog.training_manifest=manifest;catalog.training_manifest_sha256=server.sha256(manifest)
    document=dict(schema_version=1,fps=50,training_manifest_sha256=server.sha256(manifest),clips=clips)
    proof=tmp_path/'flips.json';proof.write_text(json.dumps(document))
    catalog.set_flip_sources(dict(path=str(proof),sha256=server.sha256(proof)))
    assert catalog.search('all','seed_flip_')['motions']==['seed_flip_test']
    assert set(catalog.select('all','seed_flip_test'))==set(server.ROBOTS)
    Path(rows[0]['human_sidecar']).write_bytes(b'modified')
    with __import__('pytest').raises(ValueError,match='authentic human sidecar'):
        catalog.select('q1','seed_flip_test')
    document['clips'][1]['source_id']='different'
    proof.write_text(json.dumps(document))
    other=server.Catalog(root,{r:object() for r in server.ROBOTS})
    other.training_manifest=manifest;other.training_manifest_sha256=server.sha256(manifest)
    with __import__('pytest').raises(ValueError,match='source differs'):
        other.set_flip_sources(dict(path=str(proof),sha256=server.sha256(proof)))
