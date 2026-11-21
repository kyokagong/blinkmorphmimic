"""Shared simulation clock and common-source catalog boundaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

module=Path(__file__).resolve().parents[1]/'preview/server.py'
spec=importlib.util.spec_from_file_location('preview_worker_server',module)
server=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=server
spec.loader.exec_module(server)

class PreviewWorkerTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name); self.checkpoints={}
  rows=[]
  for robot in server.ROBOTS:
   folder=self.root/robot;folder.mkdir();(folder/'dance.npz').write_bytes(b'fixture')
   self.checkpoints[robot]=SimpleNamespace(public=lambda:dict(iteration=30000))
   rows.append(dict(robot=robot,name='dance',source_id='seed/same-dance'))
  (self.root/'manifest.json').write_text(json.dumps(dict(clips=rows)))
 def tearDown(self):self.tmp.cleanup()
 def test_comparison_catalog_requires_same_source_and_all_checkpoints(self):
  catalog=server.Catalog(self.root,self.checkpoints)
  self.assertEqual(catalog.search('all','DANCE')['motions'],['dance'])
  self.assertEqual(set(catalog.select('all','dance')),set(server.ROBOTS))
  rows=json.loads((self.root/'manifest.json').read_text());rows['clips'][-1]['source_id']='seed/different'
  (self.root/'manifest.json').write_text(json.dumps(rows))
  self.assertEqual(server.Catalog(self.root,self.checkpoints).search('all','')['motions'],[])
  with self.assertRaises(ValueError):server.Catalog(self.root,{'q1':self.checkpoints['q1']}).select('all','dance')
  with self.assertRaises(ValueError):catalog.select('all','../dance')
 def test_latest_rgb_readers_never_advance_physics_and_compare_reset_is_grouped(self):
  import numpy as np
  created=[]
  class Session:
   def __init__(self,*args,**kwargs):created.append((args,kwargs));self.reset()
   def reset(self):self.steps=0;self.failed=self.completed=False
   def step(self):self.steps+=1;self.failed=self.steps>=4
   def status(self):return dict(steps=self.steps,failed=self.failed,completed=False,clip_frames=100,participants=[dict(robot=r,steps=self.steps) for r in server.ROBOTS])
   def frame_rgb(self):return np.full((16,24,3),self.steps,dtype=np.uint8)
   def close(self):pass
  preview=server.Preview(server.Catalog(self.root,self.checkpoints),'cpu',comparison_factory=Session,compare_devices=('cpu','cpu','cpu'))
  worker=threading.Thread(target=preview.run);worker.start()
  def until(fn):
   end=time.monotonic()+3
   while not fn():
    if time.monotonic()>end:self.fail('worker timeout '+str(preview.snapshot()))
    time.sleep(.005)
  try:
   preview.control(dict(action='select',robot='all',motion='dance'));until(lambda:preview.snapshot()['frame_available'])
   first=preview.latest_frame();self.assertEqual(first[1].shape,(16,24,3))
   for _ in range(100):self.assertEqual(preview.latest_frame()[0],first[0])
   self.assertEqual(preview.snapshot()['steps'],0)
   self.assertEqual(preview.snapshot()['mode'],'compare')
   self.assertEqual(preview.snapshot()['session_generation'],1)
   preview.control(dict(action='start'));until(lambda:preview.snapshot()['failed'])
   self.assertFalse(preview.snapshot()['playing']);self.assertEqual(preview.snapshot()['steps'],4)
   preview.control(dict(action='reset'));until(lambda:not preview.snapshot()['failed'])
   self.assertEqual(preview.snapshot()['steps'],0)
   self.assertEqual(preview.snapshot()['session_generation'],1)
   self.assertEqual(len(created),1)
  finally:preview.stop.set();worker.join(3)

if __name__=='__main__':unittest.main()

class PreviewAsyncTests(PreviewWorkerTests):
 def test_comparison_render_does_not_stall_control_and_reset_rejects_stale_frame(self):
  import numpy as np
  from preview.comparison import AsyncComparisonRenderer
  blocked, release = threading.Event(), threading.Event()
  class Session:
   def __init__(self,*args,**kwargs):
    self.reset(); self.renderer=AsyncComparisonRenderer(lambda: Renderer())
   def reset(self):self.steps=0;self.failed=self.completed=False
   def step(self):self.steps+=1;self.failed=self.steps>=5
   def status(self):return dict(steps=self.steps,failed=self.failed,completed=False,clip_frames=100,participants=[])
   def submit_frame(self,reset=False):self.renderer.submit({r:np.array([self.steps]) for r in server.ROBOTS},reset)
   def latest_render(self):return self.renderer.latest()
   def close(self):self.renderer.close()
  class Renderer:
   def frame_rgb(self,poses,reset):
    if poses['q1'][0] == 1:
     blocked.set();release.wait(2)
    return np.full((2,2,3),poses['q1'][0],dtype=np.uint8)
   def close(self):pass
  preview=server.Preview(server.Catalog(self.root,self.checkpoints),'cpu',comparison_factory=Session)
  thread=threading.Thread(target=preview.run);thread.start()
  def until(fn):
   end=time.monotonic()+3
   while not fn():
    if time.monotonic()>end:self.fail('worker timeout '+str(preview.snapshot()))
    time.sleep(.003)
  try:
   preview.control(dict(action='select',robot='all',motion='dance'));until(lambda:preview.snapshot()['frame_available'])
   preview.control(dict(action='start'));self.assertTrue(blocked.wait(2));until(lambda:preview.snapshot()['failed'])
   self.assertEqual(preview.snapshot()['steps'],5)
   preview.control(dict(action='reset'));until(lambda:preview.snapshot()['steps']==0)
   self.assertFalse(preview.snapshot()['frame_available'])
   release.set();until(lambda:preview.snapshot()['frame_available'])
   self.assertEqual(int(preview.latest_frame()[1][0,0,0]),0)
  finally:
   release.set();preview.stop.set();thread.join(3)

 def test_control_graph_option_reaches_comparison_factories(self):
  import numpy as np
  created=[]
  class Session:
   def __init__(self,*args,**kwargs):created.append(kwargs);self.steps=0;self.failed=self.completed=False
   def status(self):return dict(steps=0,failed=False,completed=False,clip_frames=10,participants=[])
   def frame_rgb(self):return np.zeros((2,2,3),dtype=np.uint8)
   def close(self):pass
  from unittest.mock import patch
  preview=server.Preview(server.Catalog(self.root,self.checkpoints),'cpu',comparison_factory=Session,control_graph=True)
  with patch.dict(sys.modules, {'preview.shared_policy': SimpleNamespace(SharedPolicy=object)}):
   thread=threading.Thread(target=preview.run);thread.start()
   try:
    preview.control(dict(action='select',robot='all',motion='dance'))
    end=time.monotonic()+2
    while not created and time.monotonic()<end:time.sleep(.005)
    self.assertTrue(created)
    self.assertTrue(created[0]['session_factory'].keywords['control_graph'])
    self.assertTrue(created[0]['policy_factory'].keywords['control_graph'])
   finally:preview.stop.set();thread.join(3)
