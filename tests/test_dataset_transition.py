import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch

class DatasetTransitionTests(unittest.TestCase):
 def fixture(self,p):
  def write(name,value):
   path=p/name;path.write_text(json.dumps(value));return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  (p/'old.npz').write_bytes(b'old');(p/'new.npz').write_bytes(b'new')
  old=dict(robot='q1',name='old',output=str(p/'old.npz'),output_sha256='a'*64,frames=10)
  new=dict(robot='q1',name='new',output=str(p/'new.npz'),output_sha256='b'*64,frames=20)
  osub=write('old-sub.json',dict(root='/',clips=[old['output']]));nsub=write('new-sub.json',dict(root='/',clips=[old['output'],new['output']]))
  om=write('old-manifest.json',dict(complete=True,clips=[old]));nm=write('new-manifest.json',dict(complete=True,clips=[old,new]))
  floor=dict(robot='q1',name='old',offset_m=.01,valid=True)
  of=write('old-floor.json',dict(prepared_manifest_sha256=om['sha256'],clips=[floor]));nf=write('new-floor.json',dict(prepared_manifest_sha256=nm['sha256'],clips=[floor,dict(floor,name='new')]))
  target=write('target.json',dict(approved=True))
  doc=dict(schema_version=1,parent_manifest=om,manifest=nm,parent_contact_floor=of,contact_floor=nf,approved_target=dict(review_manifest=target,frame=11446,fps=50),robots={'q1':dict(parent_subset=osub,subset=nsub,original_clips=['old'],appended_clips=['new'],original_clip_frames=[10],appended_clip_frames=[20],original_frames=10,appended_frames=20,original_clip_count=1,appended_clip_count=1)})
  receipt=write('transition.json',doc);return doc,receipt
 def test_append_only_verified_and_tampering_rejected(self):
  from morphmimic.train.dataset_transition import verify_append_transition
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);doc,r=self.fixture(p)
   result=verify_append_transition(r['path'],r['sha256'],doc['manifest']['path'],'q1')
   self.assertEqual(result['robots']['q1']['original_frames'],10)
   mp=Path(doc['manifest']['path']);original=mp.read_text();mp.write_text(original+' ')
   with self.assertRaises(ValueError):verify_append_transition(r['path'],r['sha256'],str(mp),'q1')
   mp.write_text(original)
   rp=Path(r['path']);rp.write_text(rp.read_text()+' ')
   with self.assertRaises(ValueError):verify_append_transition(str(rp),r['sha256'],str(mp),'q1')
 def test_metadata_extension_preserves_original_counts_and_rejects_reordered_prefix(self):
  from morphmimic.train.dataset_transition import extend_inventory_state
  lib=SimpleNamespace(clip_names=['old','new'],clip_len=torch.tensor([10,20]),num_frames=30,
   source_start=torch.tensor([0,0]),source_stop=torch.tensor([10,20]),source_has_phases=torch.tensor([False,False]),
   usage={'policy':torch.zeros(2,dtype=torch.long)},source_usage={'policy':torch.zeros(2,dtype=torch.long)},
   bin_start=torch.tensor([0,5,10,15,20,25]))
  cmd=SimpleNamespace(G_b=torch.zeros(6),past_softmax=torch.ones(6)/6)
  ck=dict(coverage=dict(clip_names=['old'],counts={'policy':torch.tensor([17])},source=dict(since_iteration=0,start=[0],stop=[10],has_phases=[False],counts={'policy':[13]})),sampler=dict(G_b=torch.tensor([.2,.8]),past_softmax=torch.tensor([.3,.7])))
  info=dict(original_clips=['old'],appended_clips=['new'],original_clip_frames=[10],appended_clip_frames=[20],original_frames=10)
  ck['iteration']=6000
  ck['human_coverage']=dict(clip_names=['old'],since_iteration=5000,counts=torch.tensor([19]),source_counts=torch.tensor([11]))
  result=extend_inventory_state(ck,lib,cmd,info)
  self.assertEqual(result['coverage']['counts']['policy'].tolist(),[17,0]);self.assertEqual(result['coverage']['source']['counts']['policy'],[13,0])
  self.assertTrue(torch.allclose(result['sampler']['G_b'],torch.tensor([.2,.8,0,0,0,0])))
  self.assertAlmostEqual(float(result['sampler']['past_softmax'].sum()),1.,places=6)
  self.assertEqual(result['human_coverage']['clip_names'],['old','new'])
  self.assertEqual(result['human_coverage']['counts'].tolist(),[19,0])
  self.assertEqual(result['human_coverage']['source_counts'].tolist(),[11,0])
  self.assertEqual(result['human_coverage']['since_iteration'],5000)
  self.assertEqual(ck['human_coverage']['counts'].tolist(),[19])
  self.assertEqual(ck['coverage']['clip_names'],['old'])
  lib.clip_names=['new','old']
  with self.assertRaises(ValueError):extend_inventory_state(ck,lib,cmd,info)
 def test_human_coverage_starts_at_parent_iteration_and_restores_exactly(self):
  from morphmimic.train.dataset_transition import human_coverage_state
  lib=SimpleNamespace(clip_names=['old','new'],device=torch.device('cpu'))
  fresh=human_coverage_state(None,lib,5000)
  self.assertEqual(fresh['since_iteration'],5000);self.assertEqual(fresh['counts'].tolist(),[0,0])
  fresh['counts'][1]=4;fresh['source_counts'][1]=3
  restored=human_coverage_state(fresh,lib,5250)
  self.assertEqual(restored['since_iteration'],5000);self.assertEqual(restored['counts'].tolist(),[0,4])
  with self.assertRaises(ValueError):human_coverage_state(dict(fresh,since_iteration=-1),lib,5250)
