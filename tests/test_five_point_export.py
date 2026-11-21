"""Authentic-human five-point extraction, provenance and non-destructive resumption."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from morphmimic.constants import HUMAN_JOINTS


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FivePointExportTests(unittest.TestCase):
  def setUp(self):
    self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
    self.root=Path(self.tmp.name);self.output=self.root/'export'
    self.raw=self.root/'original.bvh';self.raw.write_text('authentic source fixture')
    self.side=self.root/'human.npz'
    positions=np.arange(3*24*3,dtype=np.float32).reshape(3,24,3)
    quat=np.zeros((3,24,4),np.float32);quat[...,0]=1
    quat[:,20]=[.6,.8,0,0]
    self.arrays=dict(human_pos_w=positions,human_quat_w=quat,human_joint_names=np.array(HUMAN_JOINTS),
      human_joint_valid=np.ones(24,dtype=bool),fps=np.array(50),source_time_s=np.array([1.2,1.22,1.24]))
    self.save_side()
    self.row=dict(name='literal',dataset='seed',robot='q1',source_id='seed/literal',split='train',frames=3,fps=50,
      human_available=True,human_sidecar=str(self.side),human_sha256=sha(self.side),
      human_joint_valid=[True]*24,pair_source=dict(path=str(self.raw),sha256=sha(self.raw),fps=120,frames=200),
      pair_alignment=dict(robot_start_frame=17,robot_stop_frame=20,robot_tail_frames=1,human_tail_frames=2),
      source_body_window_original=[17,21])
    self.missing=dict(name='missing',dataset='motiondecode',robot='g1',source_id='motiondecode/missing',
                      human_available=False,frames=300,fps=50,split='train')
    self.manifest=self.root/'composite.json';self.save_manifest()

  def save_side(self): np.savez(self.side,**self.arrays)
  def save_manifest(self): self.manifest.write_text(json.dumps(dict(schema_version=1,complete=True,clips=[self.row,self.missing])))
  def export(self,**kwargs):
    from morphmimic.data.five_point_export import export_five_points
    return export_five_points(self.manifest,self.output,workers=2,**kwargs)

  def neutral_fixture(self):
    from morphmimic.data.neutral_bookends import source_frame_map
    mapping=source_frame_map(9,2);a=100;b=a+len(mapping);frames=b+100
    self.arrays['human_pos_w']=np.repeat(self.arrays['human_pos_w'][:1],frames,axis=0)
    self.arrays['human_quat_w']=np.repeat(self.arrays['human_quat_w'][:1],frames,axis=0)
    self.arrays['time_s']=np.arange(frames,dtype=float)/50
    self.arrays['source_frame_index']=np.full(frames,-1.,dtype=float)
    self.arrays['source_frame_index'][a:b]=mapping
    self.arrays['source_time_s']=np.full(frames,np.nan)
    self.arrays['source_time_s'][a:b]=1.2+mapping/50
    native=self.root/'native.npz';native.write_bytes(b'independent native bytes')
    self.row.update(frames=frames,output=str(native),output_sha256=sha(native),
      pair_alignment=dict(robot_start_frame=a,robot_stop_frame=b),
      neutral_contract=dict(robot='q1',fps=50,frames=frames,source_start_frame=a,source_stop_frame=b,
        entry_hold_end_frame=50,exit_hold_start_frame=b+50,entry_frames=50,exit_frames=50,
        original_source_frames=9,source_ramp_frames=2,source_clock_status='preserved_from_parent_sidecar'))
    self.save_side();self.row['human_sha256']=sha(self.side);self.save_manifest()
    return a,b

  def test_neutral_export_preserves_slow_source_clock_map_and_synthetic_provenance(self):
    a,b=self.neutral_fixture();report=self.export();item=report['clips'][0]
    with np.load(self.output/item['output']) as z:
      np.testing.assert_array_equal(z['source_frame_index'],self.arrays['source_frame_index'])
      np.testing.assert_array_equal(z['time_s'],self.arrays['time_s'])
      np.testing.assert_array_equal(z['source_time_s'],self.arrays['source_time_s'])
      self.assertTrue(np.isnan(z['source_time_s'][:a]).all())
      self.assertTrue((np.diff(z['source_time_s'][a:b])>0).all())
    self.assertEqual(item['neutral_contract'],self.row['neutral_contract'])
    self.assertTrue(item['synthetic_derived'])
    self.assertEqual(item['native_reference']['sha256'],self.row['output_sha256'])
    self.assertEqual(self.export(resume=True)['counts']['resumed'],1)

  def test_neutral_export_rejects_clock_map_contract_and_native_tampering(self):
    for what in ('clock','map','synthetic_index','synthetic_time','source_clock','contract','native'):
      with self.subTest(what=what):
        self.setUp();a,b=self.neutral_fixture()
        if what=='clock':self.arrays['time_s'][1]+=.01
        if what=='map':self.arrays['source_frame_index'][a+1]+=.1
        if what=='synthetic_index':self.arrays['source_frame_index'][0]=0.
        if what=='synthetic_time':self.arrays['source_time_s'][0]=0.
        if what=='source_clock':self.arrays['source_time_s'][a+1]+=.01
        if what=='contract':self.row['neutral_contract']['original_source_frames']=10
        if what=='native':Path(self.row['output']).write_bytes(b'changed native')
        self.save_side();self.row['human_sha256']=sha(self.side);self.save_manifest()
        with self.assertRaises(ValueError):self.export()

  def test_unknown_inherited_clock_requires_explicit_derived_exception(self):
    a,b=self.neutral_fixture();self.arrays['source_time_s'][a:b]=np.nan
    self.save_side();self.row['human_sha256']=sha(self.side);self.save_manifest()
    with self.assertRaises(ValueError):self.export()
    self.output=self.root/'explicit-derived'
    self.row['neutral_contract']['source_clock_status']='unavailable_in_inherited_derived_sidecar'
    self.save_manifest()
    with self.assertRaises(ValueError):self.export()
    self.output=self.root/'allowed-derived';self.row['derived']=True;self.save_manifest()
    self.assertEqual(self.export()['counts']['exported'],1)

  def test_literal_indices_timestamps_quaternions_and_provenance(self):
    report=self.export()
    self.assertEqual(report['counts'],{'exported':1,'resumed':0,'skipped':1})
    self.assertEqual(report['missing_human_by_dataset'],{'motiondecode':1})
    item=report['clips'][0]
    with np.load(self.output/item['output'],allow_pickle=False) as z:
      np.testing.assert_array_equal(z['human_pos_w'],self.arrays['human_pos_w'][:,[0,7,8,20,21]])
      np.testing.assert_array_equal(z['human_quat_w'],self.arrays['human_quat_w'][:,[0,7,8,20,21]])
      np.testing.assert_array_equal(z['source_time_s'],[1.2,1.22,1.24])
      np.testing.assert_array_equal(z['time_s'],[0.,.02,.04])
      np.testing.assert_array_equal(z['human_joint_indices'],[0,7,8,20,21])
      np.testing.assert_array_equal(z['human_joint_valid'],np.ones(5,dtype=bool))
      self.assertEqual(int(z['fps']),50)
    receipt=json.loads((self.output/item['receipt']).read_text())
    self.assertEqual(receipt['human_sha256'],self.row['human_sha256'])
    self.assertEqual(receipt['pair_source'],self.row['pair_source'])
    self.assertEqual(receipt['pair_alignment'],self.row['pair_alignment'])
    self.assertEqual(receipt['source_body_window_original'],[17,21])
    self.assertIn('no authentic human',report['skipped'][0]['reason'])
    self.assertEqual(sha(self.side),self.row['human_sha256'])

  def test_resume_verifies_output_receipt_and_source_before_skip(self):
    first=self.export();output=self.output/first['clips'][0]['output'];mtime=output.stat().st_mtime_ns
    second=self.export(resume=True)
    self.assertEqual(second['counts'],{'exported':0,'resumed':1,'skipped':1})
    self.assertEqual(mtime,output.stat().st_mtime_ns)
    self.arrays['human_pos_w'][0,0,0]+=1;self.save_side()
    with self.assertRaisesRegex(ValueError,'SHA|hash'):self.export(resume=True)
    self.assertEqual(mtime,output.stat().st_mtime_ns)

  def test_output_corruption_is_not_overwritten(self):
    report=self.export();output=self.output/report['clips'][0]['output'];output.write_bytes(b'corrupt')
    with self.assertRaisesRegex(ValueError,'output|SHA|hash'):self.export(resume=True)
    self.assertEqual(output.read_bytes(),b'corrupt')

  def test_bad_quaternion_invalid_mask_and_timing_rejected(self):
    for what in ('quat','mask','time'):
      with self.subTest(what=what):
        self.arrays['human_quat_w'][0,0]=[1,0,0,0];self.arrays['human_joint_valid'][:]=True
        self.arrays['source_time_s']=np.array([1.2,1.22,1.24])
        if what=='quat':self.arrays['human_quat_w'][0,0]=0
        if what=='mask':self.arrays['human_joint_valid'][20]=False
        if what=='time':self.arrays['source_time_s'][1]=1.3
        self.save_side();self.row['human_sha256']=sha(self.side);self.save_manifest()
        from morphmimic.data.five_point_export import export_five_points
        with self.assertRaises(ValueError):export_five_points(self.manifest,self.root/what,workers=1)

  def test_atomic_unreceipted_output_is_recovered_only_when_bytes_match(self):
    report=self.export();receipt=self.output/report['clips'][0]['receipt'];receipt.unlink()
    resumed=self.export(resume=True)
    self.assertTrue(receipt.exists());self.assertEqual(resumed['counts']['resumed'],1)
    receipt.unlink();(self.output/report['clips'][0]['output']).write_bytes(b'unrelated')
    with self.assertRaises(ValueError):self.export(resume=True)

  def test_raw_human_identity_and_unrelated_output_preserved(self):
    self.raw.write_text('changed original')
    with self.assertRaisesRegex(ValueError,'SHA|hash'):self.export()
    other=self.root/'unrelated';other.mkdir();(other/'keep').write_text('untouched')
    from morphmimic.data.five_point_export import export_five_points
    with self.assertRaises(FileExistsError):export_five_points(self.manifest,other)
    self.assertEqual((other/'keep').read_text(),'untouched')
    with self.assertRaises(ValueError):export_five_points(self.manifest,self.root/'badworkers',workers=17)

  def test_receipt_tampering_and_changed_manifest_are_rejected(self):
    result=self.export();path=self.output/result['clips'][0]['receipt']
    receipt=json.loads(path.read_text());receipt['pair_alignment']['robot_start_frame']=0
    path.write_text(json.dumps(receipt))
    with self.assertRaisesRegex(ValueError,'receipt'):self.export(resume=True)
    self.row['frames']=4;self.save_manifest()
    with self.assertRaisesRegex(ValueError,'differs'):self.export(resume=True)

  def test_path_escape_and_symlink_outputs_are_rejected(self):
    from morphmimic.data.five_point_export import export_five_points
    self.row['name']='../escape';self.save_manifest()
    with self.assertRaises(ValueError):self.export()
    other=self.root/'target';other.mkdir();link=self.root/'link';link.symlink_to(other,target_is_directory=True)
    with self.assertRaisesRegex(ValueError,'symbolic'):export_five_points(self.manifest,link)
    self.assertEqual(list(other.iterdir()),[])
