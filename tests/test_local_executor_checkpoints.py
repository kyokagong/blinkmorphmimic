import copy,json,tempfile,unittest
from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from morphmimic.train.checkpoint_set import write_checkpoint_set, read_checkpoint_set, verify_checkpoint_member
from morphmimic.train.optimizations import cpu_snapshot

class CheckpointSetTests(unittest.TestCase):
    def make(self):
        metadata={'schema':1,'executor':'single_gpu_sequential','robots':['q1','bumi','g1'],'world_size':1,'weighting':'equal_robot','iteration':2}
        states={r:{'iteration':2,'cfg':{'robot':r,'executor':'single_gpu_sequential'},'policy':{'w':torch.tensor([3.])},'local_executor':copy.deepcopy(metadata)} for r in metadata['robots']}
        return states,{'local_executor':metadata,'rng':torch.tensor([5]),'iteration':2}

    def test_committed_set_roundtrip_and_immutable_cpu_snapshot(self):
        states,coordinator=self.make()
        with tempfile.TemporaryDirectory() as tmp:
            snapshot=cpu_snapshot({'robots':states,'coordinator':coordinator})
            states['q1']['policy']['w'].add_(10)
            write_checkpoint_set(Path(tmp),snapshot)
            loaded,state=read_checkpoint_set(Path(tmp))
            torch.testing.assert_close(loaded['q1']['policy']['w'],torch.tensor([3.]))
            self.assertEqual(state['iteration'],2)
            verify_checkpoint_member(Path(tmp)/'q1/ckpt_2.pt',loaded['q1'])
            verify_checkpoint_member(Path(tmp)/'q1/latest.pt',loaded['q1'])
            with self.assertRaises(FileExistsError):write_checkpoint_set(Path(tmp),snapshot)

    def test_missing_or_tampered_member_is_rejected(self):
        states,coordinator=self.make()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_checkpoint_set(root,{'robots':states,'coordinator':coordinator})
            (root/'bumi/ckpt_2.pt').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'hash'):read_checkpoint_set(root)

    def test_uncommitted_or_mixed_iteration_rejected(self):
        states,coordinator=self.make()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises((ValueError,FileNotFoundError)):read_checkpoint_set(root)
            states['g1']['iteration']=3
            with self.assertRaisesRegex(ValueError,'iteration'):write_checkpoint_set(root,{'robots':states,'coordinator':coordinator})
            self.assertFalse((root/'commits/latest.json').exists())

if __name__=='__main__':unittest.main()


def test_receipt_requires_all_canonical_members_before_any_load(tmp_path,monkeypatch):
    import pytest
    states,coordinator=CheckpointSetTests().make()
    write_checkpoint_set(tmp_path,{'robots':states,'coordinator':coordinator})
    path=tmp_path/'commits/latest.json';original=json.loads(path.read_text())
    original_bytes={name:(tmp_path/name).read_bytes() for name in original['files']}
    # Even an unhashed modified full checkpoint must never be deserialized.
    for missing in list(original['files']):
        receipt=copy.deepcopy(original);del receipt['files'][missing]
        path.write_text(json.dumps(receipt))
        (tmp_path/missing).write_bytes(b'unhashed altered member')
        with patch_load_forbidden(monkeypatch):
            with pytest.raises(ValueError,match='member'):read_checkpoint_set(tmp_path)
        # Restore bytes and hashes for the next missing member.
        (tmp_path/missing).write_bytes(original_bytes[missing])
        path.write_text(json.dumps(original))
        if missing.startswith('g1/ckpt'):
            assert missing not in receipt['files']


def patch_load_forbidden(monkeypatch):
    from contextlib import contextmanager
    @contextmanager
    def guard():
        with monkeypatch.context() as m:
            def forbidden(*a,**kw):raise AssertionError('torch.load reached before receipt validation')
            m.setattr(torch,'load',forbidden)
            yield
    return guard()


def test_receipt_rejects_unsafe_extra_and_wrong_iteration_members(tmp_path,monkeypatch):
    import pytest
    states,coordinator=CheckpointSetTests().make()
    write_checkpoint_set(tmp_path,{'robots':states,'coordinator':coordinator})
    path=tmp_path/'commits/latest.json';original=json.loads(path.read_text())
    for extra in ('../outside.pt','q1/../g1/ckpt_2.pt','q1/ckpt_3.pt','unused.pt'):
        receipt=copy.deepcopy(original);receipt['files'][extra]='0'*64;path.write_text(json.dumps(receipt))
        with patch_load_forbidden(monkeypatch):
            with pytest.raises(ValueError,match='member'):read_checkpoint_set(tmp_path)
    for value in (-1,True,'2'):
        receipt=copy.deepcopy(original);receipt['iteration']=value;path.write_text(json.dumps(receipt))
        with patch_load_forbidden(monkeypatch):
            with pytest.raises(ValueError,match='iteration'):read_checkpoint_set(tmp_path)
    path.write_text(json.dumps(original))
    receipt=copy.deepcopy(original);receipt['local_executor']['robots']=['q1','bumi','../g1'];path.write_text(json.dumps(receipt))
    with patch_load_forbidden(monkeypatch):
        with pytest.raises(ValueError,match='robot'):read_checkpoint_set(tmp_path)


def test_coverage_members_follow_explicit_writer_inventory(tmp_path,monkeypatch):
    import pytest
    states,coordinator=CheckpointSetTests().make()
    states['q1']['coverage']={'clip_names':['clip'],'counts':{'collected':torch.ones(1,dtype=torch.long)}}
    write_checkpoint_set(tmp_path,{'robots':states,'coordinator':coordinator})
    path=tmp_path/'commits/latest.json';receipt=json.loads(path.read_text())
    assert receipt['coverage_robots']==['q1']
    read_checkpoint_set(tmp_path)
    del receipt['files']['q1/coverage_2.json'];path.write_text(json.dumps(receipt))
    with patch_load_forbidden(monkeypatch):
        with pytest.raises(ValueError,match='member'):read_checkpoint_set(tmp_path)


def test_fixed_checkpoint_and_latest_paths_resolve_same_committed_set(tmp_path):
    states,coordinator=CheckpointSetTests().make()
    write_checkpoint_set(tmp_path,{'robots':states,'coordinator':coordinator})
    for name in ('ckpt_2.pt','latest.pt'):
        loaded,owner=read_checkpoint_set(tmp_path/'g1'/name)
        assert owner['iteration']==2 and set(loaded)=={'q1','bumi','g1'}


def test_receipt_fixed_iteration_and_coverage_declaration_cannot_be_redirected(tmp_path,monkeypatch):
    import pytest
    states,coordinator=CheckpointSetTests().make()
    write_checkpoint_set(tmp_path,{'robots':states,'coordinator':coordinator})
    receipt=json.loads((tmp_path/'commits/latest.json').read_text())
    (tmp_path/'commits/iteration_3.json').write_text(json.dumps(receipt))
    with patch_load_forbidden(monkeypatch):
        with pytest.raises(ValueError,match='iteration'):read_checkpoint_set(tmp_path/'g1/ckpt_3.pt')
    for coverage in (['unknown'],['q1','q1'],['g1','q1']):
        altered=copy.deepcopy(receipt);altered['coverage_robots']=coverage
        (tmp_path/'commits/latest.json').write_text(json.dumps(altered))
        with patch_load_forbidden(monkeypatch):
            with pytest.raises(ValueError,match='coverage'):read_checkpoint_set(tmp_path)
