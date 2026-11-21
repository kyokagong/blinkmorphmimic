"""A rebuilt catalog must be complete, preserve peers, and never publish partial data."""
import importlib,json
from pathlib import Path

import numpy as np
import pytest
from test_motiondecode import bumi_source


@pytest.fixture
def legacy_catalog(bumi_source,tmp_path):
    from morphmimic.data.local_prepare import convert_motion,sha256
    old=tmp_path/'legacy';native=tmp_path/'native';native.mkdir()
    name='motiondecode_baji_fixture';source=native/(name+'.csv')
    source.write_bytes(bumi_source.read_bytes())
    clips=[]
    for robot in ['q1','bumi','g1']:
        file=old/robot/(name+'.npz');file.parent.mkdir(parents=True)
        if robot=='bumi':stats=convert_motion(source,file,robot,30)
        else:np.savez(file,preserved=np.array([1,2,3]));stats={}
        clips.append(dict(robot=robot,name=name,source_id='motiondecode/fixture',group_id='motiondecode/fixture',split='val',
                          output=f'{robot}/{name}.npz',output_sha256=sha256(file),**stats))
    manifest=dict(schema_version=1,source_fps=120,source_fps_requires_confirmation=False,
                  complete=False,clips=clips,errors=[{'robot':'q1','error':'preserve this original exclusion'}])
    (old/'manifest.json').write_text(json.dumps(manifest))
    return old,native,tmp_path/'v2',manifest


def implementation():
    import morphmimic.scripts as scripts
    from importlib.util import find_spec
    assert find_spec(scripts.__name__+'.rebuild_motiondecode_bumi') is not None,'complete versioned rebuild is missing'
    return importlib.import_module(scripts.__name__+'.rebuild_motiondecode_bumi')


def test_rebuild_preserves_peer_bytes_and_publishes_every_bumi(legacy_catalog):
    old,native,out,baseline=legacy_catalog
    result=implementation().rebuild(old,native,out,'/data/kk/supermimic_vibed',workers=1)
    assert result['bumi_rebuild']['complete'] is True
    assert result['bumi_rebuild']['expected_clips']==result['bumi_rebuild']['validated_clips']==1
    assert result['errors']==baseline['errors'] and result['complete'] is False
    assert result['common_three_robot_source_count']==1
    for robot in ['q1','g1']:
        relative=f'{robot}/motiondecode_baji_fixture.npz'
        assert (out/relative).read_bytes()==(old/relative).read_bytes()
    with np.load(out/'bumi/motiondecode_baji_fixture.npz') as z:
        expected=np.loadtxt(native/'motiondecode_baji_fixture.csv',delimiter=',')[0,7:][[2,5,9,13,17,6,10,14,18,0,3,7,11,15,19,1,4,8,12,16,20]]
        np.testing.assert_allclose(z['joint_pos'][0],expected,atol=2e-7,rtol=0)
    assert json.loads((out/'manifest.json').read_text())==result


@pytest.mark.parametrize('problem',['missing_csv','tampered_npz','nonempty_output'])
def test_rebuild_refuses_bad_inputs_without_publishing_a_catalog(legacy_catalog,problem):
    old,native,out,_=legacy_catalog
    if problem=='missing_csv':(native/'motiondecode_baji_fixture.csv').unlink()
    elif problem=='tampered_npz':(old/'g1/motiondecode_baji_fixture.npz').write_bytes(b'changed')
    else:out.mkdir();(out/'preserve.txt').write_bytes(b'keep')
    with pytest.raises((ValueError,FileNotFoundError,FileExistsError)):
        implementation().rebuild(old,native,out,'/data/kk/supermimic_vibed',workers=1)
    assert not (out/'manifest.json').exists()
    if problem=='nonempty_output':assert (out/'preserve.txt').read_bytes()==b'keep'
