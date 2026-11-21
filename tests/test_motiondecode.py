import csv
import numpy as np
import pytest
from morphmimic.constants import JOINT_NAMES


def test_motiondecode_native_conversion_preserves_meters_and_quaternion_order(tmp_path):
    from morphmimic.data.motiondecode import native_rows
    header=['root_pos_x(m)','root_pos_y(m)','root_pos_z(m)','root_rot_w','root_rot_x','root_rot_y','root_rot_z']+['dof_'+n+'(rad)' for n in JOINT_NAMES]
    path=tmp_path/'motion.csv'
    with path.open('w') as f:
        w=csv.writer(f);w.writerow(header)
        for i in range(9):w.writerow([i/120,0,.8,1,0,0,0]+[.1]*29)
    rows,info=native_rows(path,source_fps=120)
    assert rows.shape==(3,36) and info['original_frames']==9
    np.testing.assert_allclose(rows[:,0],[0,1/30,2/30]);np.testing.assert_allclose(rows[:,2],.8)
    np.testing.assert_allclose(rows[:,3:7],[[0,0,0,1]]*3)
    with pytest.raises(ValueError):native_rows(path,source_fps=0)
    path.write_text(path.read_text().replace('root_pos_x(m)','root_pos_x(mm)'))
    with pytest.raises(ValueError,match='header'):native_rows(path,source_fps=120)
    path.write_text(','.join(header)+'\n')
    with pytest.raises(ValueError,match='no motion frames'):native_rows(path,source_fps=120)


@pytest.fixture
def bumi_source(tmp_path):
    """Use three independently audited SDK rows; never run expensive retarget IK."""
    from pathlib import Path
    pytest.importorskip('mujoco')
    pytest.importorskip('supermimic.bumi_ik')
    root=Path('/data/kk/blinkmorphmimic/prepared/motiondecode-diagnostic-120/native30/bumi')
    file=root/'motiondecode_baji_WP_Baji_Quan_17851016.csv'
    if not file.is_file():pytest.skip('requires the audited Bumi SDK fixture')
    source=tmp_path/'sdk.csv'
    np.savetxt(source,np.loadtxt(file,delimiter=',',max_rows=3),delimiter=',',fmt='%.9f')
    return source


def test_prepare_one_converts_sdk_columns_to_the_named_native_joints(bumi_source,tmp_path,monkeypatch):
    """Omitting the source order at the actual import boundary must fail this test."""
    import shutil
    from pathlib import Path
    from morphmimic.scripts import prepare_motiondecode as cli
    original=Path('/data/kk/data/MotionDecode/6380f7420c83b20b6ad50124bc6cb829a7c0b178/samples/4. Martial_Arts/WP_Baji_Quan.csv')
    source=tmp_path/'WP_Baji_Quan.csv'
    with original.open() as f:source.write_text(''.join(next(f) for _ in range(10)))
    def retarget(_source,output):
        shutil.copyfile(bumi_source,output)
        return {}
    retarget.adaptation={'test_fixture':True}
    monkeypatch.setattr(cli,'_retarget_function',lambda _repo:retarget)
    monkeypatch.setattr(cli,'q1_retarget',lambda _repo:lambda _task:(None,{'status':'failed','error':'fixture excludes Q1'}))
    result=cli.prepare_one((source,'samples/4. Martial_Arts/WP_Baji_Quan.csv',tmp_path/'out','/data/kk/supermimic_vibed',120))
    row=next(r for r in result['clips'] if r['robot']=='bumi')
    # Hand-checked SDK indices in waist, left/right arm, left/right leg native order.
    expected=np.loadtxt(bumi_source,delimiter=',')[0,7:][[2,5,9,13,17,6,10,14,18,0,3,7,11,15,19,1,4,8,12,16,20]]
    with np.load(row['output']) as z:np.testing.assert_allclose(z['joint_pos'][0],expected,atol=2e-7,rtol=0)


def test_prepare_one_refuses_a_legacy_conversion_receipt(tmp_path):
    import hashlib,json
    from morphmimic.scripts.prepare_motiondecode import prepare_one
    source=tmp_path/'WP_Baji_Quan.csv';source.write_bytes(b'raw source')
    rel='samples/4. Martial_Arts/WP_Baji_Quan.csv'
    name='motiondecode_baji_WP_Baji_Quan_'+hashlib.sha256(rel.encode()).hexdigest()[:8]
    out=tmp_path/'out';(out/'receipts').mkdir(parents=True)
    old={'identity':dict(original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),original_fps=120,relative_source=rel),'clips':[]}
    (out/'receipts'/(name+'.json')).write_text(json.dumps(old))
    with pytest.raises(ValueError,match='Legacy.*conversion'):
        prepare_one((source,rel,out,'/unused',120))


def test_post_conversion_validation_rejects_the_published_wrong_order(bumi_source):
    import morphmimic.data.motiondecode as module
    validate=getattr(module,'validate_bumi_reference',None)
    assert callable(validate),'post-conversion validation is missing'
    from pathlib import Path
    from morphmimic.data.local_prepare import convert_motion
    wrong=bumi_source.with_name('wrong.npz')
    convert_motion(bumi_source,wrong,'bumi',30)
    with pytest.raises(ValueError,match='joint.*mapping'):
        validate(bumi_source,wrong)


@pytest.mark.parametrize('field,match',[('joint_names','names'),('fps','50 Hz'),('body_pos_w','FK'),('joint_vel','velocity')])
def test_post_conversion_validation_rejects_corrupt_outputs(bumi_source,field,match):
    import morphmimic.data.motiondecode as module
    convert=getattr(module,'convert_reference',None)
    assert callable(convert),'validated conversion is missing'
    output=bumi_source.with_name('good.npz');convert(bumi_source,output,'bumi')
    with np.load(output) as z:arrays={k:z[k].copy() for k in z.files}
    if field=='joint_names':arrays[field]=arrays[field][::-1]
    elif field=='fps':arrays[field]=np.array(49)
    elif field=='body_pos_w':arrays[field][1,3,0]+=.1
    else:arrays[field][1,0]+=.1
    np.savez(output,**arrays)
    with pytest.raises(ValueError,match=match):module.validate_bumi_reference(bumi_source,output)


def test_failed_bumi_validation_preserves_existing_output(bumi_source):
    import morphmimic.data.motiondecode as module
    convert=getattr(module,'convert_reference',None)
    assert callable(convert),'validated conversion is missing'
    rows=np.loadtxt(bumi_source,delimiter=',');rows[:,7+12]=-5
    np.savetxt(bumi_source,rows,delimiter=',')
    target=bumi_source.with_name('preserved.npz');target.write_bytes(b'existing data')
    with pytest.raises(ValueError,match='joint limit'):convert(bumi_source,target,'bumi')
    assert target.read_bytes()==b'existing data'
    assert sorted(p.name for p in target.parent.iterdir())==['preserved.npz','sdk.csv']
