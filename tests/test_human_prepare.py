from pathlib import Path
import pytest

def test_inventory_distinguishes_human_from_robot_and_ignores_download_cache(tmp_path):
    from morphmimic.data.human_prepare import inventory_sources
    human=tmp_path/'seed/soma_uniform/bvh/date/clip.BVH';human.parent.mkdir(parents=True);human.write_text('HIERARCHY\nROOT Hips\n')
    robot=tmp_path/'MotionDecode/revision/samples/4. Martial_Arts/WP_Baji_Quan.csv';robot.parent.mkdir(parents=True);robot.write_text('root_pos_x(m),root_rot_w,dof_left_hip_pitch_joint(rad)\n0,1,0\n')
    dance=tmp_path/'MotionDecode/revision/samples/5.Dance/genre/plain_name.csv';dance.parent.mkdir(parents=True);dance.write_text(robot.read_text())
    hidden=tmp_path/'MotionDecode/.cache/download/fake.bvh';hidden.parent.mkdir(parents=True);hidden.write_text('cached incomplete file')
    result=inventory_sources(tmp_path)
    assert [r['path'] for r in result['human_candidates']]==[str(human)]
    assert {r['path'] for r in result['motiondecode_selected']}=={str(robot),str(dance)}
    assert result['robot_csv']['motiondecode']['count']==2
    assert all(r['human_source'] is None and r['fps'] is None for r in result['motiondecode_selected'])
    assert result['errors']==[]

def test_inventory_does_not_follow_links_or_treat_robot_npz_as_human(tmp_path):
    from morphmimic.data.human_prepare import inventory_sources
    (tmp_path/'robot.npz').write_bytes(b'not a human sidecar')
    (tmp_path/'soma.bvh').symlink_to('/etc/passwd')
    result=inventory_sources(tmp_path)
    assert result['human_candidates']==[]
    assert len(result['skipped_symlinks'])==1

def test_inventory_examples_are_bounded_and_fps_never_guessed(tmp_path):
    from morphmimic.data.human_prepare import inventory_sources
    folder=tmp_path/'supermimic_g1/lafan1_g1';folder.mkdir(parents=True)
    for i in range(6):(folder/f'{i}.csv').write_text('robot CSV')
    result=inventory_sources(tmp_path,max_examples=2)
    assert result['robot_csv']['lafan1']['count']==6
    assert len(result['robot_csv']['lafan1']['examples'])==2
    assert not result['human_candidates']

def test_five_points_are_exact_source_slots_and_bad_rotations_are_rejected():
    import numpy as np
    from morphmimic.data.human_prepare import five_points, validate_human
    from morphmimic.constants import HUMAN_JOINTS
    pos=np.arange(2*24*3,dtype=np.float32).reshape(2,24,3)
    quat=np.zeros((2,24,4),dtype=np.float32);quat[...,0]=1
    human=dict(human_pos_w=pos,human_quat_w=quat,human_joint_names=np.array(HUMAN_JOINTS),fps=np.array(50))
    p,q=five_points(human)
    np.testing.assert_array_equal(p,pos[:,[0,7,8,20,21]])
    np.testing.assert_array_equal(q,quat[:,[0,7,8,20,21]])
    human['human_quat_w'][0,7]=0
    with pytest.raises(ValueError,match='quaternion'):validate_human(human)

def test_pairing_crops_padding_and_uses_common_time_without_stretching():
    import numpy as np
    from morphmimic.data.human_prepare import align_pair
    robot={'joint_pos':np.arange(210.)[:,None], 'fps':np.array(50), 'joint_names':np.array(['joint'])}
    human={'human_pos_w':np.broadcast_to(np.arange(61.)[:,None,None],(61,24,3)).copy(),
           'human_quat_w':np.broadcast_to(np.array([1.,0,0,0]),(61,24,4)).copy()}
    before=robot['joint_pos'].copy()
    r,h,record=align_pair(robot,human,start=150,stop=200,frames=50)
    np.testing.assert_array_equal(r['joint_pos'][:,0],np.arange(150,200))
    np.testing.assert_array_equal(h['human_pos_w'][:,0,0],np.arange(50))
    np.testing.assert_array_equal(record['time_s'],np.arange(50)/50)
    np.testing.assert_array_equal(robot['joint_pos'],before)
    assert record['robot_start_frame']==150 and record['human_tail_frames']==11
    assert list(r['joint_names'])==['joint']
    with pytest.raises(ValueError):align_pair(robot,human,start=150,stop=200,frames=51)

def test_seed_index_matches_actual_mirror_identity_and_rejects_ambiguity(tmp_path):
    import csv
    from morphmimic.data.human_prepare import seed_index
    p=tmp_path/'metadata.csv'
    rows=[dict(filename='dance_a__A001',is_mirror='False',move_duration_frames='120',move_soma_uniform_path='soma_uniform/bvh/day/dance_a__A001.bvh'),
          dict(filename='dance_a__A001_M',is_mirror='True',move_duration_frames='120',move_soma_uniform_path='soma_uniform/bvh/day/dance_a__A001_M.bvh')]
    def write(rows):
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    write(rows)
    result=seed_index(p,{'dance_a__A001_M'})
    assert result['dance_a__A001_M']['is_mirror'] is True
    assert result['dance_a__A001_M']['source_frames']==120
    write(rows+[rows[1]])
    with pytest.raises(ValueError,match='duplicate'):seed_index(p,{'dance_a__A001_M'})

def test_real_lafan_conversion_uses_source_positions_and_marks_missing_hand_tips():
    import numpy as np
    from morphmimic.data.lafan_convert import lafan_slot_poses
    from morphmimic.data.bvh import load_bvh_poses
    from morphmimic.data.human_convert import YUP_CM_TO_ZUP_M
    path=Path('/data/kk/data/blinkmorph_human_raw/lafan1/dance1_subject1.bvh')
    if not path.exists():pytest.skip('official LAFAN fixture not mounted')
    p,q,dt,valid=lafan_slot_poses(path)
    original,_,names,_=load_bvh_poses(path);original,_=YUP_CM_TO_ZUP_M(original,np.broadcast_to([1.,0,0,0],(*original.shape[:-1],4)))
    expected=np.stack([-original[...,1],original[...,0],original[...,2]],axis=-1)
    np.testing.assert_allclose(p[:,0],expected[:,names['Hips']],atol=1e-7)
    np.testing.assert_allclose(p[:,20],expected[:,names['LeftHand']],atol=1e-7)
    assert p.shape==(len(original),24,3) and q.shape==(len(original),24,4) and round(1/dt)==30
    np.testing.assert_allclose(np.linalg.norm(q,axis=-1),1,atol=1e-6)
    assert valid[[0,7,8,20,21]].all() and not valid[22:].any()
    np.testing.assert_array_equal(p[:,22],p[:,20])

def test_preparation_preserves_parent_files_and_writes_time_bound_three_robot_pairs(tmp_path):
    import json,numpy as np
    from morphmimic.constants import HUMAN_JOINTS
    from morphmimic.data.human_prepare import write_pairs
    rows=[];robots={}
    for r,n,start in (('q1',210,150),('bumi',52,0),('g1',54,0)):
        robots[r]={'joint_pos':np.arange(n)[:,None].astype(float),'joint_names':np.array(['joint']),'fps':np.array(50)}
        rows.append(dict(name='clip',robot=r,dataset='lafan1',source_id='lafan1/clip',group_id='lafan1/clip',split='train',
                         frames=n,phases={'entry':3.,'dance':4.} if r=='q1' else {}))
    human=dict(human_pos_w=np.broadcast_to(np.arange(61.)[:,None,None],(61,24,3)).copy(),
               human_quat_w=np.broadcast_to([1.,0,0,0],(61,24,4)).copy(),human_joint_names=np.array(HUMAN_JOINTS),fps=np.array(50))
    before={r:z['joint_pos'].copy() for r,z in robots.items()}
    records=write_pairs(tmp_path,rows,robots,human,source=dict(path='raw.bvh',sha256='a'*64,fps=30,frames=37))
    assert len(records)==3
    for record in records:
        assert record['frames']==50 and record['phases']=={} and record['fps']==50 and record['alignment_eligible']
        z=np.load(tmp_path/record['robot']/'clip.npz');h=np.load(tmp_path/'human50'/record['robot']/'clip.npz')
        assert len(z['joint_pos'])==len(h['human_pos_w'])==50
        np.testing.assert_array_equal(h['source_time_s'],np.arange(50)/50)
        assert record['pair_source']['sha256']=='a'*64
        np.testing.assert_array_equal(before[record['robot']],robots[record['robot']]['joint_pos'])
    with pytest.raises(FileExistsError):write_pairs(tmp_path,rows,robots,human,source={})

def test_atomic_writer_does_not_delete_another_incomplete_file(tmp_path):
    import numpy as np
    from morphmimic.data.human_prepare import _atomic_npz
    path=tmp_path/'clip.npz';temporary=tmp_path/'clip.npz.tmp';temporary.write_bytes(b'existing partial')
    with pytest.raises(FileExistsError):_atomic_npz(path,{'x':np.array([1])})
    assert temporary.read_bytes()==b'existing partial'

def test_atomic_writer_can_resume_only_identical_arrays(tmp_path):
    import numpy as np
    from morphmimic.data.human_prepare import _atomic_npz
    path=tmp_path/'clip.npz';_atomic_npz(path,{'x':np.array([1])});original=path.read_bytes()
    _atomic_npz(path,{'x':np.array([1])},resume=True)
    assert path.read_bytes()==original
    with pytest.raises(ValueError):_atomic_npz(path,{'x':np.array([2])},resume=True)


@pytest.mark.parametrize('changed',['output','human_sidecar'])
def test_pair_identity_checks_actual_arrays_even_when_manifest_is_unchanged(tmp_path,changed):
    import hashlib,json,numpy as np
    from morphmimic.data.human_prepare import verify_prepared_pairs
    record=dict(robot='q1',name='dance',output='q1/dance.npz',human_sidecar='human50/q1/dance.npz')
    for field,key in (('output','output_sha256'),('human_sidecar','human_sha256')):
        p=tmp_path/record[field];p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,x=np.array([1.]))
        record[key]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(clips=[record])))
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert verify_prepared_pairs(tmp_path,robot='q1',names=['dance.npz'])==digest
    np.savez(tmp_path/record[changed],x=np.array([2.]))
    with pytest.raises(ValueError,match='SHA'):verify_prepared_pairs(tmp_path,robot='q1',names=['dance.npz'])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==digest


@pytest.mark.parametrize('change',['root','split'])
def test_subset_identity_checks_resolved_files_and_validation_role(tmp_path,change):
    import hashlib,json,numpy as np,yaml
    from morphmimic.data.human_prepare import verify_prepared_subset
    rows=[]
    for name,split in (('train_clip','train'),('val_clip','val')):
        row=dict(robot='q1',name=name,split=split,output=f'q1/{name}.npz',human_sidecar=f'human50/q1/{name}.npz')
        for field,key in (('output','output_sha256'),('human_sidecar','human_sha256')):
            p=tmp_path/row[field];p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,x=np.array([1.]))
            row[key]=hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(row)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(clips=rows)))
    subset=tmp_path/'val.yaml';spec=dict(root=str(tmp_path/'q1'),clips=['val_clip.npz']);subset.write_text(yaml.safe_dump(spec))
    verify_prepared_subset(tmp_path,subset,robot='q1',split='val')
    if change=='root':
        (tmp_path/'wrong').mkdir();np.savez(tmp_path/'wrong/val_clip.npz',x=np.array([2.]));spec['root']=str(tmp_path/'wrong')
    else:spec['clips']=['train_clip.npz']
    subset.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError,match='subset'):verify_prepared_subset(tmp_path,subset,robot='q1',split='val')
