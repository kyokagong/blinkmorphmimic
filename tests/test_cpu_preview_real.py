"""Real fixed-checkpoint CPU preview smoke; no GPU or rendered reference animation."""
from functools import partial
import hashlib,json,os
from pathlib import Path
import numpy as np
import pytest,torch
from preview.server import Catalog,FixedCheckpoint,PolicySession
from preview.comparison import ComparisonSession
from preview.shared_policy import SharedPolicy

DATA=Path('/data/kk/blinkmorphmimic/prepared/allopt-20261112/manifest.json')
RUN=Path('/data/kk/blinkmorphmimic/runs/scratch-muon-2048-5000-20261113-r2')
CURRENT=Path('/data/kk/blinkmorphmimic/prepared/inplace-recovery-neutral-20261117-v2')
VIEW= CURRENT if (CURRENT/'manifest.json').exists() else Path('/data/kk/blinkmorphmimic/prepared/optimized-preview-20261112-capacity')
PAIRED_DIGEST=hashlib.sha256((CURRENT/'manifest.json').read_bytes()).hexdigest() if (CURRENT/'manifest.json').exists() else '088d90efbe44ab8af6bfc977bcd7a74e49138c36a2c3c7cf416ff4bd96af08a0'
DIGEST='088d90efbe44ab8af6bfc977bcd7a74e49138c36a2c3c7cf416ff4bd96af08a0'
ROBOTS=('q1','bumi','g1')
EXPECTED={'q1':'aeb7026b1029dcab5db5bba32142d0cb48e7f4e95ce663da500fffb14268a0a2','bumi':'d72d70cecb284449868c10146fd31b106f3529f43d9e94c1127af2cd55ad0f09','g1':'95f67305b43db88218bf965e841a8ab8ae265986fbfa2fc428e8daff0a18c8be'}

@pytest.fixture(scope='module')
def assets():
    assert not torch.cuda.is_available()
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==DIGEST
    manifest=json.loads(((CURRENT/'manifest.json') if (CURRENT/'manifest.json').exists() else DATA).read_text());checkpoints={r:FixedCheckpoint.pin(RUN/r/'ckpt_5000.pt',r) for r in ROBOTS}
    assert {r:c.sha256 for r,c in checkpoints.items()}==EXPECTED
    return manifest,checkpoints


def row(manifest,robot,dataset):
    return min((r for r in manifest['clips'] if r['robot']==robot and r['dataset']==dataset and r['split']=='val'),key=lambda r:r['frames'])


def test_real_single_motiondecode_and_human_five_h1(assets):
    manifest,checkpoints=assets
    for interface,dataset in (('motion','motiondecode'),('human','seed')):
        clip=row(manifest,'q1',dataset)
        session=PolicySession('q1',Path(clip['output']),checkpoints['q1'],'cpu',render=False,backend='cpu',interface=interface,
            human_root=manifest['human_root'] if interface=='human' else None,training_manifest_sha256=DIGEST,paired_manifest_sha256=PAIRED_DIGEST)
        try:
            assert session.env.device.type=='cpu' and session.status()['physics_backend']=='cpu'
            if interface=='human':
                valid=session.goals['human_joint_valid'];assert bool(valid[0,[0,7,8,20,21]].all())
                assert bool(session.goals['goal_kp'].abs().sum()>0)
                assert session.status()['input_mode']=='human-five-h1'
            else:assert not session.command.library.has_human
            session.step();assert session.steps==1 and not session.failed
            session.reset();assert session.steps==0 and not session.failed and not session.completed
            history=session.history.flat().reshape(1,session.meta.hist_len,-1)
            torch.testing.assert_close(history[:,0],history[:,-1])
            session.env.native_data.qpos[2]+=1.
            session.env.sim.forward();session.step();assert session.failed
            with pytest.raises(RuntimeError,match='reset'):session.step()
            session.reset();session.step();assert session.steps==1
        finally:session.close()


@pytest.mark.parametrize('interface,dataset',[('motion','motiondecode'),('human','seed')])
def test_real_three_body_shared_cpu_actor(assets,interface,dataset):
    manifest,checkpoints=assets
    selected=row(manifest,'q1',dataset);matches={r:next(x for x in manifest['clips'] if x['robot']==r and x['source_id']==selected['source_id']) for r in ROBOTS}
    paths={r:Path(matches[r]['output']) for r in ROBOTS}
    extra=dict(interface=interface,training_manifest_sha256=DIGEST,paired_manifest_sha256=PAIRED_DIGEST)
    if interface=='human':extra.update(human_points='five',human_root=manifest['human_root'])
    sf=partial(PolicySession,backend='cpu',**extra)
    pf=partial(SharedPolicy,interface=interface,human_points='five',training_manifest_sha256=DIGEST)
    session=ComparisonSession(paths,checkpoints,('cpu',)*3,session_factory=sf,policy_factory=pf)
    try:
        assert session.policy.device=='cpu' and session.executor is None
        session.step();status=session.status()
        assert session.steps==1 and not session.failed and status['shared_policy'] and status['actor_forward_calls']==1
        assert all(x['physics_backend']=='cpu' and x['physics']['overflow_bits']==0 for x in status['participants'])
        session.reset();assert session.steps==0 and session.policy.calls==0
        session.step();assert session.steps==1 and not session.failed
    finally:session.close()

@pytest.mark.parametrize('robot',['bumi','g1'])
def test_real_reset_clamps_out_of_range_reference_to_native_soft_limits(assets,robot,tmp_path):
    manifest,checkpoints=assets
    clip=next(r for r in manifest['clips'] if r['robot']==robot and r['name']=='dance_seed_dance_retro_jackson_jump_on_R_002__A308')
    path=Path(clip['output'])
    if 'neutral_contract' in clip:
        # Exercise the inherited out-of-soft-limit source pose, after the new neutral entry.
        with np.load(path,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        frames=clip['frames'];start=clip['neutral_contract']['source_start_frame']
        arrays={k:(v[start:] if v.ndim and len(v)==frames and k not in ('body_names','joint_names') else v) for k,v in arrays.items()}
        path=tmp_path/(robot+'.npz');np.savez(path,**arrays)
    session=PolicySession(robot,path,checkpoints[robot],'cpu',render=False,backend='cpu',training_manifest_sha256=DIGEST,paired_manifest_sha256=PAIRED_DIGEST)
    try:
        env=session.env;model=env.sim.mj_model
        joints=[model.joint('robot/'+name).id for name in env.spec.joint_names]
        ranges=torch.as_tensor(model.jnt_range[joints],dtype=torch.float32)
        factor=env.cfg.scene.entities['robot'].articulation.soft_joint_pos_limit_factor
        middle=ranges.mean(-1);radius=(ranges[:,1]-ranges[:,0])*factor/2
        raw=torch.as_tensor(env.reference_qpos(0)[env.joint_qadr],dtype=torch.float32)
        expected=raw.clamp(middle-radius,middle+radius)
        assert not torch.equal(raw,expected),'fixture must exercise an out-of-soft-limit reference'
        actual=torch.as_tensor(env.native_data.qpos[env.joint_qadr],dtype=torch.float32)
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-7)
    finally:session.close()
