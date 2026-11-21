"""CPU preview selection is explicit and never silently enables CUDA graphs."""
from inspect import signature
import pytest
from preview.server import PolicySession,Preview


def test_cpu_selection_keeps_existing_warp_default():
    assert signature(PolicySession).parameters['backend'].default=='warp'
    assert signature(Preview).parameters['backend'].default=='warp'


def test_cpu_control_graph_and_gpu_actor_are_refused_before_loading():
    with pytest.raises(ValueError,match='CPU.*control graph'):
        PolicySession('q1',None,None,'cpu',backend='cpu',control_graph=True)
    with pytest.raises(ValueError,match='CPU.*actor'):
        PolicySession('q1',None,None,'cuda:0',backend='cpu')


def test_cli_refuses_cpu_graph_and_non_cpu_actor():
    import subprocess,sys
    from pathlib import Path
    script=Path(__file__).parents[1]/'preview/server.py'
    for arguments in (['--backend','cpu','--control-graph'],['--backend','cpu','--device','cuda:0'],
                      ['--backend','cpu','--compare-devices','cpu,cpu,cuda:0']):
        p=subprocess.run([sys.executable,str(script),'--data-root','/unused',*arguments],capture_output=True,text=True,timeout=10)
        assert p.returncode==2 and 'CPU preview' in p.stderr


def test_native_cpu_world_rejects_cuda_storage_without_constructing_scene():
    from preview.cpu_backend import CpuPreviewEnv
    with pytest.raises(ValueError,match='CPU'):
        CpuPreviewEnv(None,device='cuda:0')
