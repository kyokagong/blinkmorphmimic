"""Single-device interleaving has global intervals, not per-body WM timings."""
from types import SimpleNamespace
from collections import defaultdict
import pytest
from morphmimic.train.trainer import _PhaseTimer, _Window, Trainer
from morphmimic.train.single_gpu import SingleGpuMultiRobotExecutor


def test_global_intervals_and_legacy_intervals_keep_distinct_tags():
    global_timer = _PhaseTimer(False, names=('global_collect', 'global_learning'))
    global_timer.marks = [0., 2., 7.]
    window = _Window(); window.timers = [global_timer, global_timer]
    assert window._phase_times() == {'time/global_collect': 2., 'time/global_learning': 5.}
    legacy = _PhaseTimer(False); legacy.marks = [0., 2., 5., 9.]
    window.timers = [legacy]
    assert window._phase_times() == {'time/collect': 2., 'time/wm': 3., 'time/actor_critic': 4.}
    window.timers.append(global_timer)
    with pytest.raises(ValueError, match='phase'):
        window._phase_times()
    window.timers = [global_timer]; global_timer.marks.pop()
    with pytest.raises(ValueError, match='phase'):
        window._phase_times()


def test_interleaved_iteration_uses_one_complete_global_timer(monkeypatch):
    import morphmimic.train.trainer as trainer_module
    import morphmimic.train.single_gpu as executor_module
    events=[]
    class Timer(_PhaseTimer):
        def __init__(self, cuda, **kwargs):
            super().__init__(False, **kwargs)
        def mark(self):
            self.marks.append(float(len(events))); events.append('mark')
    monkeypatch.setattr(trainer_module, '_PhaseTimer', Timer)
    def phases(index, *, update_targets=True, phases=None):
        assert phases is None and update_targets is False
        events.append(('learning', index))
        yield index
        return ({'body': index}, True)
    def equal(generators, accumulators, on_step):
        results=[]
        for g in generators:
            next(g)
            try: next(g)
            except StopIteration as e: results.append(e.value)
        on_step('wm')
        return results
    monkeypatch.setattr(executor_module, 'run_equal_phases', equal)
    executor=object.__new__(SingleGpuMultiRobotExecutor)
    executor.cfg=SimpleNamespace(lr_decay_iters=100,lr_final_frac=1.)
    executor.trainers=[SimpleNamespace(it=0,_collect_steps=lambda: 2,
        learning_phases=lambda i=i,**kw: phases(i,**kw), _actor_lr=SimpleNamespace(factor=lambda _:1.)) for i in range(3)]
    executor.owner._base_lr=[]
    executor.owner._update_target=lambda:events.append('target')
    executor.owner.policy_avg=None
    executor.accumulators={};executor.stage_updates={}
    executor.collect=lambda n: events.append(('collect',n)) or [None]*3
    result=executor.iteration()
    assert events == ['mark',('collect',2),'mark',('learning',0),('learning',1),('learning',2),'target','mark']
    assert executor.stage_updates == {'wm':1}
    assert [t.it for t in executor.trainers] == [1]*3
    assert len({id(out['_phases']) for out in result}) == 1
    assert result[0]['_phases'].names == ('global_collect','global_learning')
    assert len(result[0]['_phases'].seconds()) == 2


def test_console_global_interval_label_does_not_claim_body_phase_times(capsys):
    dummy=SimpleNamespace(it=1,cfg=SimpleNamespace(policy_goal='motion',kp_interface=False))
    metrics=defaultdict(float,{'time/global_collect':2.,'time/global_learning':5.})
    Trainer._print_line(dummy,metrics)
    output=capsys.readouterr().out
    assert 'global collect/learning' in output
    assert 'collect/wm/actor-critic' not in output
    assert not {'time/collect','time/wm','time/actor_critic'} & metrics.keys()
