import pytest
from scripts.plot_optimized import training_checkpoint_rows


def test_selected_checkpoints_stop_at_requested_bound_and_keep_missing():
    train={"q1":{"tags":{"loss":[{"iteration":5000,"status":"ok","value":.25}]}}}
    rows=training_checkpoint_rows(train,iterations=[0,1000,2000,3000,4000,5000])
    assert len(rows)==18
    assert {r["iteration"] for r in rows}==set(range(0,5001,1000))
    assert next(r for r in rows if r["robot"]=="q1" and r["iteration"]==5000)["value"]==.25
    assert next(r for r in rows if r["robot"]=="bumi" and r["iteration"]==5000)["status"]=="missing"


def test_default_5000_and_explicit_historical_range_and_invalid_selection_rejected():
    train={"q1":{"tags":{"loss":[]}}}
    assert {r["iteration"] for r in training_checkpoint_rows(train)}==set(range(0,5001,1000))
    historical=list(range(0,10001,1000))
    assert {r["iteration"] for r in training_checkpoint_rows(train,iterations=historical)}==set(historical)
    for bad in ([0,0],[500],[],[11000]):
        with pytest.raises(ValueError):training_checkpoint_rows(train,iterations=bad)
