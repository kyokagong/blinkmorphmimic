"""Evaluation reports identify the immutable checkpoint actually evaluated."""
import pytest
import torch


def test_evaluation_captures_fixed_checkpoint_hash_and_weight_selection(tmp_path):
    import hashlib
    from morphmimic.scripts.eval import checkpoint_identity, verify_checkpoint
    path = tmp_path / 'ckpt_8.pt'
    torch.save({'policy': {'weight': torch.tensor([1.])}, 'ema_policy': {}}, path)
    identity = checkpoint_identity(path)
    assert identity == dict(checkpoint=str(path),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), weights='ema')
    verify_checkpoint(identity)
    torch.save({'policy': {}}, path)
    assert checkpoint_identity(path)['weights'] == 'online'


@pytest.mark.parametrize('name', ['latest.pt', 'last.pt', 'ckpt_bad.pt'])
def test_evaluation_rejects_mutable_checkpoint_names(tmp_path, name):
    from morphmimic.scripts.eval import checkpoint_identity
    path = tmp_path / name
    torch.save({'policy': {}}, path)
    with pytest.raises(ValueError, match='fixed'):
        checkpoint_identity(path)


@pytest.mark.parametrize('change', ['modify', 'replace', 'symlink'])
def test_evaluation_refuses_report_when_pinned_checkpoint_changes(tmp_path, change):
    from morphmimic.scripts.eval import checkpoint_identity, verify_checkpoint
    path = tmp_path / 'ckpt_8.pt'
    torch.save({'policy': {'weight': torch.tensor([1.])}}, path)
    identity = checkpoint_identity(path)
    if change == 'modify':
        torch.save({'policy': {'weight': torch.tensor([2.])}}, path)
    else:
        replacement = tmp_path / 'ckpt_9.pt'
        torch.save({'policy': {'weight': torch.tensor([2.])}}, replacement)
        if change == 'replace':
            replacement.replace(path)
        else:
            path.unlink()
            path.symlink_to(replacement)
    with pytest.raises(ValueError, match='changed'):
        verify_checkpoint(identity)


def test_evaluation_rejects_checkpoint_symlinks_at_start(tmp_path):
    from morphmimic.scripts.eval import checkpoint_identity
    path = tmp_path / 'ckpt_8.pt'
    torch.save({'policy': {}}, path)
    alias = tmp_path / 'ckpt_9.pt'
    alias.symlink_to(path)
    with pytest.raises(ValueError, match='fixed'):
        checkpoint_identity(alias)


def test_evaluation_cli_writes_no_report_if_checkpoint_changes_during_rollout(tmp_path, monkeypatch):
    from morphmimic.scripts import eval as cli
    path, output = tmp_path / 'ckpt_8.pt', tmp_path / 'results.csv'
    torch.save({'policy': {}}, path)
    args = cli.EvalArgs(ckpt=str(path), subset_yaml=str(tmp_path / 'val.yaml'),
        out=str(output), manifest=str(tmp_path / 'manifest.csv'))
    monkeypatch.setattr(cli.tyro, 'cli', lambda _: args)
    monkeypatch.setattr(cli, 'select_gpu', lambda *_: None)
    monkeypatch.setattr(cli, 'configure_torch_backends', lambda: None)
    monkeypatch.setattr(cli, 'summarize', lambda *_: {})
    monkeypatch.setattr(cli, 'format_summary', lambda _: '')
    def rollout(*_, **kwargs):
        replacement = tmp_path / 'ckpt_9.pt'
        torch.save({'policy': {'weight': torch.tensor([2.])}}, replacement)
        replacement.replace(path)
        result = cli.pd.DataFrame({'name': ['clip']})
        result.attrs.update(iteration=8, mode='motion')
        return result
    monkeypatch.setattr(cli, 'evaluate', rollout)
    with pytest.raises(ValueError, match='changed'):
        cli.main()
    assert not output.exists()
    assert not output.with_suffix('.json').exists()
