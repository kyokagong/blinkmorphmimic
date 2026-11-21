"""Budgeted training-only probe sampling and reset-bin boundary contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

SRC = Path(__file__).parents[1] / 'src/morphmimic'


def load_file(relative, name):
    path = SRC / relative
    assert path.exists(), f'{relative} is not implemented'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def library():
    return SimpleNamespace(fps=50, clip_names=['a', 'b'], clip_len=torch.tensor([151, 101]),
                           clip_start=torch.tensor([0, 151]), num_bins=5,
                           bin_start=torch.tensor([0, 50, 100, 151, 201]),
                           bin_len=torch.tensor([50, 50, 51, 50, 51]),
                           bin_clip=torch.tensor([0, 0, 0, 1, 1]))


def document(failures):
    return {'schema_version': 1, 'robot': 'q1', 'interface': 'motion', 'fps': 50,
            'checkpoint_sha256': 'a' * 64, 'training_subset_sha256': 'b' * 64,
            'failures': [dict(robot='q1', source_id='seed/source', clip=clip, fail_frame=frame)
                         for clip, frame in failures]}


class ProbeFocusTests(unittest.TestCase):
    def setUp(self):
        self.focus = load_file('train/probe_focus.py', 'probe_focus')

    def test_repeated_failures_accumulate_without_crossing_clip_or_failure(self):
        result = self.focus.focus_probabilities(document([('a', 150), ('a', 150), ('b', 100)]), library(), 'q1', expected_subset_hash='b' * 64)
        torch.testing.assert_close(result, torch.tensor([.4, .4, 0, .2, 0]))

    def test_only_complete_bins_in_preceding_ten_seconds_are_focused(self):
        lib = SimpleNamespace(fps=50, clip_names=['long'], clip_len=torch.tensor([1201]),
                              clip_start=torch.tensor([0]), num_bins=24,
                              bin_start=torch.arange(0, 1200, 50),
                              bin_len=torch.tensor([50] * 23 + [51]), bin_clip=torch.zeros(24, dtype=torch.long))
        result = self.focus.focus_probabilities(document([('long', 1100)]), lib, 'q1')
        torch.testing.assert_close(result.nonzero().flatten(), torch.arange(12, 22))
        torch.testing.assert_close(result[12:22], torch.full((10,), .1))

    def test_no_failure_or_short_prefix_falls_back_exactly_to_base(self):
        base = torch.tensor([.1, .2, .3, .15, .25])
        for failures in ([], [('a', 0)], [('a', 10)]):
            focus = self.focus.focus_probabilities(document(failures), library(), 'q1')
            normalized = self.focus.normalize_probe_focus(focus, 5, .15)
            self.assertIsNone(normalized)
            torch.testing.assert_close(self.focus.mix_probe_focus(base, normalized, .15), base, rtol=0, atol=0)

    def test_robot_unknown_clip_invalid_frame_and_subset_mismatch_are_rejected(self):
        for doc, robot, subset_hash in [(document([('missing', 2)]), 'q1', None),
                                         (document([('a', 151)]), 'q1', None),
                                         (document([('a', -1)]), 'q1', None),
                                         (document([('a', 10)]), 'g1', None),
                                         (document([]), 'q1', 'c' * 64)]:
            with self.subTest(doc=doc, robot=robot):
                with self.assertRaises(ValueError):
                    self.focus.focus_probabilities(doc, library(), robot, expected_subset_hash=subset_hash)
        doc = document([]); doc['schema_version'] = 2
        with self.assertRaises(ValueError):
            self.focus.focus_probabilities(doc, library(), 'q1')

    def test_probabilities_and_share_are_validated_once_before_sampling(self):
        for p, rho in [(torch.ones(4), .15), (torch.tensor([1., -1, 0, 0, 0]), .15),
                       (torch.full((5,), float('nan')), .15), (torch.ones(5), 1.),
                       (torch.ones(5), -.1), (torch.ones(5), float('nan'))]:
            with self.assertRaises(ValueError):
                self.focus.normalize_probe_focus(p, 5, rho)
        focus = self.focus.normalize_probe_focus(torch.tensor([0., 0, 0, 3, 0]), 5, .15)
        torch.testing.assert_close(self.focus.mix_probe_focus(torch.full((5,), .2), focus, .15), torch.tensor([.17, .17, .17, .32, .17]))

    @unittest.skipUnless(importlib.util.find_spec('mjlab'), 'mjlab runtime required')
    def test_command_reset_sampling_uses_installed_local_probe_mixture(self):
        from morphmimic.data.motion_library import MotionLibrary
        from morphmimic.env.commands import MultiMotionCommand
        lib = MotionLibrary.__new__(MotionLibrary)
        lib.__dict__.update(library().__dict__)
        lib.device = torch.device('cpu')
        lib.frame_clip = torch.repeat_interleave(torch.arange(2), lib.clip_len)
        command = SimpleNamespace(library=lib, device='cpu', G_b=torch.zeros(5), past_softmax=torch.ones(5),
                                  cfg=SimpleNamespace(sampling='return', softmax_temp=25., mix=(1., 0., 0.),
                                                      softmax_cap=None, failure_reset=0.),
                                  metrics={'sampling_entropy': torch.zeros(10000)})
        MultiMotionCommand.set_probe_focus(command, torch.tensor([0., 0, 0, 1, 0]), .9)
        torch.manual_seed(7)
        frames, bins = MultiMotionCommand._sample_start(command, torch.arange(10000))
        frequency = torch.bincount(bins, minlength=5) / 10000
        torch.testing.assert_close(frequency, torch.tensor([.02, .02, .02, .92, .02]), atol=.01, rtol=0)
        self.assertTrue((frames >= 0).all())
        self.assertTrue((frames < 252).all())
        MultiMotionCommand.set_probe_focus(command, torch.zeros(5), .15)
        self.assertIsNone(command.probe_probabilities)
        self.assertEqual(command.probe_rho, 0.)

    @unittest.skipUnless(importlib.util.find_spec('yaml'), 'YAML runtime required')
    def test_probe_cli_rejects_validation_records_even_inside_train_yaml(self):
        cli = load_file('scripts/probe.py', 'probe_cli')
        with tempfile.TemporaryDirectory() as root:
            prepared = Path(root) / 'prepared'
            robot = prepared / 'q1'
            robot.mkdir(parents=True)
            yaml = robot / 'train.yaml'
            yaml.write_text('root: .\nclips: [a.npz]\n')
            (robot / 'a.npz').write_bytes(b'fixture')
            meta = {'complete': True, 'clips': [{'name': 'a', 'robot': 'q1', 'frames': 51,
                                               'source_id': 'seed/a', 'split': 'val'}]}
            (prepared / 'manifest.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                cli.training_clips(yaml, 'q1')
            meta['clips'][0]['split'] = 'train'
            (prepared / 'manifest.json').write_text(json.dumps(meta))
            clips, proof = cli.training_clips(yaml, 'q1')
            self.assertEqual(clips[0]['source_id'], 'seed/a')
            self.assertEqual(clips[0]['path'], str(robot / 'a.npz'))
            self.assertEqual(len(proof['training_subset_sha256']), 64)
            wrong = robot / 'all.yaml'; wrong.write_text(yaml.read_text())
            with self.assertRaises(ValueError):
                cli.training_clips(wrong, 'q1')
            with self.assertRaises(ValueError):
                cli.training_clips(yaml, 'g1')

    def test_probe_budget_checks_actual_motion_length_instead_of_trusting_metadata(self):
        import numpy as np
        cli = load_file('scripts/probe.py', 'probe_cli')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'clip.npz'
            np.savez(path, fps=50, joint_pos=np.zeros((51, 22)))
            cli.validate_probe_clips([{'path': str(path), 'frames': 51}])
            with self.assertRaises(ValueError):
                cli.validate_probe_clips([{'path': str(path), 'frames': 50}])
            np.savez(path, fps=30, joint_pos=np.zeros((51, 22)))
            with self.assertRaises(ValueError):
                cli.validate_probe_clips([{'path': str(path), 'frames': 51}])

    def test_probe_cli_requires_fixed_checkpoint_and_matching_robot(self):
        cli = load_file('scripts/probe.py', 'probe_cli')
        with tempfile.TemporaryDirectory() as root:
            fixed = Path(root) / 'ckpt_8.pt'
            torch.save({'cfg': {'robot': 'q1'}}, fixed)
            self.assertEqual(cli.checkpoint_identity(fixed, 'q1')['checkpoint_sha256'], cli.file_hash(fixed))
            with self.assertRaises(ValueError):
                cli.checkpoint_identity(fixed, 'g1')
            latest = Path(root) / 'latest.pt'
            torch.save({'cfg': {'robot': 'q1'}}, latest)
            with self.assertRaises(ValueError):
                cli.checkpoint_identity(latest, 'q1')

    def test_probe_rejects_checkpoint_replaced_between_load_and_identity_hash(self):
        from unittest.mock import patch
        cli = load_file('scripts/probe.py', 'probe_cli')
        with tempfile.TemporaryDirectory() as root:
            fixed, replacement = Path(root) / 'ckpt_8.pt', Path(root) / 'ckpt_9.pt'
            torch.save({'cfg': {'robot': 'q1', 'goal_anchor': 'reference'}}, fixed)
            torch.save({'cfg': {'robot': 'q1', 'goal_anchor': 'robot'}}, replacement)
            original_hash = cli.file_hash(fixed)
            self.assertNotEqual(original_hash, cli.file_hash(replacement))
            load = torch.load
            def load_then_replace(*args, **kwargs):
                checkpoint = load(*args, **kwargs)
                replacement.replace(fixed)
                return checkpoint
            with patch('torch.load', side_effect=load_then_replace):
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    cli.checkpoint_identity(fixed, 'q1')

    def test_whole_clip_budget_never_exceeds_frames_or_256_clips(self):
        selected = self.focus.select_probe_clips([{'name': 'long', 'frames': 1000}, {'name': 'a', 'frames': 101}, {'name': 'b', 'frames': 51}], max_clips=2, max_frames=150, seed=0)
        self.assertEqual({x['name'] for x in selected}, {'a', 'b'})
        self.assertEqual(sum(x['frames'] - 1 for x in selected), 150)
        with self.assertRaises(ValueError):
            self.focus.select_probe_clips([{'name': 'a', 'frames': 2}], max_clips=257, max_frames=10, seed=0)


if __name__ == '__main__':
    unittest.main()

class CombinedFocusTests(unittest.TestCase):
    def test_interface_weights_control_failure_bin_contribution(self):
        focus=load_file('train/probe_focus.py','combined_focus')
        doc=document([('a',150),('b',100)])
        doc.update(interface='combined',interface_weights={'motion':.5,'human-five':.25,'human-full':.25})
        doc['failures'][0].update(interface='motion',weight=.5)
        doc['failures'][1].update(interface='human-five',weight=.25)
        torch.testing.assert_close(focus.focus_probabilities(doc,library(),'q1'),torch.tensor([.4,.4,0,.2,0]))
        for bad in (-1,float('nan')):
            doc['failures'][0]['weight']=bad
            with self.assertRaises(ValueError):focus.focus_probabilities(doc,library(),'q1')
        doc['failures'][0].update(interface='unknown',weight=.5)
        with self.assertRaises(ValueError):focus.focus_probabilities(doc,library(),'q1')
