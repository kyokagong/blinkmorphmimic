import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from aiohttp.test_utils import TestClient, TestServer
from preview.pico import PicoRecordings, read_recording, SLOTS, ROLES
from preview.server import Catalog, Preview
from preview.webrtc import create_app


def recording():
    header = dict(type='header', schema='embodied.mocap.five-point-recording/v1',
                  coordinate_frame='pico_source_y_up', position_unit='m',
                  quaternion_order='wxyz', slots=list(SLOTS), roles=list(ROLES))
    frames = [dict(type='frame', elapsed_ns=str(i * 20000000),
                   positions_m=[[0, 1, 0], [-.1, 0, 0], [.1, 0, 0], [-.3, .6, 0], [.3, .6, 0]],
                   quaternions_wxyz=[[1, 0, 0, 0]] * 5) for i in range(3)]
    return [header, *frames, dict(type='end', complete=True, frame_count=3)]


def encode(rows):
    return '\n'.join(json.dumps(row) for row in rows).encode()


class PicoUploadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        templates = {}
        for robot in ('q1', 'bumi', 'g1'):
            motion = root / (robot + '.npz')
            human = root / (robot + '-human.npz')
            np.savez(motion, fps=np.array(50), joint_pos=np.zeros((1, 2)), joint_vel=np.zeros((1, 2)),
                     body_pos_w=np.zeros((1, 2, 3)), body_quat_w=np.tile([1., 0, 0, 0], (1, 2, 1)),
                     body_lin_vel_w=np.zeros((1, 2, 3)), body_ang_vel_w=np.zeros((1, 2, 3)))
            np.savez(human, human_quat_w=np.tile([1., 0, 0, 0], (1, 24, 1)))
            templates[robot] = dict(motion=motion, human=human)
        self.store = PicoRecordings(root / 'uploads', templates)
        self.preview = Preview(Catalog(root, dict.fromkeys(templates, object()), interface='human'))
        self.preview.recordings = self.store
        self.client = TestClient(TestServer(create_app(self.preview)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def upload(self, data, **headers):
        return await self.client.post('/api/recordings', data=data,
                                     headers={'Content-Type': 'application/x-ndjson', 'X-Filename': 'test.jsonl', **headers})

    async def test_upload_persists_deduplicates_and_loads_with_selected_model(self):
        raw = encode(recording())
        response = await self.upload(raw)
        self.assertEqual(response.status, 201, await response.text())
        meta = (await response.json())['recording']
        again = await self.upload(raw)
        self.assertEqual((await again.json())['recording']['id'], meta['id'])
        reopened = PicoRecordings(self.store.root, self.store.templates)
        self.assertEqual(reopened.list(), [meta])
        catalog = reopened.catalog(self.preview.catalog, meta['id'])
        self.assertEqual(set(catalog.select('all', 'pico')), {'q1', 'bumi', 'g1'})
        with np.load(catalog.root / 'human50/q1/pico.npz') as human:
            self.assertEqual(np.flatnonzero(human['human_joint_valid']).tolist(), list(SLOTS))
            self.assertEqual(human['human_pos_w'].shape, (3, 24, 3))
            np.testing.assert_allclose(human['human_pos_w'][:, 7:9, 2], .08, atol=1e-6)
        self.preview.control(dict(action='select', robot='all', motion='pico', recording=meta['id']))
        self.assertEqual(self.preview.requests.get_nowait()['human_points'], 'five')
        with self.assertRaises(ValueError):
            self.preview.control(dict(action='select', robot='q1', motion='pico', recording='../bad'))
        with self.assertRaisesRegex(ValueError, 'Human'):
            reopened.catalog(SimpleNamespace(interface='motion'), meta['id'])
        response = await self.client.get('/api/recordings')
        self.assertEqual((await response.json())['recordings'], [meta])
        (catalog.root / 'human50/q1/pico.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            catalog.select('q1', 'pico')

    async def test_invalid_files_are_rejected_without_publishing_partial_records(self):
        bad = []
        for mutate in (lambda r: r[-1].update(complete=False),
                       lambda r: r[-1].update(frame_count=4),
                       lambda r: r[2].update(elapsed_ns='0'),
                       lambda r: r[1].update(positions_m=[[float('nan'), 0, 0]] * 5),
                       lambda r: r[1].update(quaternions_wxyz=[[0, 0, 0, 0]] * 5),
                       lambda r: r[0].update(coordinate_frame='unknown')):
            rows = copy.deepcopy(recording()); mutate(rows); bad.append(encode(rows))
        bad.extend([b'', b'not json', b'[]\n{}\n{}\n{}'])
        for raw in bad:
            with self.subTest(raw=raw[:100]):
                response = await self.upload(raw)
                self.assertEqual(response.status, 400, await response.text())
                self.assertEqual(self.store.list(), [])
                self.assertEqual(list(self.store.root.iterdir()), [])
        response = await self.upload(encode(recording()), Origin='https://unrelated.example')
        self.assertEqual(response.status, 403)
        self.assertEqual(self.store.list(), [])


if __name__ == '__main__':
    unittest.main()
