"""Transport tests use real local aiortc peers; the simulator is never imported."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("aiohttp", "aiortc", "av", "numpy"))
PORT_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")
PUBLIC_PORTS = PORT_RANGE.is_file() and PORT_RANGE.read_text().split() == ["31870", "31889"]
transport = None
if AVAILABLE:
    import numpy as np
    from aiohttp.test_utils import TestClient, TestServer
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from aioice import stun

    spec = importlib.util.spec_from_file_location("preview_webrtc", Path(__file__).resolve().parents[1] / "preview" / "webrtc.py")
    transport = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = transport
    spec.loader.exec_module(transport)


class FakePreview:
    def __init__(self):
        self.frame_id, self.rgb, self.steps = 0, None, 23
        self.commands = []
        self.catalog = SimpleNamespace(search=self.search)

    def latest_frame(self):
        return self.frame_id, self.rgb

    def snapshot(self):
        return dict(steps=self.steps, frame_id=self.frame_id, frame_available=self.rgb is not None)

    def control(self, data):
        if not isinstance(data, dict) or data != {"action": "pause"}:
            raise ValueError("Invalid test control")
        self.commands.append(data)

    def search(self, robot, query, release=None):
        if robot not in ("q1", "all") or len(query) > 256:
            raise ValueError("Invalid robot or query")
        return dict(robot=robot, motions=["dance"] if query.casefold() in "dance" else [], total=1)


@unittest.skipUnless(AVAILABLE, "requires preview aiohttp/aiortc/PyAV dependencies")
class TrackTests(unittest.IsolatedAsyncioTestCase):
    async def test_latest_frame_placeholder_and_monotonic_timestamps(self):
        preview = FakePreview()
        track = transport.LatestFrameTrack(preview)
        try:
            idle = await track.recv()
            self.assertEqual((idle.width, idle.height), (960, 640))
            preview.frame_id, preview.rgb = 1, np.full((48, 64, 3), (220, 40, 10), dtype=np.uint8)
            frame = await track.recv()
            np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), preview.rgb)
            self.assertGreater(frame.pts, idle.pts)
            preview.frame_id, preview.rgb = 2, None
            loading = await track.recv()
            self.assertEqual((loading.width, loading.height), (960, 640))
            self.assertGreater(loading.pts, frame.pts)
            self.assertEqual(preview.steps, 23)
        finally:
            track.stop()

    async def test_frame_wraps_immutable_rgb_and_keeps_it_alive_after_replacement(self):
        import gc,weakref
        preview=FakePreview()
        rgb=np.full((48,64,3),(50,120,220),dtype=np.uint8);rgb.setflags(write=False)
        owner=weakref.ref(rgb);pointer=rgb.ctypes.data
        preview.frame_id,preview.rgb=1,rgb
        track=transport.LatestFrameTrack(preview)
        try:
            frame=await track.recv()
            self.assertEqual(frame.planes[0].buffer_ptr,pointer,'published RGB must not be copied for each viewer/frame')
            preview.rgb=None;del rgb;gc.collect()
            self.assertIsNotNone(owner(),'AVFrame must retain the source array')
            np.testing.assert_array_equal(frame.to_ndarray(format='rgb24')[0,0],[50,120,220])
        finally:track.stop()

    async def test_two_tracks_do_not_advance_physics_or_replay_a_frame_queue(self):
        preview = FakePreview()
        tracks = [transport.LatestFrameTrack(preview), transport.LatestFrameTrack(preview)]
        try:
            await asyncio.gather(*(track.recv() for track in tracks))
            for frame_id in range(1, 101):
                preview.frame_id, preview.rgb = frame_id, np.full((16, 16, 3), frame_id, dtype=np.uint8)
            frames = await asyncio.gather(*(track.recv() for track in tracks))
            for frame in frames:
                self.assertEqual(int(frame.to_ndarray(format="rgb24")[0, 0, 0]), 100)
            self.assertEqual(preview.steps, 23)
        finally:
            for track in tracks:
                track.stop()

    async def test_unchanged_source_waits_for_newest_frame_and_keeps_paused_video_alive(self):
        preview=FakePreview()
        preview.frame_id,preview.rgb=1,np.zeros((16,16,3),dtype=np.uint8)
        track=transport.LatestFrameTrack(preview)
        try:
            first=await track.recv()
            pending=asyncio.create_task(track.recv())
            await asyncio.sleep(.12)
            self.assertFalse(pending.done(),'unchanged RGB must not be encoded at30fps')
            preview.frame_id,preview.rgb=7,np.full((16,16,3),77,dtype=np.uint8)
            latest=await asyncio.wait_for(pending,.15)
            self.assertEqual(int(latest.to_ndarray(format='rgb24')[0,0,0]),77)
            self.assertGreater(latest.pts,first.pts)
            keepalive=await asyncio.wait_for(track.recv(),1.5)
            self.assertGreater(keepalive.pts,latest.pts)
            self.assertEqual(preview.steps,23)
        finally:track.stop()

    async def test_slow_receiver_does_not_burst_catch_up_frames(self):
        track = transport.LatestFrameTrack(FakePreview())
        try:
            await track.recv()
            await asyncio.sleep(.12)
            await track.recv()
            start = asyncio.get_running_loop().time()
            await track.recv()
            self.assertGreaterEqual(asyncio.get_running_loop().time() - start, .02)
        finally:
            track.stop()

    def test_public_candidates_require_mapped_udp_ports_and_hide_private_addresses(self):
        sdp = "v=0\r\nm=video 31870 UDP/TLS/RTP/SAVPF 97\r\nc=IN IP4 172.18.0.3\r\na=candidate:abc 1 udp 2130706431 172.18.0.3 31870 typ host\r\n"
        published = transport.public_sdp(sdp, "1.94.0.106")
        self.assertNotIn("172.18.0.3", published)
        self.assertIn("1.94.0.106 31870 typ host", published)
        for port in (31869, 31890):
            with self.assertRaisesRegex(ValueError, "31870"):
                transport.public_sdp(sdp.replace("31870", str(port)), "1.94.0.106")
        with self.assertRaises(ValueError):
            transport.public_sdp(sdp.replace(" udp ", " tcp "), "1.94.0.106")
        with self.assertRaises(ValueError):
            transport.public_sdp(sdp.replace("typ host", "typ srflx"), "1.94.0.106")
        with self.assertRaises(ValueError):
            transport.create_app(FakePreview(), "1.94.0.106\r\na=bad")


@unittest.skipUnless(AVAILABLE, "requires preview aiohttp/aiortc/PyAV dependencies")
class WebRTCTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.preview = FakePreview()
        self.app = transport.create_app(self.preview)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.receivers = []
        self.loop_errors = []
        loop = asyncio.get_running_loop()
        def record_error(loop, context):
            self.loop_errors.append(context)
            loop.default_exception_handler(context)
        loop.set_exception_handler(record_error)

    async def asyncTearDown(self):
        await asyncio.gather(*(peer.close() for peer in self.receivers))
        await self.client.close()
        await asyncio.sleep(0)
        self.assertFalse(self.loop_errors, "WebRTC emitted asynchronous callback/task errors")

    async def test_status_remains_available_after_sender_releases_its_track(self):
        class ClosingPeer:
            connectionState = iceConnectionState = "closed"
            def getSenders(self):
                return [SimpleNamespace(track=None)]
            async def close(self):
                pass
        self.app["pcs"].add(ClosingPeer())
        response = await self.client.get('/api/status')
        self.assertEqual(response.status, 200)
        data = await response.json()
        self.assertEqual(data['steps'], 23)
        self.assertEqual(data['webrtc']['connections'][0]['frames_sent'], 0)
        self.assertEqual(data['webrtc']['connections'][0]['source_frame_id'], 0)

    async def offer(self, connect=True):
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.receivers.append(peer)
        incoming = asyncio.Queue()
        peer.on("track", incoming.put_nowait)
        peer.addTransceiver("video", direction="recvonly")
        await peer.setLocalDescription(await peer.createOffer())
        response = await self.client.post("/api/offer", json=dict(sdp=peer.localDescription.sdp, type="offer"))
        data = await response.json()
        self.assertEqual(response.status, 200, data)
        self.assertEqual(data["type"], "answer")
        if connect:
            await peer.setRemoteDescription(RTCSessionDescription(**data))
            return peer, await asyncio.wait_for(incoming.get(), 5)
        return peer, data

    async def test_two_real_receivers_decode_latest_rgb_switch_and_idle_without_physics(self):
        self.preview.frame_id = 7
        self.preview.rgb = np.full((48, 64, 3), (220, 40, 10), dtype=np.uint8)
        first, second = await self.offer(), await self.offer()
        for _, track in (first, second):
            frames = [await asyncio.wait_for(track.recv(), 5) for _ in range(3)]
            self.assertTrue(all(a.pts < b.pts for a, b in zip(frames, frames[1:])))
            rgb = frames[-1].to_ndarray(format="rgb24")
            self.assertEqual(rgb.shape, (48, 64, 3))
            self.assertGreater(float(rgb[:, :, 0].mean()), 180)
        self.preview.frame_id, self.preview.rgb = 8, None
        for _, track in (first, second):
            for _ in range(20):
                frame = await asyncio.wait_for(track.recv(), 5)
                if frame.width == 960:
                    break
            self.assertEqual(frame.width, 960)
        self.preview.frame_id, self.preview.rgb = 9, np.full((48, 64, 3), (10, 220, 40), dtype=np.uint8)
        for _, track in (first, second):
            for _ in range(20):
                frame = await asyncio.wait_for(track.recv(), 5)
                if frame.width == 64 and frame.to_ndarray(format="rgb24")[:, :, 1].mean() > 180:
                    break
            self.assertEqual(frame.width, 64)
            self.assertGreater(float(frame.to_ndarray(format="rgb24")[:, :, 1].mean()), 180)
        self.assertEqual(self.preview.steps, 23)
        self.assertEqual(len(self.app["pcs"]), 2)

    async def test_lossless_diagnostic_frame_preserves_source_pixels(self):
        self.preview.rgb = np.full((48,64,3),(110,70,220),dtype=np.uint8)
        response=await self.client.get('/api/frame?format=png')
        self.assertEqual(response.status,200)
        self.assertEqual(response.content_type,'image/png')
        import io
        from PIL import Image
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(await response.read()))),self.preview.rgb)

    async def test_routes_diagnostic_frame_and_bounded_json_controls(self):
        response = await self.client.get("/api/status")
        status = await response.json()
        self.assertEqual(status["steps"], 23)
        self.assertEqual(status["webrtc"]["peers"], 0)
        response = await self.client.get("/api/motions?robot=all&q=DAN")
        self.assertEqual((await response.json())["motions"], ["dance"])
        for path, status in (("/", 200), ("/app.js", 200), ("/style.css", 200),
                             ("/api/frame", 409), ("/api/motions?robot=human", 400),
                             ("/server.py", 404)):
            response = await self.client.get(path)
            self.assertEqual(response.status, status, path)
        self.preview.rgb = np.full((48, 64, 3), 150, dtype=np.uint8)
        response = await self.client.get("/api/frame")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "image/jpeg")
        self.assertTrue((await response.read()).startswith(b"\xff\xd8"))
        response = await self.client.post("/api/control", json={"action": "pause"})
        self.assertEqual(response.status, 202)
        self.assertEqual(self.preview.commands, [{"action": "pause"}])
        for kwargs in ({"json": []}, {"data": b"{", "headers": {"Content-Type": "application/json"}},
                       {"data": b"{}"}, {"json": {"action": "pause", "padding": "x" * 4096}},
                       {"json": {"action": "pause"}, "headers": {"Origin": "https://other.example"}},
                       {"json": {"action": "pause"}, "headers": {"Origin": "null"}}):
            response = await self.client.post("/api/control", **kwargs)
            self.assertIn(response.status, (400, 403, 413, 415))

    async def test_bad_offers_are_rejected_without_leaking_peers(self):
        for data in (None, [], {}, {"type": "answer", "sdp": "v=0\r\n"},
                     {"type": "offer", "sdp": 3}, {"type": "offer", "sdp": "invalid"},
                     {"type": "offer", "sdp": "x" * (128 * 1024 + 1)},
                     {"type": "offer", "sdp": "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 97\r\na=recvonly\r\n"}):
            response = await self.client.post("/api/offer", data=json.dumps(data), headers={"Content-Type": "application/json"})
            self.assertIn(response.status, (400, 413))
            self.assertEqual(len(self.app["pcs"]), 0)
        response = await self.client.post("/api/offer", json={"type": "offer", "sdp": "v=0\r\n"}, headers={"Origin": "https://other.example"})
        self.assertEqual(response.status, 403)

    async def test_send_enabled_offer_cannot_allocate_an_incoming_video_queue(self):
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.receivers.append(peer)
        peer.addTransceiver("video", direction="recvonly")
        await peer.setLocalDescription(await peer.createOffer())
        # The SDP parser uses the final direction attribute, even if recvonly appeared earlier.
        response = await self.client.post("/api/offer", json={"type": "offer", "sdp": peer.localDescription.sdp + "a=sendrecv\r\n"})
        self.assertEqual(response.status, 400)
        self.assertFalse(self.app["pcs"])

    async def test_four_peer_limit_and_shutdown_closes_tracks(self):
        for _ in range(4):
            await self.offer(connect=False)
        self.assertEqual(len(self.app["pcs"]), 4)
        peer = self.receivers[0]
        response = await self.client.post("/api/offer", json={"type": "offer", "sdp": peer.localDescription.sdp})
        self.assertEqual(response.status, 503)
        peers = list(self.app["pcs"])
        tracks = [pc.getSenders()[0].track for pc in peers]
        await self.client.close()
        self.assertFalse(self.app["pcs"])
        self.assertFalse(self.app["peer_tasks"])
        self.assertTrue(all(pc.connectionState == "closed" for pc in peers))
        self.assertTrue(all(track.readyState == "ended" for track in tracks))

    async def test_shutdown_joins_peer_close_already_in_progress(self):
        entered, release = asyncio.Event(), asyncio.Event()
        class SlowPeer:
            remoteDescription = None
            calls = 0
            def getSenders(self): return []
            async def close(self):
                self.calls += 1
                entered.set()
                await release.wait()
        peer = SlowPeer()
        self.app['pcs'].add(peer)
        first = asyncio.create_task(self.app.on_shutdown[0](self.app))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            self.assertFalse(self.app['pcs'])
            second = asyncio.create_task(self.app.on_shutdown[0](self.app))
            await asyncio.sleep(.02)
            self.assertFalse(second.done(), 'shutdown returned while peer close was still running')
        finally:
            release.set()
            await asyncio.gather(first, *( [second] if second else [] ))
        self.assertEqual(peer.calls, 1)

    async def test_shutdown_reports_inflight_peer_close_error(self):
        entered, release = asyncio.Event(), asyncio.Event()
        class FailingPeer:
            remoteDescription = None
            def getSenders(self): return []
            async def close(self):
                entered.set()
                await release.wait()
                raise RuntimeError('peer close failed')
        self.app['pcs'].add(FailingPeer())
        first = asyncio.create_task(self.app.on_shutdown[0](self.app))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            second = asyncio.create_task(self.app.on_shutdown[0](self.app))
            await asyncio.sleep(.02)
        finally:
            release.set()
        errors = await asyncio.gather(first, second, return_exceptions=True)
        self.assertTrue(all(isinstance(error, RuntimeError) and str(error) == 'peer close failed'
                            for error in errors), errors)

    async def assert_ice_tasks_drained(self, peer):
        await asyncio.sleep(.03)
        connections = {t.receiver.transport.transport._connection for t in peer.getTransceivers()}
        checks = [pair.task for connection in connections for pair in connection._check_list
                  if pair.task is not None]
        for task in checks:
            self.assertTrue(task.done(), "Closed server peer retained an ICE candidate task")
            if not task.cancelled():
                self.assertIsNone(task.exception(), "Closed server peer's ICE check failed after transport shutdown")
        pending = [task for task in asyncio.all_tasks()
                   if task.get_coro().__qualname__.endswith("__connect")
                   and task.get_coro().cr_frame is not None
                   and task.get_coro().cr_frame.f_locals.get("self") is peer]
        self.assertFalse(pending, "Closed server peer retained its ICE negotiation task")

    async def test_shutdown_drains_pending_ice_candidate_tasks(self):
        await self.offer(connect=False)
        peer = next(iter(self.app['pcs']))
        await self.client.close()
        await self.assert_ice_tasks_drained(peer)

    async def test_timed_out_peer_drains_pending_ice_candidate_tasks(self):
        with patch.object(transport, 'HANDSHAKE_TIMEOUT', .5):
            await self.offer(connect=False)
            peer = next(iter(self.app['pcs']))
            for _ in range(100):
                if peer.connectionState == 'closed':
                    break
                await asyncio.sleep(.02)
        self.assertEqual(peer.connectionState, 'closed')
        await self.assert_ice_tasks_drained(peer)

    async def test_peer_close_drains_candidates_and_does_not_restart_negotiation(self):
        await self.offer(connect=False)
        peer = next(iter(self.app['pcs']))
        await peer.close()
        await peer.close()
        await peer._RTCPeerConnection__connect()
        await self.assert_ice_tasks_drained(peer)

    async def test_private_ice_cleanup_rejects_unqualified_dependency_versions(self):
        with patch.object(transport, 'version', return_value='unqualified'):
            with self.assertRaisesRegex(RuntimeError, 'Preview ICE cleanup requires'):
                transport.PreviewPeerConnection(RTCConfiguration(iceServers=[]))

    async def test_incomplete_handshake_expires_and_closed_peer_is_removed(self):
        # Host-network CI can enumerate thousands of candidate pairs before yielding.
        with patch.object(transport, "HANDSHAKE_TIMEOUT", .5):
            await self.offer(connect=False)
            for _ in range(50):
                if not self.app["pcs"]:
                    break
                await asyncio.sleep(.02)
        self.assertFalse(self.app["pcs"])
        self.assertFalse(self.app["peer_tasks"])
        peer, track = await self.offer()
        await asyncio.wait_for(track.recv(), 5)
        server_peer = next(iter(self.app["pcs"]))
        await peer.close()
        for _ in range(50):
            if not self.app["pcs"]:
                break
            await asyncio.sleep(.02)
        self.assertFalse(self.app["pcs"])
        self.assertEqual(server_peer.connectionState, "closed")


@unittest.skipUnless(AVAILABLE, "requires preview aiohttp/aiortc/PyAV dependencies")
class PublicIceTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(PUBLIC_PORTS, "requires isolated container UDP ephemeral range 31870-31889")
    async def test_public_unreachable_peer_timeout_releases_pending_ice_task(self):
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        peer.addTransceiver("video", direction="recvonly")
        await peer.setLocalDescription(await peer.createOffer())
        offered = transport.SessionDescription.parse(peer.localDescription.sdp)
        app = transport.create_app(FakePreview(), public_host=offered.media[0].ice_candidates[0].ip)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            with patch.object(transport, "HANDSHAKE_TIMEOUT", .1):
                response = await client.post("/api/offer", json={"type": "offer", "sdp": peer.localDescription.sdp})
                self.assertEqual(response.status, 200)
                await response.json()
                server_peer = next(iter(app["pcs"]))
                await asyncio.sleep(.15)
            self.assertFalse(app["pcs"])
            self.assertFalse(app["peer_tasks"])
            self.assertEqual(server_peer.connectionState, "closed")
            pending = [task for task in asyncio.all_tasks()
                       if "RTCPeerConnection.__connect" in task.get_coro().__qualname__]
            self.assertFalse(pending, "Timed-out public peer left an ICE connection task running")
        finally:
            await peer.close()
            await client.close()

    @unittest.skipUnless(PUBLIC_PORTS, "requires isolated container UDP ephemeral range 31870-31889")
    async def test_public_receive_initiated_late_peer_reflexive_check_decodes_video(self):
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        peer.addTransceiver("video", direction="recvonly")
        incoming = asyncio.Queue()
        peer.on("track", incoming.put_nowait)
        await peer.setLocalDescription(await peer.createOffer())
        offered = transport.SessionDescription.parse(peer.localDescription.sdp)
        candidate = offered.media[0].ice_candidates[0]
        rejected = asyncio.Event()

        class RejectInitialRoute(asyncio.DatagramProtocol):
            def connection_made(self, udp):
                self.udp = udp

            def datagram_received(self, data, addr):
                request = stun.parse_message(data)
                if request.message_class == stun.Class.REQUEST:
                    rejected.set()
                    response = stun.Message(message_method=stun.Method.BINDING, message_class=stun.Class.ERROR,
                                            transaction_id=request.transaction_id)
                    response.attributes["ERROR-CODE"] = (400, "Initial route rejected")
                    response.add_message_integrity(offered.media[0].ice.password.encode())
                    self.udp.sendto(bytes(response), addr)

        udp, _ = await asyncio.get_running_loop().create_datagram_endpoint(RejectInitialRoute,
                                                                        local_addr=(candidate.ip, 0))
        candidate.port = udp.get_extra_info("sockname")[1]
        client = TestClient(TestServer(transport.create_app(FakePreview(), public_host=candidate.ip)))
        await client.start_server()
        try:
            response = await client.post("/api/offer", json={"type": "offer", "sdp": str(offered)})
            self.assertEqual(response.status, 200)
            answer = await response.json()
            # A delayed browser check must still arrive before the existing 15 s watchdog.
            await asyncio.sleep(.1)
            self.assertEqual(len(client.server.app["pcs"]), 1, "ICE failed before the delayed valid peer check")
            server_peer = next(iter(client.server.app["pcs"]))
            accepted = transport.SessionDescription.parse(server_peer.remoteDescription.sdp).media[0]
            self.assertTrue(accepted.ice.password == offered.media[0].ice.password, "ICE credentials changed")
            self.assertTrue(accepted.ice.usernameFragment == offered.media[0].ice.usernameFragment, "ICE username changed")
            self.assertEqual(accepted.dtls.fingerprints, offered.media[0].dtls.fingerprints)
            self.assertEqual(accepted.ice_candidates, [])
            self.assertFalse(accepted.ice_candidates_complete)
            self.assertFalse(rejected.is_set(), "Public server probed the unusable offered route")
            await peer.setRemoteDescription(RTCSessionDescription(**answer))
            track = await asyncio.wait_for(incoming.get(), 5)
            frame = await asyncio.wait_for(track.recv(), 5)
            self.assertEqual((frame.width, frame.height), (960, 640))
            self.assertEqual(server_peer.connectionState, "connected")
            self.assertEqual(server_peer.iceConnectionState, "completed")
            accepted = transport.SessionDescription.parse(server_peer.remoteDescription.sdp).media[0]
            self.assertTrue(accepted.ice_candidates_complete)
        finally:
            udp.close()
            await peer.close()
            await client.close()


if __name__ == "__main__":
    unittest.main()

@unittest.skipUnless(AVAILABLE and os.environ.get('BLINKMORPH_TEST_NVENC')=='1', 'requires explicit NVENC hardware test environment')
class NvencTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.preview=FakePreview()
        self.preview.frame_id=1
        self.preview.rgb=np.full((1080,1920,3),(50,150,80),dtype=np.uint8)
        import torch
        if torch.cuda.device_count()<2:self.skipTest('requires a separate encoding GPU')
        self.app=transport.create_app(self.preview,nvenc=True,encoder_gpu=1)
        self.client=TestClient(TestServer(self.app));await self.client.start_server()
        self.peer=RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.incoming=asyncio.Queue();self.peer.on('track',self.incoming.put_nowait)
        self.peer.addTransceiver('video',direction='recvonly')

    async def asyncTearDown(self):
        await self.peer.close();await self.client.close()

    async def test_negotiated_h264_uses_nvenc_and_actual_1080p_reception(self):
        await self.peer.setLocalDescription(await self.peer.createOffer())
        offer=transport.SessionDescription.parse(self.peer.localDescription.sdp)
        for codec in offer.media[0].rtp.codecs:
            if codec.mimeType.lower()=='video/h264':
                codec.parameters['max-recv-level']=codec.parameters['profile-level-id'][2:4]+'28'
        response=await self.client.post('/api/offer',json={'type':'offer','sdp':str(offer)})
        data=await response.json();self.assertEqual(response.status,200,data)
        await self.peer.setRemoteDescription(RTCSessionDescription(**data))
        track=await asyncio.wait_for(self.incoming.get(),5)
        frame=await asyncio.wait_for(track.recv(),5)
        self.assertEqual((frame.width,frame.height),(1920,1080))
        status=await (await self.client.get('/api/status')).json()
        self.assertTrue(status['webrtc']['nvenc'])
        self.assertEqual(status['webrtc']['encoder_gpu'],1)
        self.assertEqual(status['webrtc']['connections'][0]['encoder'],'h264_nvenc')
        self.assertGreater(status['webrtc']['connections'][0]['encoder_frames'],0)

    async def test_vp8_only_offer_cannot_silently_choose_software(self):
        await self.peer.setLocalDescription(await self.peer.createOffer())
        offer=transport.SessionDescription.parse(self.peer.localDescription.sdp)
        offer.media[0].rtp.codecs=[c for c in offer.media[0].rtp.codecs if c.mimeType.lower()=='video/vp8']
        response=await self.client.post('/api/offer',json={'type':'offer','sdp':str(offer)})
        self.assertEqual(response.status,400,await response.json())
        self.assertFalse(self.app['pcs'])
