"""HTTP signaling and independent latest-RGB video tracks for the shared preview."""
from __future__ import annotations

import asyncio
from fractions import Fraction
from ipaddress import ip_address
from importlib.metadata import version
import io
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from aiohttp import web
from aiortc import RTCBundlePolicy, RTCConfiguration, RTCPeerConnection, RTCSessionDescription, RTCRtpSender, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import SessionDescription
from av import VideoFrame
import numpy as np

STATIC = Path(__file__).parent / "static"
FPS = 30
MAX_PEERS = 4
MAX_OFFER_BYTES = 128 * 1024
HANDSHAKE_TIMEOUT = 15
EMPTY_RGB = np.zeros((640, 960, 3), dtype=np.uint8)
EMPTY_RGB.setflags(write=False)


class PreviewPeerConnection(RTCPeerConnection):
    """Drain ICE work before transport close; private hooks support only the pinned runtime."""

    def __init__(self, *args, **kwargs):
        for package, expected in (('aiortc', '1.15.0'), ('aioice', '0.10.2')):
            if version(package) != expected:
                raise RuntimeError(f'Preview ICE cleanup requires {package}=={expected}')
        super().__init__(*args, **kwargs)
        self._ice_connect_tasks = set()
        self._ice_stopping = False

    async def _RTCPeerConnection__connect(self):
        if self._ice_stopping:
            return
        task = asyncio.current_task()
        self._ice_connect_tasks.add(task)
        try:
            await super()._RTCPeerConnection__connect()
        finally:
            self._ice_connect_tasks.discard(task)

    @staticmethod
    async def _cancel_ice_tasks(tasks):
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                asyncio.get_running_loop().call_exception_handler(dict(
                    message='Preview ICE task failed before shutdown', exception=result, task=task))

    async def close(self):
        self._ice_stopping = True
        connections = {t.receiver.transport.transport._connection for t in self.getTransceivers()}
        # Incoming authenticated checks can otherwise schedule new tasks while close awaits cancellation.
        for connection in connections:
            connection.check_start_task = lambda pair: None
        await self._cancel_ice_tasks(list(self._ice_connect_tasks))
        checks = [pair.task for connection in connections for pair in connection._check_list
                  if pair.task is not None]
        await self._cancel_ice_tasks(checks)
        await super().close()


class LatestFrameTrack(VideoStreamTrack):
    """A viewer owns a clock, never a simulator or a queue of simulation frames."""

    def __init__(self, preview):
        super().__init__()
        self.preview = preview
        self.deadline = self.started = self.last_sent = None
        self.pts = -1
        self.frames_sent = 0
        self.source_frame_id = 0

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self.started is None:
            self.started = self.deadline = now
        await asyncio.sleep(max(0, self.deadline - now))
        if self.readyState != "live":
            raise MediaStreamError
        now = loop.time()
        frame_id, rgb = self.preview.latest_frame()
        # Encode each published pose once; retain a one-second paused-video heartbeat.
        while self.frames_sent and frame_id == self.source_frame_id and now - self.last_sent < 1:
            await asyncio.sleep(min(1 / FPS, 1 - (now - self.last_sent)))
            if self.readyState != "live":
                raise MediaStreamError
            now = loop.time()
            frame_id, rgb = self.preview.latest_frame()
        self.deadline, self.last_sent = now + 1 / FPS, now
        self.source_frame_id = frame_id
        # Published RGB is never mutated; PyAV retains its array while this frame is in use.
        frame = VideoFrame.from_numpy_buffer(EMPTY_RGB if rgb is None else rgb, format="rgb24")
        self.pts = max(self.pts + 1, round((now - self.started) * 90000))
        frame.pts, frame.time_base = self.pts, Fraction(1, 90000)
        self.frames_sent += 1
        return frame


def public_sdp(sdp, public_host):
    """Publish only sockets that the deployment maps to its public UDP endpoint."""
    address = ip_address(public_host)
    lines, candidates = [], 0
    for line in sdp.splitlines():
        if line.startswith("a=candidate:"):
            fields = line.split()
            if len(fields) < 8 or fields[2].lower() != "udp" or fields[6:8] != ["typ", "host"]:
                raise ValueError("Public preview requires UDP host candidates")
            if not 31870 <= int(fields[5]) <= 31889:
                raise ValueError("Media UDP port must be in mapped range 31870-31889; check container port configuration")
            fields[4] = str(address)
            line = " ".join(fields)
            candidates += 1
        elif line.startswith("c=IN "):
            line = f"c=IN IP{address.version} {address}"
        lines.append(line)
    if not candidates:
        raise ValueError("No mapped UDP host candidate is available")
    return "\r\n".join(lines) + "\r\n"


def create_app(preview, public_host=None, nvenc=False, encoder_gpu=0):
    if nvenc:
        from preview.nvenc import check_nvenc
        check_nvenc(gpu=encoder_gpu)
    if public_host is not None:
        public_host = str(ip_address(public_host))
    pcs, peer_tasks, closing = set(), {}, {}

    @web.middleware
    async def response_headers(request, handler):
        try:
            response = await handler(request)
        except web.HTTPException as error:
            response = web.json_response(dict(error=error.reason), status=error.status)
        except (ValueError, TypeError, KeyError) as error:
            response = web.json_response(dict(error=str(error)), status=400)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
        return response

    from preview.pico import MAX_BYTES
    app = web.Application(client_max_size=MAX_BYTES, middlewares=[response_headers])
    app["pcs"], app["peer_tasks"] = pcs, peer_tasks

    async def close_peer(pc):
        if pc not in pcs:
            return
        completion = asyncio.get_running_loop().create_future()
        # The original caller still receives errors; consume the future's copy if
        # no concurrent shutdown happens to wait for it.
        completion.add_done_callback(lambda future: future.exception())
        closing[pc] = completion
        pcs.discard(pc)
        try:
            for sender in pc.getSenders():
                if sender.track is not None:
                    sender.track.stop()
            task = peer_tasks.pop(pc, None)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if public_host is not None and pc.remoteDescription is not None:
                # End an empty checklist too, so aioice does not keep waiting after close.
                await pc.addIceCandidate(None)
            await pc.close()
        except BaseException as error:
            completion.set_exception(error)
            raise
        else:
            completion.set_result(None)
        finally:
            closing.pop(pc, None)

    async def expire_peer(pc):
        try:
            await asyncio.sleep(HANDSHAKE_TIMEOUT)
            if pc.connectionState != "connected":
                await close_peer(pc)
        finally:
            peer_tasks.pop(pc, None)

    async def shutdown(_):
        while pcs or closing:
            await asyncio.gather(*(close_peer(pc) for pc in tuple(pcs)),
                                 *(asyncio.shield(done) for done in tuple(closing.values())))

    def check_origin(request):
        origin = request.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme != request.scheme or parsed.netloc != request.host or parsed.path or parsed.query or parsed.fragment:
                raise web.HTTPForbidden(reason="Cross-origin requests are refused")
    async def read_json(request, limit):
        check_origin(request)
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType(reason="Expected application/json")
        if request.content_length is not None and not 0 < request.content_length <= limit:
            raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=request.content_length)
        body = await request.read()
        if not 0 < len(body) <= limit:
            raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=len(body))
        return json.loads(body)

    async def status(_):
        peers = []
        for pc in tuple(pcs):
            sender = next(iter(pc.getSenders()), None)
            track = sender.track if sender is not None else None
            encoder = getattr(sender, '_RTCRtpSender__encoder', None)
            codec = getattr(encoder, 'codec', None)
            peers.append(dict(connection=pc.connectionState, ice=pc.iceConnectionState,
                              frames_sent=getattr(track, "frames_sent", 0),
                              source_frame_id=getattr(track, "source_frame_id", 0),
                              encoder=getattr(codec, 'name', None),
                              encoder_frames=getattr(encoder, 'frames_encoded', 0),
                              target_bitrate=getattr(encoder, 'target_bitrate', None)))
        return web.json_response({**preview.snapshot(), "webrtc": dict(peers=len(pcs), target_fps=FPS, nvenc=nvenc, encoder_gpu=encoder_gpu if nvenc else None, connections=peers)})

    async def motions(request):
        return web.json_response(preview.search(request.query.get("robot", "q1"), request.query.get("q", ""),request.query.get('release')))

    async def control(request):
        preview.control(await read_json(request, 4096))
        return web.json_response(dict(accepted=True), status=202)

    async def recordings(request):
        if getattr(preview, 'recordings', None) is None:
            return web.json_response(dict(recordings=[], enabled=False))
        return web.json_response(dict(recordings=await asyncio.to_thread(preview.recordings.list), enabled=True))

    upload_lock = asyncio.Lock()

    async def upload_pico(request):
        check_origin(request)
        store = getattr(preview, 'recordings', None)
        if store is None:
            raise web.HTTPServiceUnavailable(reason="PICO 上传尚未启用")
        if request.content_type != 'application/x-ndjson':
            raise web.HTTPUnsupportedMediaType(reason="Expected application/x-ndjson")
        # Reject concurrent conversions before buffering a second full file.
        if upload_lock.locked():
            raise web.HTTPConflict(reason="已有 PICO 文件正在处理，请稍后重试")
        async with upload_lock:
            raw = await request.read()
            filename = unquote(request.headers.get('X-Filename', 'pico.jsonl'))
            record = await asyncio.to_thread(store.upload, raw, filename)
        return web.json_response(dict(recording=record), status=201)

    async def offer(request):
        data = await read_json(request, MAX_OFFER_BYTES)
        if not isinstance(data, dict) or set(data) != {"sdp", "type"} or data["type"] != "offer" or not isinstance(data["sdp"], str):
            raise ValueError("Expected an SDP offer object with only sdp and type")
        try:
            media = SessionDescription.parse(data["sdp"]).media
        except Exception as error:
            raise ValueError("Invalid WebRTC SDP") from error
        if len(media) != 1 or media[0].kind != "video" or media[0].direction != "recvonly" or not media[0].port:
            raise ValueError("Offer must request one receive-only video stream")
        if len(pcs) >= MAX_PEERS:
            raise web.HTTPServiceUnavailable(reason="Preview already has four video viewers")
        pc = PreviewPeerConnection(RTCConfiguration(iceServers=[], bundlePolicy=RTCBundlePolicy.MAX_BUNDLE))
        sender = pc.addTrack(LatestFrameTrack(preview))
        if nvenc:
            codecs = [c for c in RTCRtpSender.getCapabilities('video').codecs
                      if c.mimeType.lower() in ('video/h264', 'video/rtx')]
            pc.getTransceivers()[0].setCodecPreferences(codecs)
        pcs.add(pc)
        peer_tasks[pc] = asyncio.create_task(expire_peer(pc))

        @pc.on("connectionstatechange")
        @pc.on("iceconnectionstatechange")
        async def connection_changed():
            if public_host is not None and pc.connectionState == "connected":
                await pc.addIceCandidate(None)
            if pc.connectionState in ("failed", "closed", "disconnected") or pc.iceConnectionState in ("failed", "closed", "disconnected"):
                await close_peer(pc)

        try:
            remote_sdp = data["sdp"]
            if public_host is not None:
                # Let authenticated browser checks discover the return route through public UDP forwarding.
                remote_sdp = "\r\n".join(line for line in remote_sdp.splitlines()
                                         if not line.startswith(("a=candidate:", "a=end-of-candidates"))) + "\r\n"
            async with asyncio.timeout(HANDSHAKE_TIMEOUT):
                await pc.setRemoteDescription(RTCSessionDescription(sdp=remote_sdp, type="offer"))
                answer = await pc.createAnswer()
                if nvenc:
                    from preview.nvenc import NvencH264Encoder, receiver_size
                    chosen = next(c for c in SessionDescription.parse(answer.sdp).media[0].rtp.codecs
                                  if c.mimeType.lower() == 'video/h264')
                    remote = next(c for c in media[0].rtp.codecs if c.payloadType == chosen.payloadType)
                    # aiortc1.15 has no per-sender encoder factory; set its pinned slot before RTP starts.
                    sender._RTCRtpSender__encoder = NvencH264Encoder(max_size=receiver_size(remote.parameters), gpu=encoder_gpu)
                await pc.setLocalDescription(answer)
            sdp = pc.localDescription.sdp
            if public_host is not None:
                try:
                    sdp = public_sdp(sdp, public_host)
                except ValueError as error:
                    raise web.HTTPServiceUnavailable(reason=str(error)) from error
            return web.json_response(dict(sdp=sdp, type=pc.localDescription.type))
        except asyncio.CancelledError:
            await close_peer(pc)
            raise
        except web.HTTPException:
            await close_peer(pc)
            raise
        except Exception as error:
            await close_peer(pc)
            raise ValueError("Invalid or timed out WebRTC offer") from error

    async def frame(request):
        _, rgb = preview.latest_frame()
        if rgb is None:
            return web.json_response(dict(error="No policy simulation frame; select a configured robot/motion"), status=409)

        def encode():
            from PIL import Image
            output = io.BytesIO()
            Image.fromarray(rgb).save(output, format="PNG" if lossless else "JPEG", quality=80)
            return output.getvalue()
        lossless = request.query.get('format') == 'png'
        return web.Response(body=await asyncio.to_thread(encode), content_type="image/png" if lossless else "image/jpeg")

    async def static(request):
        filename, mime = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                          "/style.css": ("style.css", "text/css"),
                          "/preview-font.woff2": ("preview-font.woff2", "font/woff2")}[request.path]
        return web.Response(body=(STATIC / filename).read_bytes(), content_type=mime,
                            charset="utf-8" if mime.startswith("text/") else None)

    app.on_shutdown.append(shutdown)
    app.router.add_get("/api/recordings", recordings)
    app.router.add_post("/api/recordings", upload_pico)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/motions", motions)
    app.router.add_post("/api/control", control)
    app.router.add_post("/api/offer", offer)
    app.router.add_get("/api/frame", frame)
    for path in ("/", "/app.js", "/style.css", "/preview-font.woff2"):
        app.router.add_get(path, static)
    return app
