"""NVENC behind aiortc's existing H.264 RTP packetizer and RTCP bitrate input."""
from fractions import Fraction

import av
from av.video.reformatter import VideoReformatter
from aiortc.codecs.h264 import H264Encoder


def receiver_size(parameters):
    profile = str(parameters.get('profile-level-id', '42000a'))
    if len(profile) != 6:
        raise ValueError('Invalid H264 profile-level-id')
    level = int(profile[-2:], 16)
    if str(parameters.get('level-asymmetry-allowed', '0')) == '1':
        maximum = str(parameters.get('max-recv-level', profile[-4:]))
        if len(maximum) != 4:
            raise ValueError('Invalid H264 max-recv-level')
        level = max(level, int(maximum[-2:], 16))
    else:
        level = min(level, 31)
    if level < 31:
        raise ValueError('Preview requires H264 level3.1 or higher')
    return (1920, 1080) if level >= 40 else (1280, 720)


class NvencH264Encoder(H264Encoder):
    def __init__(self, max_size=(1920, 1080), gpu=0):
        super().__init__()
        self.gpu = int(gpu)
        if self.gpu < 0:
            raise ValueError('NVENC requires an explicit nonnegative GPU index')
        self.max_size = max_size
        self._bitrate = 6_000_000
        self.frames_encoded = 0
        self.reformatter = VideoReformatter()

    @property
    def target_bitrate(self):
        return self._bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate):
        self._bitrate = max(250_000, min(int(bitrate), 8_000_000))

    def _encode_frame(self, frame, force_keyframe):
        scale = min(1, self.max_size[0] / frame.width, self.max_size[1] / frame.height)
        width, height = (max(2, int(v * scale) // 2 * 2) for v in (frame.width, frame.height))
        if self.codec and ((width, height) != (self.codec.width, self.codec.height)
                           or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > .1):
            self.codec = None
        if self.codec is None:
            codec = av.CodecContext.create('h264_nvenc', 'w')
            codec.width, codec.height = width, height
            codec.pix_fmt, codec.bit_rate = 'yuv420p', self.target_bitrate
            codec.framerate, codec.time_base = Fraction(30, 1), Fraction(1, 30)
            codec.gop_size, codec.max_b_frames = 60, 0
            codec.options = dict(preset='p1', tune='ll', gpu=str(self.gpu), rc='vbr',
                                 maxrate=str(self.target_bitrate), bufsize=str(self.target_bitrate),
                                 zerolatency='1', delay='0', profile='baseline',
                                 **{'forced-idr': '1'},
                                 level='4.0' if width * height > 1280 * 720 else '3.1')
            codec.open()
            self.codec = codec
            force_keyframe = True
        yuv = self.reformatter.reformat(frame, width=width, height=height, format='yuv420p')
        yuv.pict_type = av.video.frame.PictureType.I if force_keyframe else av.video.frame.PictureType.NONE
        data = b''.join(bytes(p) for p in self.codec.encode(yuv))
        self.frames_encoded += 1
        if data:
            yield from self._split_bitstream(data)


def check_nvenc(gpu=0):
    import numpy as np
    encoder = NvencH264Encoder(gpu=gpu)
    frame = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), dtype=np.uint8), format='rgb24')
    frame.pts, frame.time_base = 0, Fraction(1, 30)
    packets, _ = encoder.encode(frame, True)
    if not packets or encoder.codec.name != 'h264_nvenc':
        raise RuntimeError('NVENC initialization failed; software fallback is disabled')
