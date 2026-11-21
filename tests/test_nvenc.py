"""Exercise actual NVENC bitstreams through aiortc's H.264 depacketizer/decoder."""
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import importlib.util
import os
AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ('av','numpy','aiortc'))
if AVAILABLE:
    import av
    import numpy as np
    from aiortc.codecs.h264 import H264Decoder,H264Encoder,h264_depayload
    from aiortc.jitterbuffer import JitterFrame

@unittest.skipUnless(AVAILABLE and os.environ.get('BLINKMORPH_TEST_NVENC')=='1','requires explicit NVENC hardware test environment')
class NvencTests(unittest.TestCase):
    def encoder(self, **kwargs):
        from preview.nvenc import NvencH264Encoder
        return NvencH264Encoder(**kwargs)

    def frame(self,width=1920,height=1080,pts=0):
        rgb=np.zeros((height,width,3),dtype=np.uint8);rgb[:,:,1]=130;rgb[100:250,100:250]=[200,60,40]
        f=av.VideoFrame.from_ndarray(rgb,format='rgb24');f.pts=pts;f.time_base=Fraction(1,90000)
        return f

    def decode(self,encoder,decoder,frame,force=False):
        payloads,ts=encoder.encode(frame,force)
        self.assertTrue(payloads)
        raw=b''.join(h264_depayload(p) for p in payloads)
        decoded=decoder.decode(JitterFrame(data=raw,timestamp=ts))
        self.assertEqual(len(decoded),1)
        return decoded[0],raw,ts

    def test_real_1080p_nvenc_packets_decode_and_honor_keyframe_requests(self):
        encoder=self.encoder();decoder=H264Decoder()
        first,_,_=self.decode(encoder,decoder,self.frame())
        self.assertEqual((first.width,first.height),(1920,1080))
        self.assertEqual(encoder.codec.name,'h264_nvenc')
        _,_,ts=self.decode(encoder,decoder,self.frame(pts=3000))
        self.assertEqual(ts,3000)
        _,raw,ts=self.decode(encoder,decoder,self.frame(pts=6000),True)
        nal=list(H264Encoder._split_bitstream(raw))
        self.assertTrue(any(n[0]&31==5 for n in nal),'PLI must produce an IDR')
        sps=next(n for n in nal if n[0]&31==7)
        self.assertEqual(sps[3],40,'1080p30 must use level4.0')

    def test_receiver_720p_limit_downscales_without_changing_rtp_clock(self):
        encoder=self.encoder(max_size=(1280,720));decoder=H264Decoder()
        frame,_,ts=self.decode(encoder,decoder,self.frame(pts=3000))
        self.assertEqual((frame.width,frame.height),(1280,720));self.assertEqual(ts,3000)

    def test_native_720p_frame_decodes_at_720p_for_1080p_capable_peer(self):
        encoder=self.encoder(max_size=(1920,1080));decoder=H264Decoder()
        frame,raw,ts=self.decode(encoder,decoder,self.frame(1280,720,3000))
        self.assertEqual((frame.width,frame.height),(1280,720));self.assertEqual(ts,3000)
        sps=next(n for n in H264Encoder._split_bitstream(raw) if n[0]&31==7)
        self.assertEqual(sps[3],31)

    def test_resize_and_bitrate_update_keep_decodable_output(self):
        encoder=self.encoder();decoder=H264Decoder()
        self.decode(encoder,decoder,self.frame())
        encoder.target_bitrate=2000000
        frame,_,_=self.decode(encoder,decoder,self.frame(640,480,3000))
        self.assertEqual((frame.width,frame.height),(640,480))
        self.assertEqual(encoder.codec.bit_rate,2000000)
        encoder.target_bitrate=100000000
        self.assertEqual(encoder.target_bitrate,8000000)
        encoder.target_bitrate=0
        self.assertEqual(encoder.target_bitrate,250000)

    def test_hardware_initialization_failure_is_not_a_software_fallback(self):
        encoder=self.encoder();frame=self.frame(64,64)
        create=SimpleNamespace(create=lambda *a: (_ for _ in ()).throw(RuntimeError('hardware unavailable')))
        with patch.object(av,'CodecContext',create):
            with self.assertRaisesRegex(RuntimeError,'hardware unavailable'):
                encoder.encode(frame)

    def test_receiver_level_limits_and_verified_1080p_extension(self):
        from preview.nvenc import receiver_size
        self.assertEqual(receiver_size({'profile-level-id':'42e01f'}),(1280,720))
        self.assertEqual(receiver_size({'profile-level-id':'42e01f','level-asymmetry-allowed':'1','max-recv-level':'e028'}),(1920,1080))
        self.assertEqual(receiver_size({'profile-level-id':'42e01f','level-asymmetry-allowed':'0','max-recv-level':'e028'}),(1280,720))
        with self.assertRaises(ValueError):receiver_size({'profile-level-id':'42000a'})

    def test_dedicated_gpu1_encodes_without_using_the_control_gpu(self):
        import torch
        if torch.cuda.device_count()<2:self.skipTest('requires a separate encoding GPU')
        encoder=self.encoder(gpu=1)
        frame,_,_=self.decode(encoder,H264Decoder(),self.frame(640,480))
        self.assertEqual((frame.width,frame.height),(640,480))
        self.assertEqual(encoder.gpu,1)
        self.assertEqual(encoder.codec.name,'h264_nvenc')

    def test_static_frames_do_not_fill_the_entire_network_budget(self):
        encoder=self.encoder();encoder.target_bitrate=8000000
        sizes=[]
        for i in range(90):
            payloads,_=encoder.encode(self.frame(640,480,i*3000))
            self.assertTrue(payloads)
            if i>=30:sizes.append(sum(len(p) for p in payloads))
        self.assertLess(sum(sizes)/len(sizes)*30*8,1000000,'unchanged video must not send padding at8Mbps')
