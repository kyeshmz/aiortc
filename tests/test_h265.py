import fractions
from unittest import TestCase

import av
from aiortc.codecs import depayload, get_decoder, get_encoder
from aiortc.codecs.h265 import (
    H265Decoder,
    H265Encoder,
    H265PayloadDescriptor,
    h265_depayload,
    parse_h265_parameters,
)
from aiortc.jitterbuffer import JitterFrame
from aiortc.rtcpeerconnection import (
    filter_preferred_codecs,
    find_common_codecs,
    is_codec_compatible,
)
from aiortc.rtcrtpparameters import (
    ParametersDict,
    RTCRtpCodecCapability,
    RTCRtpCodecParameters,
)
from aiortc.sdp import SessionDescription

from .codecs import CodecTestCase

H265_CODEC = RTCRtpCodecParameters(
    mimeType="video/H265", clockRate=90000, payloadType=103
)


class H265NegotiationTest(TestCase):
    def test_factories_and_depayload(self) -> None:
        self.assertIsInstance(get_encoder(H265_CODEC), H265Encoder)
        self.assertIsInstance(get_decoder(H265_CODEC), H265Decoder)
        self.assertEqual(
            depayload(H265_CODEC, b"\x40\x01a"), b"\x00\x00\x00\x01\x40\x01a"
        )

    def test_supported_parameters(self) -> None:
        cases: list[tuple[ParametersDict, int]] = [
            ({}, 93),
            ({"profile-id": "1", "tier-flag": "0", "level-id": "93"}, 93),
            ({"profile-id": 1, "tier-flag": 0, "level-id": 93}, 93),
            ({"profile-space": "0", "tx-mode": "SRST", "sprop-max-don-diff": "0"}, 93),
            (
                {
                    "profile-compatibility-indicator": "40000000",
                    "interop-constraints": "B00000000000",
                },
                93,
            ),
        ]
        for level in (30, 60, 63, 90, 93, 120, 123, 150, 153, 156, 180, 183, 186):
            cases.append(({"level-id": str(level)}, level))
        for parameters, expected in cases:
            with self.subTest(parameters=parameters):
                remote = RTCRtpCodecParameters(
                    mimeType="video/h265",
                    clockRate=90000,
                    payloadType=110,
                    parameters=parameters,
                )
                self.assertEqual(parse_h265_parameters(parameters), expected)
                self.assertTrue(is_codec_compatible(H265_CODEC, remote))
                common = find_common_codecs([H265_CODEC], [remote])
                self.assertEqual(len(common), 1)
                self.assertEqual(
                    common[0].parameters, {"level-id": str(min(93, expected))}
                )
                self.assertEqual(common[0].payloadType, 110)
                self.assertEqual(H265_CODEC.parameters, {})

    def test_unsupported_parameters(self) -> None:
        cases: list[ParametersDict] = [
            {"profile-id": "2"},
            {"profile-id": "-1"},
            {"profile-id": "32"},
            {"profile-space": "1"},
            {"tier-flag": "1"},
            {"tier-flag": "2"},
            {"tx-mode": "MRST"},
            {"tx-mode": "MRMT"},
            {"tx-mode": "invalid"},
            {"sprop-max-don-diff": "1"},
            {"sprop-max-don-diff": "-1"},
            {"sprop-max-don-diff": "invalid"},
            {"sprop-depack-buf-nalus": "1"},
            {"sprop-vps": "QAEMAf//"},
            {"sprop-sps": "QgE="},
            {"sprop-pps": "RAE="},
            {"recv-sub-layer-id": "0"},
            {"sprop-sub-layer-id": "0"},
            {"max-recv-level-id": "120"},
            {"max-fps": "1500"},
            {"level-id": "0"},
            {"level-id": "31"},
            {"level-id": "256"},
            {"level-id": "-1"},
            {"level-id": "3.1"},
            {"level-id": "1e2"},
            {"level-id": "invalid"},
            {"level-id": " 93"},
            {"level-id": None},
            {"interop-constraints": "000000000000"},
            {"interop-constraints": "b00000000001"},
            {"profile-compatibility-indicator": "60000000"},
            {"packetization-mode": "1"},
        ]
        for parameters in cases:
            remote = RTCRtpCodecParameters(
                mimeType="video/H265",
                clockRate=90000,
                payloadType=110,
                parameters=parameters,
            )
            rtx = RTCRtpCodecParameters(
                mimeType="video/rtx",
                clockRate=90000,
                payloadType=111,
                parameters={"apt": 110},
            )
            with self.subTest(parameters=parameters):
                self.assertFalse(is_codec_compatible(H265_CODEC, remote))
                self.assertFalse(is_codec_compatible(remote, H265_CODEC))
                self.assertEqual(find_common_codecs([H265_CODEC], [remote, rtx]), [])
                with self.assertRaises(ValueError):
                    get_encoder(remote)
                with self.assertRaises(ValueError):
                    get_decoder(remote)

    def test_lower_level_preferences_and_rtx(self) -> None:
        remote = RTCRtpCodecParameters(
            mimeType="video/H265",
            clockRate=90000,
            payloadType=110,
            parameters={"level-id": "60"},
        )
        rtx = RTCRtpCodecParameters(
            mimeType="video/rtx",
            clockRate=90000,
            payloadType=111,
            parameters={"apt": 110},
        )
        common = find_common_codecs([H265_CODEC], [remote, rtx])
        preferred = [
            RTCRtpCodecCapability(
                mimeType="video/H265", clockRate=90000, parameters={"level-id": "93"}
            ),
            RTCRtpCodecCapability(mimeType="video/rtx", clockRate=90000),
        ]
        filtered = filter_preferred_codecs(common, preferred)
        self.assertEqual(filtered, [remote, rtx])
        encoder = get_encoder(filtered[0])
        assert isinstance(encoder, H265Encoder)
        self.assertEqual(encoder.level_id, 60)
        self.assertEqual(encoder.target_bitrate, 1000000)
        rtx.parameters["apt"] = 109
        self.assertEqual(find_common_codecs([H265_CODEC], [remote, rtx]), [remote])
        rtx.parameters["apt"] = 110
        rtx.clockRate = 8000
        self.assertEqual(find_common_codecs([H265_CODEC], [remote, rtx]), [remote])

    def test_sdp_parameters_roundtrip(self) -> None:
        sdp = (
            "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"
            "m=video 9 UDP/TLS/RTP/SAVPF 110 111\r\n"
            "c=IN IP4 0.0.0.0\r\n"
            "a=rtpmap:110 H265/90000\r\n"
            "a=fmtp:110 profile-id=1;tier-flag=0;level-id=60;"
            "tx-mode=SRST;sprop-max-don-diff=0\r\n"
            "a=rtpmap:111 rtx/90000\r\na=fmtp:111 apt=110\r\n"
        )
        parsed = SessionDescription.parse(sdp)
        codec, rtx = parsed.media[0].rtp.codecs
        self.assertEqual(
            codec.parameters,
            {
                "profile-id": "1",
                "tier-flag": "0",
                "level-id": "60",
                "tx-mode": "SRST",
                "sprop-max-don-diff": "0",
            },
        )
        self.assertEqual(rtx.parameters, {"apt": 110})
        self.assertIn("a=rtpmap:110 H265/90000\r\n", str(parsed))
        self.assertIn(
            "a=fmtp:110 profile-id=1;tier-flag=0;level-id=60;"
            "tx-mode=SRST;sprop-max-don-diff=0\r\n",
            str(parsed),
        )


class H265PayloadTest(TestCase):
    def test_single_nal(self) -> None:
        descriptor, data = H265PayloadDescriptor.parse(b"\x40\x01abc")
        self.assertTrue(descriptor.first_fragment)
        self.assertEqual(repr(descriptor), "H265PayloadDescriptor(FF=True)")
        self.assertEqual(data, b"\x00\x00\x00\x01\x40\x01abc")
        self.assertEqual(h265_depayload(b"\x02\x03abc"), b"\x00\x00\x00\x01\x02\x03abc")

    def test_aggregation(self) -> None:
        payload = b"\x60\x01\x00\x03\x40\x01a\x00\x04\x42\x03bc"
        descriptor, data = H265PayloadDescriptor.parse(payload)
        self.assertTrue(descriptor.first_fragment)
        self.assertEqual(data, b"\x00\x00\x00\x01\x40\x01a\x00\x00\x00\x01\x42\x03bc")
        self.assertEqual(
            H265Encoder._packetize([b"\x40\x01a", b"\x42\x03bc"]), [payload]
        )

    def test_fragmentation(self) -> None:
        start, data = H265PayloadDescriptor.parse(b"\x62\x03\x93abc")
        self.assertTrue(start.first_fragment)
        self.assertEqual(data, b"\x00\x00\x00\x01\x26\x03abc")
        middle, data = H265PayloadDescriptor.parse(b"\x62\x03\x13de")
        self.assertFalse(middle.first_fragment)
        self.assertEqual(data, b"de")
        end, data = H265PayloadDescriptor.parse(b"\x62\x03\x53f")
        self.assertFalse(end.first_fragment)
        self.assertEqual(data, b"f")

    def test_nal_types_and_temporal_ids(self) -> None:
        for nal_type in range(48):
            for temporal_id in range(1, 8):
                with self.subTest(nal_type=nal_type, temporal_id=temporal_id):
                    unit = bytes([nal_type << 1, temporal_id]) + b"a"
                    self.assertEqual(h265_depayload(unit), b"\x00\x00\x00\x01" + unit)
                    self.assertEqual(
                        h265_depayload(bytes([98, temporal_id, 128 | nal_type]) + b"a"),
                        b"\x00\x00\x00\x01" + unit,
                    )

    def test_truncated_aggregation(self) -> None:
        payload = b"\x60\x01\x00\x03\x40\x01a\x00\x04\x42\x01bc"
        for end in range(len(payload)):
            with self.subTest(end=end), self.assertRaises(ValueError):
                h265_depayload(payload[:end])

    def test_malformed(self) -> None:
        payloads = [
            b"",
            b"\x40",
            b"\x40\x01",
            b"\xc0\x01a",
            b"\x40\x00a",
            b"\x41\x01a",
            b"\x40\x09a",
            b"\x64\x01a",
            b"\x7e\x01a",
            b"\x60\x01\x00",
            b"\x60\x01\x00\x03\x40\x01",
            b"\x60\x01\x00\x00",
            b"\x60\x01\x00\x01a",
            b"\x60\x01\x00\x02\x40\x01",
            b"\x60\x01\x00\x03\x40\x01a",
            b"\x60\x01\x00\x03\x60\x01a\x00\x03\x40\x01a",
            b"\x60\x01\x00\x03\x62\x01a\x00\x03\x40\x01a",
            b"\x60\x01\x00\x03\xc0\x01a\x00\x03\x40\x01a",
            b"\x60\x01\x00\x03\x40\x00a\x00\x03\x40\x01a",
            b"\x60\x02\x00\x03\x40\x01a\x00\x03\x40\x02a",
            b"\x60\x01\x00\x03\x40\x01a\x00\x03\x40\x01a\x00",
            b"\x62\x01\x93",
            b"\x62\x01\xd3a",
            b"\x62\x01\xb0a",
            b"\x62\x01\xb1a",
            b"\x62\x01\xb2a",
        ]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                h265_depayload(payload)

    def test_payload_limits(self) -> None:
        for size in (3, 1296, 1297, 1299, 1300, 1301, 2596, 2597, 2598, 65536):
            unit = b"\x26\x01" + b"a" * (size - 2)
            with self.subTest(size=size):
                payloads = H265Encoder._packetize([unit])
                self.assertTrue(payloads)
                self.assertTrue(all(0 < len(p) <= 1300 for p in payloads))
                self.assertEqual(
                    b"".join(map(h265_depayload, payloads)), b"\x00\x00\x00\x01" + unit
                )
                if size <= 1300:
                    self.assertEqual(payloads, [unit])
                else:
                    self.assertEqual(payloads[0][:3], b"\x62\x01\x93")
                    self.assertEqual(payloads[-1][:3], b"\x62\x01\x53")

    def test_aggregation_limits_and_order(self) -> None:
        units = [b"\x40\x01" + b"a" * 644, b"\x42\x01" + b"b" * 646]
        payloads = H265Encoder._packetize(units)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(len(payloads[0]), 1300)
        units += [b"\x26\x01" + b"c" * 1300, b"\x02\x01d"]
        payloads = H265Encoder._packetize(units)
        self.assertEqual(len(payloads), 4)
        self.assertEqual(
            b"".join(map(h265_depayload, payloads)),
            b"".join(b"\x00\x00\x00\x01" + u for u in units),
        )
        self.assertEqual(H265Encoder._packetize([]), [])
        self.assertEqual(
            H265Encoder._packetize([b"\x40\x01a"] * 20),
            [b"\x60\x01" + b"\x00\x03\x40\x01a" * 20],
        )

    def test_vcl_temporal_layers_are_not_mixed(self) -> None:
        for units in (
            [b"\x40\x01a", b"\x02\x03b"],
            [b"\x02\x03a", b"\x40\x01b"],
            [b"\x02\x01a", b"\x02\x03b"],
        ):
            with self.subTest(units=units):
                self.assertEqual(H265Encoder._packetize(units), units)

    def test_split_bitstream(self) -> None:
        self.assertEqual(
            list(
                H265Encoder._split_bitstream(
                    b"\x00\x00\x00\x01\x40\x01a\x00\x00\x01\x42\x01b"
                )
            ),
            [b"\x40\x01a", b"\x42\x01b"],
        )
        for data in (b"", b"abcd", b"abc\x00\x00\x01\x40\x01a"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                list(H265Encoder._split_bitstream(data))
        for unit in (b"", b"\x60\x01a", b"\x40\x00a", b"\x41\x01a"):
            with self.subTest(unit=unit), self.assertRaises(ValueError):
                H265Encoder._packetize([unit])


class H265Test(CodecTestCase):
    def test_encoder_buffering(self) -> None:
        encoder = H265Encoder()
        codec = self.ensureIsInstance(
            av.CodecContext.create("libx265", "w"), av.VideoCodecContext
        )
        codec.width = 320
        codec.height = 240
        codec.bit_rate = 1000000
        codec.pix_fmt = "yuv420p"
        codec.time_base = fractions.Fraction(1, 30)
        codec.framerate = fractions.Fraction(30)
        codec.thread_count = 1
        codec.options = {
            "preset": "ultrafast",
            "x265-params": (
                "pools=none:frame-threads=1:wpp=0:log-level=error:"
                "bframes=0:rc-lookahead=1"
            ),
        }
        encoder.codec = codec
        self.assertEqual(encoder.encode(self.create_video_frame(320, 240, 0)), ([], 0))
        self.assertEqual(len(codec.encode(None)), 1)

    def test_decoder_junk(self) -> None:
        for data in (
            b"123",
            b"\x00\x00\x00\x01\x26\x01a",
            b"\x00\x00\x00\x01\x40\x01a",
            b"\x00\x00\x00\x01\xff" * 5,
        ):
            with self.subTest(data=data):
                self.assertEqual(H265Decoder().decode(JitterFrame(data, 0)), [])

    def test_roundtrip(self) -> None:
        for width, height in ((320, 240), (640, 480), (960, 540), (1280, 720)):
            for time_base in (
                fractions.Fraction(1, 90000),
                fractions.Fraction(1, 9000),
            ):
                with self.subTest(width=width, height=height, time_base=time_base):
                    encoder = H265Encoder()
                    decoder = H265Decoder()
                    for i in range(3):
                        frame = self.create_video_frame(
                            width, height, int(i / time_base / 30), time_base=time_base
                        )
                        payloads, timestamp = encoder.encode(frame)
                        self.assertTrue(payloads)
                        self.assertEqual(timestamp, i * 3000)
                        decoded = decoder.decode(
                            JitterFrame(
                                b"".join(map(h265_depayload, payloads)), timestamp
                            )
                        )
                        self.assertEqual(len(decoded), 1)
                        assert isinstance(decoded[0], av.VideoFrame)
                        self.assertEqual(
                            (decoded[0].width, decoded[0].height), (width, height)
                        )
                        self.assertEqual(decoded[0].pts, i * 3000)
                        self.assertEqual(
                            decoded[0].time_base, fractions.Fraction(1, 90000)
                        )

    def test_roundtrip_source_timestamps(self) -> None:
        for time_base, source_pts, expected_timestamps in (
            (
                fractions.Fraction(1, 90000),
                [0, 3600, 7200, 10800],
                [0, 3600, 7200, 10800],
            ),
            (
                fractions.Fraction(1, 9000),
                [91, 466, 841, 1216],
                [910, 4660, 8410, 12160],
            ),
        ):
            with self.subTest(time_base=time_base, source_pts=source_pts):
                encoder = H265Encoder()
                decoder = H265Decoder()
                timestamps = []
                decoded_pts = []
                for pts in source_pts:
                    frame = self.create_video_frame(320, 240, pts, time_base=time_base)
                    payloads, timestamp = encoder.encode(frame)
                    self.assertTrue(payloads)
                    timestamps.append(timestamp)
                    decoded = decoder.decode(
                        JitterFrame(b"".join(map(h265_depayload, payloads)), timestamp)
                    )
                    self.assertEqual(len(decoded), 1)
                    decoded_pts.append(decoded[0].pts)
                    self.assertEqual(decoded[0].time_base, fractions.Fraction(1, 90000))
                self.assertEqual(timestamps, expected_timestamps)
                self.assertEqual(decoded_pts, expected_timestamps)

    def test_keyframes_bitrate_and_resolution(self) -> None:
        encoder = H265Encoder()
        decoder = H265Decoder()
        self.assertEqual(encoder.target_bitrate, 1000000)
        encoder.target_bitrate = -1
        self.assertEqual(encoder.target_bitrate, 500000)
        encoder.target_bitrate = 100000000
        self.assertEqual(encoder.target_bitrate, 3000000)
        for i, (width, height, force) in enumerate(
            (
                (640, 480, False),
                (640, 480, False),
                (640, 480, True),
                (640, 480, False),
                (320, 240, False),
                (320, 256, False),
            )
        ):
            if i == 3:
                encoder.target_bitrate = 1200000
            payloads, timestamp = encoder.encode(
                self.create_video_frame(width, height, i * 3000), force
            )
            data = b"".join(map(h265_depayload, payloads))
            types = {(u[0] >> 1) & 63 for u in H265Encoder._split_bitstream(data)}
            frames = decoder.decode(JitterFrame(data, timestamp))
            self.assertEqual(len(frames), 1)
            assert isinstance(frames[0], av.VideoFrame)
            self.assertEqual(
                (frames[0].width, frames[0].height, frames[0].pts),
                (width, height, i * 3000),
            )
            if i == 1:
                self.assertFalse(types.intersection(range(16, 24)))
                self.assertFalse(frames[0].key_frame)
            else:
                self.assertTrue(types.issuperset({32, 33, 34}))
                self.assertTrue(types.intersection({19, 20}))
                ordered_types = [u[0] >> 1 for u in H265Encoder._split_bitstream(data)]
                self.assertEqual(ordered_types[:3], [32, 33, 34])
                self.assertTrue(frames[0].key_frame)
                fresh_frames = H265Decoder().decode(JitterFrame(data, timestamp))
                self.assertEqual(len(fresh_frames), 1)
                self.assertEqual(fresh_frames[0].pts, i * 3000)

    def test_pack(self) -> None:
        encoder = H265Encoder()
        payloads, timestamp = encoder.pack(
            self.create_packet(b"\x00\x00\x01\x26\x01abc", 1)
        )
        self.assertEqual(payloads, [b"\x26\x01abc"])
        self.assertEqual(timestamp, 90)
        with self.assertRaises(ValueError):
            encoder.pack(self.create_packet(b"not Annex B", 0))

    def test_rgb24_roundtrip(self) -> None:
        encoder = H265Encoder()
        frame = self.create_video_frame(320, 240, 0, format="rgb24")
        for plane in frame.planes:
            plane.update(b"\xff\x00\x00" * (plane.buffer_size // 3))
        payloads, timestamp = encoder.encode(frame, force_keyframe=True)
        self.assertTrue(payloads)
        self.assertEqual(timestamp, 0)
        data = b"".join(map(h265_depayload, payloads))
        frames = H265Decoder().decode(JitterFrame(data, timestamp))
        self.assertEqual(len(frames), 1)
        assert isinstance(frames[0], av.VideoFrame)
        self.assertEqual(
            (frames[0].width, frames[0].height, frames[0].pts), (320, 240, 0)
        )
        self.assertTrue(frames[0].key_frame)
        pixel = frames[0].to_ndarray(format="rgb24")[120, 160]
        self.assertGreaterEqual(int(pixel[0]), 240)
        self.assertLessEqual(int(pixel[1]), 10)
        self.assertLessEqual(int(pixel[2]), 10)

    def test_level_limits(self) -> None:
        with self.assertRaises(ValueError):
            H265Encoder(level_id=120)
        encoder = H265Encoder(level_id=30)
        self.assertEqual(encoder.target_bitrate, 128000)
        encoder.target_bitrate = 3000000
        self.assertEqual(encoder.target_bitrate, 128000)
        level2_encoder = H265Encoder(level_id=60)
        level2_encoder.target_bitrate = 3000000
        self.assertEqual(level2_encoder.target_bitrate, 1500000)
        with self.assertRaises(ValueError):
            encoder.encode(self.create_video_frame(640, 480, 0))
        with self.assertRaisesRegex(
            ValueError, "Frame dimensions exceed the negotiated HEVC level at 30 fps"
        ):
            encoder.encode(self.create_video_frame(192, 128, 0))
        payloads, timestamp = encoder.encode(self.create_video_frame(128, 96, 0))
        frames = H265Decoder().decode(
            JitterFrame(b"".join(map(h265_depayload, payloads)), timestamp)
        )
        self.assertEqual(len(frames), 1)
        assert isinstance(frames[0], av.VideoFrame)
        self.assertEqual((frames[0].width, frames[0].height), (128, 96))
        for width, height in (
            (15, 32),
            (32, 15),
            (31, 32),
            (32, 31),
            (1920, 1080),
            (4096, 32),
        ):
            with (
                self.subTest(width=width, height=height),
                self.assertRaises(ValueError),
            ):
                H265Encoder().encode(self.create_video_frame(width, height, 0))

    def test_signaled_profile_tier_and_level(self) -> None:
        for level, width, height in (
            (30, 128, 96),
            (60, 320, 240),
            (63, 640, 360),
            (90, 640, 480),
            (93, 1280, 720),
        ):
            with self.subTest(level=level):
                encoder = H265Encoder(level_id=level)
                payloads, _ = encoder.encode(self.create_video_frame(width, height, 0))
                data = b"".join(map(h265_depayload, payloads))
                units = list(H265Encoder._split_bitstream(data))
                sps = next(u for u in units if u[0] >> 1 == 33)
                rbsp = sps[2:].replace(b"\x00\x00\x03", b"\x00\x00")
                self.assertEqual(rbsp[1] & 0xE0, 0)
                self.assertEqual(rbsp[1] & 0x1F, 1)
                self.assertEqual(rbsp[12], level)
