import fractions
from unittest import TestCase

import av
from aiortc.codecs.h265 import (
    H265Decoder,
    H265Encoder,
    H265PayloadDescriptor,
    h265_depayload,
)
from aiortc.jitterbuffer import JitterFrame

from .codecs import CodecTestCase


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
    def test_decoder_junk(self) -> None:
        self.assertEqual(H265Decoder().decode(JitterFrame(b"123", 0)), [])

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

    def test_level_limits(self) -> None:
        encoder = H265Encoder(level_id=30)
        self.assertEqual(encoder.target_bitrate, 128000)
        encoder.target_bitrate = 3000000
        self.assertEqual(encoder.target_bitrate, 128000)
        with self.assertRaises(ValueError):
            encoder.encode(self.create_video_frame(640, 480, 0))
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
