from typing import Optional
from unittest import TestCase

from aiortc.jitterbuffer import JitterBuffer
from aiortc.rtp import RtpPacket


class JitterBufferTest(TestCase):
    def assertPackets(
        self, jbuffer: JitterBuffer, expected: list[Optional[int]]
    ) -> None:
        found = [x.sequence_number if x else None for x in jbuffer._packets]
        self.assertEqual(found, expected)

    def test_create(self) -> None:
        jbuffer = JitterBuffer(capacity=1)
        self.assertEqual(jbuffer.capacity, 1)
        self.assertEqual(jbuffer._packets, [None])
        self.assertEqual(jbuffer._origin, None)

        jbuffer = JitterBuffer(capacity=2)
        self.assertEqual(jbuffer._packets, [None, None])
        self.assertEqual(jbuffer._origin, None)

        jbuffer = JitterBuffer(capacity=4)
        self.assertEqual(jbuffer._packets, [None, None, None, None])
        self.assertEqual(jbuffer._origin, None)

    def test_create_zero_capacity(self) -> None:
        with self.assertRaisesRegex(AssertionError, "^capacity must be a power of 2$"):
            JitterBuffer(capacity=0)

    def test_create_invalid_capacity(self) -> None:
        for capacity in (-2, -1, 3, 6):
            with self.subTest(capacity=capacity):
                with self.assertRaisesRegex(
                    AssertionError, "^capacity must be a power of 2$"
                ):
                    JitterBuffer(capacity=capacity)

    def test_add_ordered(self) -> None:
        jbuffer = JitterBuffer(capacity=4)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [0, None, None, None])
        self.assertEqual(jbuffer._origin, 0)
        self.assertFalse(pli_flag)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [0, 1, None, None])
        self.assertEqual(jbuffer._origin, 0)
        self.assertFalse(pli_flag)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [0, 1, 2, None])
        self.assertEqual(jbuffer._origin, 0)
        self.assertFalse(pli_flag)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [0, 1, 2, 3])
        self.assertEqual(jbuffer._origin, 0)
        self.assertFalse(pli_flag)

    def test_add_unordered(self) -> None:
        jbuffer = JitterBuffer(capacity=4)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, 1, None, None])
        self.assertEqual(jbuffer._origin, 1)
        self.assertFalse(pli_flag)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, 1, None, 3])
        self.assertEqual(jbuffer._origin, 1)
        self.assertFalse(pli_flag)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, 1, 2, 3])
        self.assertEqual(jbuffer._origin, 1)
        self.assertFalse(pli_flag)

    def test_add_seq_too_low_drop(self) -> None:
        jbuffer = JitterBuffer(capacity=4)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, None, 2, None])
        self.assertEqual(jbuffer._origin, 2)
        self.assertFalse(pli_flag)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, None, 2, None])
        self.assertEqual(jbuffer._origin, 2)
        self.assertFalse(pli_flag)

    def test_add_seq_too_low_reset(self) -> None:
        jbuffer = JitterBuffer(capacity=4)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=2000, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [2000, None, None, None])
        self.assertEqual(jbuffer._origin, 2000)
        self.assertFalse(pli_flag)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, 1, None, None])
        self.assertEqual(jbuffer._origin, 1)
        self.assertFalse(pli_flag)

    def test_add_seq_too_high_discard_one(self) -> None:
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=4, timestamp=1234))
        self.assertEqual(jbuffer._origin, 4)

        self.assertPackets(jbuffer, [4, None, None, None])

    def test_add_seq_too_high_discard_one_v2(self) -> None:
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1235))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=4, timestamp=1235))
        self.assertEqual(jbuffer._origin, 3)

        self.assertPackets(jbuffer, [4, None, None, 3])

    def test_add_seq_too_high_discard_four(self) -> None:
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=7, timestamp=1235))
        self.assertEqual(jbuffer._origin, 7)

        self.assertPackets(jbuffer, [None, None, None, 7])

    def test_add_seq_too_high_discard_more(self) -> None:
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)

        jbuffer.add(RtpPacket(sequence_number=8, timestamp=1234))
        self.assertEqual(jbuffer._origin, 8)

        self.assertPackets(jbuffer, [8, None, None, None])

    def test_add_seq_too_high_reset(self) -> None:
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        self.assertPackets(jbuffer, [0, None, None, None])

        jbuffer.add(RtpPacket(sequence_number=3000, timestamp=1234))
        self.assertEqual(jbuffer._origin, 3000)
        self.assertPackets(jbuffer, [3000, None, None, None])

    def test_remove(self) -> None:
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        self.assertPackets(jbuffer, [0, 1, 2, 3])

        # remove 1 packet
        jbuffer.remove(1)
        self.assertEqual(jbuffer._origin, 1)
        self.assertPackets(jbuffer, [None, 1, 2, 3])

        # remove 2 packets
        jbuffer.remove(2)
        self.assertEqual(jbuffer._origin, 3)
        self.assertPackets(jbuffer, [None, None, None, 3])

    def test_smart_remove(self) -> None:
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1235))
        self.assertEqual(jbuffer._origin, 0)
        self.assertPackets(jbuffer, [0, 1, None, 3])

        # remove 1 packet
        jbuffer.smart_remove(1)
        self.assertEqual(jbuffer._origin, 3)
        self.assertPackets(jbuffer, [None, None, None, 3])

    def test_smart_remove_count_boundaries(self) -> None:
        cases: list[tuple[int, bool, int, list[Optional[int]]]] = [
            (0, False, 0, [0, 1, 2, 3]),
            (1, False, 2, [None, None, 2, 3]),
            (2, False, 2, [None, None, 2, 3]),
            (3, True, 4, [None, None, None, None]),
            (4, True, 4, [None, None, None, None]),
            (5, True, 4, [None, None, None, None]),
            (100, True, 4, [None, None, None, None]),
        ]
        for count, cleared, origin, packets in cases:
            with self.subTest(count=count):
                jbuffer = JitterBuffer(capacity=4, prefetch=4)
                for sequence, timestamp in ((0, 100), (1, 100), (2, 200), (3, 200)):
                    packet = RtpPacket(sequence_number=sequence, timestamp=timestamp)
                    packet._data = b"payload"  # type: ignore
                    self.assertEqual(jbuffer.add(packet), (False, None))

                self.assertIs(jbuffer.smart_remove(count), cleared)
                self.assertEqual(jbuffer._origin, origin)
                self.assertPackets(jbuffer, packets)

    def test_smart_remove_retains_complete_frame(self) -> None:
        for start, retained, next_sequence in ((0, 3, 4), (65534, 1, 2), (65535, 2, 3)):
            with self.subTest(start=start):
                jbuffer = JitterBuffer(capacity=4)
                for offset, timestamp, data in (
                    (0, 100, b"a"),
                    (1, 100, b"b"),
                    (3, 200, b"retained"),
                ):
                    packet = RtpPacket(
                        sequence_number=(start + offset) % 65536, timestamp=timestamp
                    )
                    packet._data = data  # type: ignore
                    self.assertEqual(jbuffer.add(packet), (False, None))

                self.assertFalse(jbuffer.smart_remove(1))
                self.assertEqual(jbuffer._origin, retained)
                self.assertEqual(
                    [x.sequence_number for x in jbuffer._packets if x is not None],
                    [retained],
                )

                packet = RtpPacket(sequence_number=next_sequence, timestamp=300)
                packet._data = b"next"  # type: ignore
                pli_flag, frame = jbuffer.add(packet)
                self.assertFalse(pli_flag)
                self.assertIsNotNone(frame)
                self.assertEqual(frame.timestamp, 200)
                self.assertEqual(frame.data, b"retained")

    def test_smart_remove_clears_sparse_window(self) -> None:
        for start, cleared_origin, empty_origin in ((0, 4, 8), (65534, 2, 6)):
            with self.subTest(start=start):
                jbuffer = JitterBuffer(capacity=4)
                for offset in (0, 2):
                    self.assertEqual(
                        jbuffer.add(
                            RtpPacket(
                                sequence_number=(start + offset) % 65536, timestamp=100
                            )
                        ),
                        (False, None),
                    )

                self.assertTrue(jbuffer.smart_remove(1))
                self.assertEqual(jbuffer._origin, cleared_origin)
                self.assertPackets(jbuffer, [None, None, None, None])
                self.assertTrue(jbuffer.smart_remove(0))
                self.assertEqual(jbuffer._origin, empty_origin)
                self.assertPackets(jbuffer, [None, None, None, None])

    def test_remove_audio_frame(self) -> None:
        """
        Audio jitter buffer.
        """
        jbuffer = JitterBuffer(capacity=16, prefetch=4)

        packet = RtpPacket(sequence_number=0, timestamp=1234)
        packet._data = b"0000"  # type: ignore
        pli_flag, frame = jbuffer.add(packet)
        self.assertIsNone(frame)

        packet = RtpPacket(sequence_number=1, timestamp=1235)
        packet._data = b"0001"  # type: ignore
        pli_flag, frame = jbuffer.add(packet)
        self.assertIsNone(frame)

        packet = RtpPacket(sequence_number=2, timestamp=1236)
        packet._data = b"0002"  # type: ignore
        pli_flag, frame = jbuffer.add(packet)
        self.assertIsNone(frame)

        packet = RtpPacket(sequence_number=3, timestamp=1237)
        packet._data = b"0003"  # type: ignore
        pli_flag, frame = jbuffer.add(packet)
        self.assertIsNone(frame)

        packet = RtpPacket(sequence_number=4, timestamp=1238)
        packet._data = b"0003"  # type: ignore
        pli_flag, frame = jbuffer.add(packet)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.data, b"0000")
        self.assertEqual(frame.timestamp, 1234)

        packet = RtpPacket(sequence_number=5, timestamp=1239)
        packet._data = b"0004"  # type: ignore
        pli_flag, frame = jbuffer.add(packet)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.data, b"0001")
        self.assertEqual(frame.timestamp, 1235)

    def test_remove_video_frame(self) -> None:
        """
        Video jitter buffer.
        """
        jbuffer = JitterBuffer(capacity=128, is_video=True)

        packet = RtpPacket(sequence_number=0, timestamp=1234)
        packet._data = b"0000"  # type: ignore
        pli_flag, frame = jbuffer.add(packet)
        self.assertIsNone(frame)

        packet = RtpPacket(sequence_number=1, timestamp=1234)
        packet._data = b"0001"  # type: ignore
        pli_flag, frame = jbuffer.add(packet)
        self.assertIsNone(frame)

        packet = RtpPacket(sequence_number=2, timestamp=1234)
        packet._data = b"0002"  # type: ignore
        pli_flag, frame = jbuffer.add(packet)
        self.assertIsNone(frame)

        packet = RtpPacket(sequence_number=3, timestamp=1235)
        packet._data = b"0003"  # type: ignore
        pli_flag, frame = jbuffer.add(packet)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.data, b"000000010002")
        self.assertEqual(frame.timestamp, 1234)

    def test_pli_flag(self) -> None:
        """
        Video jitter buffer.
        """
        jbuffer = JitterBuffer(capacity=128, is_video=True)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=2000, timestamp=1234))
        self.assertIsNone(frame)
        self.assertEqual(jbuffer._origin, 2000)
        self.assertFalse(pli_flag)

        # test_add_seq_too_low_reset for video (capacity >= 128)
        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertIsNone(frame)
        self.assertEqual(jbuffer._origin, 1)
        self.assertTrue(pli_flag)

        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=128, timestamp=1235))
        self.assertIsNone(frame)
        self.assertEqual(jbuffer._origin, 1)
        self.assertFalse(pli_flag)

        # test_add_seq_too_high_discard_one for video (capacity >= 128)
        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=129, timestamp=1235))
        self.assertIsNone(frame)
        self.assertEqual(jbuffer._origin, 128)
        self.assertTrue(pli_flag)

        # test_add_seq_too_high_reset for video (capacity >= 128)
        pli_flag, frame = jbuffer.add(RtpPacket(sequence_number=2000, timestamp=2345))
        self.assertIsNone(frame)
        self.assertEqual(jbuffer._origin, 2000)
        self.assertTrue(pli_flag)
