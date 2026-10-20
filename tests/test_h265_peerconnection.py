import asyncio
import fractions
import os
import tempfile
from struct import unpack_from
from unittest.mock import patch

import av
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.codecs.h265 import H265Encoder
from aiortc.contrib.media import MediaPlayer
from aiortc.exceptions import OperationError
from aiortc.mediastreams import MediaStreamTrack
from aiortc.rtcrtpparameters import ParametersDict
from aiortc.rtp import RtpPacket, is_rtcp
from aiortc.sdp import SessionDescription
from av.video.stream import VideoStream

from .codecs import CodecTestCase
from .utils import asynctest


class H265PeerConnectionTest(CodecTestCase):
    def create_hevc_file(self, path: str) -> None:
        with av.open(path, "w") as container:
            stream = self.ensureIsInstance(
                container.add_stream("libx265", rate=30), VideoStream
            )
            stream.width = 320
            stream.height = 240
            stream.pix_fmt = "yuv420p"
            stream.codec_context.thread_count = 1
            stream.codec_context.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "x265-params": (
                    "pools=none:frame-threads=1:wpp=0:bframes=0:log-level=error"
                ),
            }
            for frame in self.create_video_frames(320, 240, count=12):
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

    def prefer_hevc(self, pc: RTCPeerConnection) -> None:
        capabilities = RTCRtpSender.getCapabilities("video")
        pc.getTransceivers()[0].setCodecPreferences(
            [
                c
                for c in capabilities.codecs
                if c.mimeType in ("video/H265", "video/rtx")
            ]
        )

    async def connect(self, pc1: RTCPeerConnection, pc2: RTCPeerConnection) -> None:
        await pc1.setLocalDescription(await pc1.createOffer())
        await pc2.setRemoteDescription(pc1.localDescription)
        await pc2.setLocalDescription(await pc2.createAnswer())
        await pc1.setRemoteDescription(pc2.localDescription)
        for pc in (pc1, pc2):
            description = SessionDescription.parse(pc.localDescription.sdp)
            codecs = description.media[0].rtp.codecs
            self.assertEqual([c.mimeType for c in codecs], ["video/H265", "video/rtx"])
            self.assertEqual(
                codecs[0].parameters,
                {
                    "profile-id": "1",
                    "tier-flag": "0",
                    "level-id": "93",
                    "tx-mode": "SRST",
                    "sprop-max-don-diff": "0",
                },
            )
            self.assertEqual(codecs[1].parameters, {"apt": 103})
            self.assertTrue(description.media[0].ice_candidates)
            self.assertTrue(description.media[0].dtls.fingerprints)

    async def receive_frames(
        self, track: MediaStreamTrack, width: int, height: int
    ) -> None:
        frames = [await asyncio.wait_for(track.recv(), timeout=10) for _ in range(3)]
        for i, frame in enumerate(frames):
            assert isinstance(frame, av.VideoFrame)
            self.assertEqual((frame.width, frame.height), (width, height))
            self.assertEqual(frame.pts, i * 3000)
            self.assertEqual(frame.time_base, fractions.Fraction(1, 90000))

    @asynctest
    async def test_bidirectional_decoded_frames(self) -> None:
        pc1 = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        pc2 = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            for pc in (pc1, pc2):
                pc.addTrack(VideoStreamTrack())
                self.prefer_hevc(pc)
            await self.connect(pc1, pc2)
            await asyncio.gather(
                self.receive_frames(pc1.getReceivers()[0].track, 640, 480),
                self.receive_frames(pc2.getReceivers()[0].track, 640, 480),
            )
            for pc in (pc1, pc2):
                self.assertEqual(pc.connectionState, "connected")
                self.assertEqual(pc.iceConnectionState, "completed")
                self.assertEqual(pc.getSenders()[0].transport.state, "connected")
                stats = await pc.getStats()
                inbound = [s for s in stats.values() if s.type == "inbound-rtp"]
                outbound = [s for s in stats.values() if s.type == "outbound-rtp"]
                self.assertEqual(len(inbound), 1)
                self.assertEqual(len(outbound), 1)
                self.assertGreaterEqual(inbound[0].packetsReceived, 3)
                self.assertGreaterEqual(outbound[0].packetsSent, 3)
        finally:
            await asyncio.gather(pc1.close(), pc2.close())

    @asynctest
    async def test_rtx_recovers_lost_hevc_packet(self) -> None:
        pc1 = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        pc2 = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            pc1.addTrack(VideoStreamTrack())
            pc2.addTransceiver("video", direction="recvonly")
            for pc in (pc1, pc2):
                self.prefer_hevc(pc)
            transport = pc1.getSenders()[0].transport
            send_rtp = transport._send_rtp
            first_timestamp = None
            dropped: list[RtpPacket] = []
            retransmissions: list[RtpPacket] = []

            async def drop_one_packet(data: bytes) -> None:
                nonlocal first_timestamp
                if not is_rtcp(data):
                    packet = RtpPacket.parse(data)
                    if packet.payload_type == 103:
                        if first_timestamp is None:
                            first_timestamp = packet.timestamp
                        elif packet.timestamp != first_timestamp and not dropped:
                            dropped.append(packet)
                            return
                    elif packet.payload_type == 104:
                        retransmissions.append(packet)
                await send_rtp(data)

            with patch.object(transport, "_send_rtp", new=drop_one_packet):
                await self.connect(pc1, pc2)
                await self.receive_frames(pc2.getReceivers()[0].track, 640, 480)
            self.assertEqual(len(dropped), 1)
            self.assertGreaterEqual(len(retransmissions), 1)
            recovered = retransmissions[0]
            self.assertEqual(
                unpack_from("!H", recovered.payload)[0], dropped[0].sequence_number
            )
            self.assertEqual(recovered.payload[2:], dropped[0].payload)
            self.assertEqual(recovered.timestamp, dropped[0].timestamp)
            self.assertNotEqual(recovered.ssrc, dropped[0].ssrc)
        finally:
            await asyncio.gather(pc1.close(), pc2.close())

    @asynctest
    async def test_mediaplayer_compressed_passthrough(self) -> None:
        for extension in ("mp4", "ts"):
            with (
                self.subTest(extension=extension),
                tempfile.TemporaryDirectory() as tmp,
            ):
                path = os.path.join(tmp, "hevc." + extension)
                self.create_hevc_file(path)
                player = MediaPlayer(path, decode=False)
                self.assertIsNotNone(player.video)
                self.assertIsNone(player.audio)
                pc1 = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                pc2 = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                try:
                    pc1.addTrack(player.video)
                    self.prefer_hevc(pc1)
                    pc2.addTransceiver("video", direction="recvonly")
                    self.prefer_hevc(pc2)
                    with patch.object(
                        H265Encoder,
                        "encode",
                        side_effect=AssertionError(
                            "Passthrough must not encode raw frames"
                        ),
                    ):
                        await self.connect(pc1, pc2)
                        await self.receive_frames(pc2.getReceivers()[0].track, 320, 240)
                    self.assertEqual(pc2.connectionState, "connected")
                finally:
                    await asyncio.gather(pc1.close(), pc2.close())
                    player.video.stop()

    @asynctest
    async def test_negotiate_lower_level(self) -> None:
        offerer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        answerer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            for pc in (offerer, answerer):
                pc.addTransceiver("video", direction="recvonly")
                self.prefer_hevc(pc)
            offer = SessionDescription.parse((await offerer.createOffer()).sdp)
            offer.media[0].rtp.codecs[0].parameters["level-id"] = "60"
            await answerer.setRemoteDescription(
                RTCSessionDescription(sdp=str(offer), type="offer")
            )
            answer = SessionDescription.parse((await answerer.createAnswer()).sdp)
            self.assertEqual(
                answer.media[0].rtp.codecs[0].parameters,
                {
                    "profile-id": "1",
                    "tier-flag": "0",
                    "level-id": "60",
                    "tx-mode": "SRST",
                    "sprop-max-don-diff": "0",
                },
            )
            self.assertEqual(answer.media[0].rtp.codecs[1].parameters, {"apt": 103})
        finally:
            await asyncio.gather(offerer.close(), answerer.close())

    @asynctest
    async def test_reject_unsupported_sdp(self) -> None:
        cases: list[ParametersDict] = [
            {"profile-id": "2"},
            {"tier-flag": "1"},
            {"sprop-max-don-diff": "1"},
            {"tx-mode": "MRST"},
            {"tx-mode": "MRMT"},
            {"level-id": "invalid"},
        ]
        for parameters in cases:
            with self.subTest(parameters=parameters):
                offerer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                answerer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                try:
                    offerer.addTrack(VideoStreamTrack())
                    self.prefer_hevc(offerer)
                    offer = await offerer.createOffer()
                    parsed = SessionDescription.parse(offer.sdp)
                    parsed.media[0].rtp.codecs[0].parameters.update(parameters)
                    with self.assertRaisesRegex(
                        OperationError,
                        "Failed to set remote video description send parameters",
                    ):
                        await answerer.setRemoteDescription(
                            RTCSessionDescription(sdp=str(parsed), type="offer")
                        )
                finally:
                    await asyncio.gather(offerer.close(), answerer.close())
