from unittest import TestCase

from aiortc.rtcdtlstransport import RTCCertificate, RTCDtlsTransport
from aiortc.rtcicetransport import RTCIceGatherer, RTCIceTransport
from aiortc.rtcrtpparameters import RTCRtpCodecCapability
from aiortc.rtcrtpreceiver import RTCRtpReceiver
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.rtcrtptransceiver import RTCRtpTransceiver

from .utils import asynctest


class RTCRtpTransceiverTest(TestCase):
    @asynctest
    async def test_current_direction_transitions(self) -> None:
        ice = RTCIceTransport(RTCIceGatherer(iceServers=[]))
        transport = RTCDtlsTransport(ice, [RTCCertificate.generateCertificate()])
        directions = (
            ("sendrecv", True, True),
            ("sendonly", True, False),
            ("recvonly", False, True),
            ("inactive", False, False),
        )
        try:
            for previous, sender_before, receiver_before in directions:
                for direction, sender_enabled, receiver_enabled in directions:
                    with self.subTest(previous=previous, direction=direction):
                        sender = RTCRtpSender("audio", transport)
                        receiver = RTCRtpReceiver("audio", transport)
                        transceiver = RTCRtpTransceiver("audio", receiver, sender)
                        try:
                            self.assertIsNone(transceiver.currentDirection)
                            transceiver._setCurrentDirection(previous)
                            self.assertEqual(transceiver.currentDirection, previous)
                            self.assertIs(sender._enabled, sender_before)
                            self.assertIs(receiver._enabled, receiver_before)
                            transceiver._setCurrentDirection(direction)
                            self.assertEqual(transceiver.currentDirection, direction)
                            self.assertIs(sender._enabled, sender_enabled)
                            self.assertIs(receiver._enabled, receiver_enabled)
                            self.assertEqual(transceiver.direction, "sendrecv")
                        finally:
                            await transceiver.stop()
        finally:
            await transport.stop()
            await ice.stop()

    def test_codec_preferences(self) -> None:
        transceiver = RTCRtpTransceiver("audio", None, None)
        self.assertEqual(transceiver._preferred_codecs, [])

        # set empty preferences
        transceiver.setCodecPreferences([])
        self.assertEqual(transceiver._preferred_codecs, [])

        # set single codec
        transceiver.setCodecPreferences(
            [RTCRtpCodecCapability(mimeType="audio/PCMU", clockRate=8000, channels=1)]
        )
        self.assertEqual(
            transceiver._preferred_codecs,
            [RTCRtpCodecCapability(mimeType="audio/PCMU", clockRate=8000, channels=1)],
        )

        # set single codec (duplicated)
        transceiver.setCodecPreferences(
            [
                RTCRtpCodecCapability(
                    mimeType="audio/PCMU", clockRate=8000, channels=1
                ),
                RTCRtpCodecCapability(
                    mimeType="audio/PCMU", clockRate=8000, channels=1
                ),
            ]
        )
        self.assertEqual(
            transceiver._preferred_codecs,
            [RTCRtpCodecCapability(mimeType="audio/PCMU", clockRate=8000, channels=1)],
        )

        # set single codec (invalid)
        with self.assertRaises(ValueError) as cm:
            transceiver.setCodecPreferences(
                [
                    RTCRtpCodecCapability(
                        mimeType="audio/bogus", clockRate=8000, channels=1
                    )
                ]
            )
        self.assertEqual(str(cm.exception), "Codec is not in capabilities")
