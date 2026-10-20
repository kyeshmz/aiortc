import fractions
import logging
import re
from collections.abc import Iterable, Iterator
from struct import pack, unpack_from
from typing import Optional, cast

import av
from av.frame import Frame
from av.packet import Packet
from av.video.codeccontext import VideoCodecContext
from av.video.frame import PictureType

from ..jitterbuffer import JitterFrame
from ..mediastreams import VIDEO_TIME_BASE, convert_timebase
from ..rtcrtpparameters import ParametersDict
from .base import Decoder, Encoder

logger = logging.getLogger(__name__)

DEFAULT_BITRATE = 1000000
MIN_BITRATE = 500000
MAX_BITRATE = 3000000
MAX_FRAME_RATE = 30
PACKET_MAX = 1300
DEFAULT_LEVEL = 93
H265_PARAMETERS: ParametersDict = {
    "profile-id": "1",
    "tier-flag": "0",
    "level-id": "93",
    "tx-mode": "SRST",
    "sprop-max-don-diff": "0",
}

NAL_TYPE_AP = 48
NAL_TYPE_FU = 49
START_CODE = b"\x00\x00\x00\x01"

# MaxLumaPs, MaxLumaSr, and main-tier MaxBR from H.265 Annex A.
LEVEL_LIMITS = {
    30: (36864, 552960, 128000),
    60: (122880, 3686400, 1500000),
    63: (245760, 7372800, 3000000),
    90: (552960, 16588800, 6000000),
    93: (983040, 33177600, 10000000),
}
LEVEL_IDS = (*LEVEL_LIMITS, 120, 123, 150, 153, 156, 180, 183, 186)


def parse_h265_parameters(parameters: ParametersDict) -> int:
    supported = {
        "profile-space": "0",
        "profile-id": "1",
        "tier-flag": "0",
        "profile-compatibility-indicator": "40000000",
        "interop-constraints": "b00000000000",
        "tx-mode": "SRST",
        "sprop-max-don-diff": "0",
    }
    for key, value in parameters.items():
        if key == "level-id":
            continue
        if key not in supported or str(value).lower() != supported[key].lower():
            raise ValueError(f"Unsupported HEVC parameter {key}={value}")

    level = str(parameters.get("level-id", DEFAULT_LEVEL))
    if not re.fullmatch(r"[0-9]{1,3}", level) or int(level) not in LEVEL_IDS:
        raise ValueError("Invalid HEVC level-id")
    return int(level)


def negotiate_h265_parameters(
    local: ParametersDict, remote: ParametersDict
) -> ParametersDict:
    level = min(parse_h265_parameters(local), parse_h265_parameters(remote))
    return {**local, "level-id": str(level)}


def _nal_type(data: bytes) -> int:
    if len(data) < 3:
        raise ValueError("HEVC NAL unit is too short")
    if data[0] & 0x80:
        raise ValueError("HEVC forbidden bit is set")
    if data[0] & 1 or data[1] & 0xF8:
        raise ValueError("HEVC multilayer streams are not supported")
    if not data[1] & 7:
        raise ValueError("HEVC temporal ID is zero")
    return (data[0] >> 1) & 0x3F


class H265PayloadDescriptor:
    def __init__(self, first_fragment: bool) -> None:
        self.first_fragment = first_fragment

    def __repr__(self) -> str:
        return f"H265PayloadDescriptor(FF={self.first_fragment})"

    @classmethod
    def parse(cls, data: bytes) -> tuple["H265PayloadDescriptor", bytes]:
        nal_type = _nal_type(data)
        if nal_type < NAL_TYPE_AP:
            return cls(first_fragment=True), START_CODE + data
        if nal_type == NAL_TYPE_AP:
            units = []
            pos = 2
            while pos < len(data):
                if pos + 2 > len(data):
                    raise ValueError("HEVC AP length field is truncated")
                size = unpack_from("!H", data, pos)[0]
                pos += 2
                if pos + size > len(data):
                    raise ValueError("HEVC AP data is truncated")
                unit = data[pos : pos + size]
                if _nal_type(unit) >= NAL_TYPE_AP:
                    raise ValueError("HEVC AP contains a nested packet")
                units.append(unit)
                pos += size
            if len(units) < 2:
                raise ValueError("HEVC AP must contain at least two NAL units")
            if data[1] != min(unit[1] for unit in units):
                raise ValueError("HEVC AP temporal ID is inconsistent")
            return cls(first_fragment=True), b"".join(START_CODE + u for u in units)
        if nal_type == NAL_TYPE_FU:
            if len(data) < 4:
                raise ValueError("HEVC FU payload is empty")
            start = bool(data[2] & 0x80)
            end = bool(data[2] & 0x40)
            original_type = data[2] & 0x3F
            if start and end:
                raise ValueError("HEVC FU start and end bits are both set")
            if original_type >= NAL_TYPE_AP:
                raise ValueError("HEVC FU contains a nested packet")
            prefix = START_CODE + bytes([original_type << 1, data[1]]) if start else b""
            return cls(first_fragment=start), prefix + data[3:]
        raise ValueError(f"HEVC NAL unit type {nal_type} is not supported")


def h265_depayload(payload: bytes) -> bytes:
    return H265PayloadDescriptor.parse(payload)[1]


class H265Decoder(Decoder):
    def __init__(self) -> None:
        codec = av.CodecContext.create("hevc", "r")
        assert isinstance(codec, VideoCodecContext)
        self.codec = codec
        self.codec.thread_count = 1

    def decode(self, encoded_frame: JitterFrame) -> list[Frame]:
        try:
            packet = av.Packet(encoded_frame.data)
            packet.pts = encoded_frame.timestamp
            packet.time_base = VIDEO_TIME_BASE
            return cast(list[Frame], self.codec.decode(packet))
        except av.FFmpegError as exc:
            logger.warning("H265Decoder() failed to decode, skipping packet: %s", exc)
            return []


class H265Encoder(Encoder):
    def __init__(self, level_id: int = DEFAULT_LEVEL) -> None:
        if level_id not in LEVEL_LIMITS:
            raise ValueError("Unsupported HEVC encoder level")
        self.codec: Optional[VideoCodecContext] = None
        self.level_id = level_id
        self.__target_bitrate = min(DEFAULT_BITRATE, LEVEL_LIMITS[level_id][2])

    @staticmethod
    def _split_bitstream(data: bytes) -> Iterator[bytes]:
        units = re.split(b"\x00{2,}\x01", data)
        if len(units) < 2 or units[0].strip(b"\x00"):
            raise ValueError("HEVC packet must contain an Annex B access unit")
        yield from units[1:]

    @staticmethod
    def _packetize(units: Iterable[bytes]) -> list[bytes]:
        payloads: list[bytes] = []
        pending: list[bytes] = []
        size = 2

        def flush() -> None:
            if len(pending) == 1:
                payloads.append(pending[0])
            elif pending:
                header = bytes([NAL_TYPE_AP << 1, min(unit[1] for unit in pending)])
                payloads.append(
                    header + b"".join(pack("!H", len(unit)) + unit for unit in pending)
                )
            pending.clear()

        for unit in units:
            nal_type = _nal_type(unit)
            if nal_type >= NAL_TYPE_AP:
                raise ValueError("HEVC packetizer requires NAL units")
            mixed_vcl_tid = any(
                previous[1] != unit[1] and (nal_type < 32 or previous[0] >> 1 < 32)
                for previous in pending
            )
            if size + 2 + len(unit) > PACKET_MAX or mixed_vcl_tid:
                flush()
                size = 2
            if len(unit) > PACKET_MAX:
                header = bytes([NAL_TYPE_FU << 1, unit[1]])
                for offset in range(2, len(unit), PACKET_MAX - 3):
                    end = min(offset + PACKET_MAX - 3, len(unit))
                    flags = (0x80 if offset == 2 else 0) | (
                        0x40 if end == len(unit) else 0
                    )
                    payloads.append(
                        header + bytes([flags | nal_type]) + unit[offset:end]
                    )
            else:
                pending.append(unit)
                size += 2 + len(unit)
        flush()
        return payloads

    def encode(
        self, frame: Frame, force_keyframe: bool = False
    ) -> tuple[list[bytes], int]:
        assert isinstance(frame, av.VideoFrame)
        max_pixels, max_samples, _ = LEVEL_LIMITS[self.level_id]
        pixels = frame.width * frame.height
        if (
            frame.width < 16
            or frame.height < 16
            or frame.width % 2
            or frame.height % 2
            or pixels > max_pixels
            or pixels * MAX_FRAME_RATE > max_samples
            or max(frame.width, frame.height) ** 2 > 8 * max_pixels
        ):
            raise ValueError(
                "Frame dimensions exceed the negotiated HEVC level at 30 fps"
            )

        if self.codec and (
            frame.width != self.codec.width
            or frame.height != self.codec.height
            or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate
            > 0.1
        ):
            self.codec = None

        frame.pict_type = PictureType.I if force_keyframe else PictureType.NONE
        if self.codec is None:
            codec = av.CodecContext.create("libx265", "w")
            assert isinstance(codec, VideoCodecContext)
            self.codec = codec
            self.codec.width = frame.width
            self.codec.height = frame.height
            self.codec.bit_rate = self.target_bitrate
            self.codec.pix_fmt = "yuv420p"
            self.codec.framerate = fractions.Fraction(MAX_FRAME_RATE, 1)
            self.codec.time_base = fractions.Fraction(1, MAX_FRAME_RATE)
            self.codec.thread_count = 1
            self.codec.profile = "Main"
            self.codec.options = {
                "preset": "veryfast",
                "tune": "zerolatency",
                "forced-idr": "1",
                "x265-params": (
                    "pools=none:frame-threads=1:wpp=0:log-level=error:"
                    "repeat-headers=1:annexb=1:bframes=0:open-gop=0:"
                    f"level-idc={self.level_id / 30:g}:high-tier=0:"
                    f"vbv-maxrate={self.target_bitrate // 1000}:"
                    f"vbv-bufsize={self.target_bitrate // 1000}"
                ),
            }

        data = b"".join(bytes(packet) for packet in self.codec.encode(frame))
        time_base = fractions.Fraction(
            frame.time_base.numerator, frame.time_base.denominator
        )
        timestamp = convert_timebase(frame.pts, time_base, VIDEO_TIME_BASE)
        return self._packetize(self._split_bitstream(data)) if data else [], timestamp

    def pack(self, packet: Packet) -> tuple[list[bytes], int]:
        assert isinstance(packet, av.Packet)
        time_base = fractions.Fraction(
            packet.time_base.numerator, packet.time_base.denominator
        )
        timestamp = convert_timebase(packet.pts, time_base, VIDEO_TIME_BASE)
        if packet.stream is not None:
            bitstream_filter = av.BitStreamFilterContext(
                b"hevc_mp4toannexb", packet.stream
            )
            data = b"".join(bytes(p) for p in bitstream_filter.filter(packet))
        else:
            data = bytes(packet)
        return self._packetize(self._split_bitstream(data)), timestamp

    @property
    def target_bitrate(self) -> int:
        return self.__target_bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        maximum = min(MAX_BITRATE, LEVEL_LIMITS[self.level_id][2])
        self.__target_bitrate = max(min(MIN_BITRATE, maximum), min(bitrate, maximum))
