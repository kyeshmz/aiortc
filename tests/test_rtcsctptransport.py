import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import fields
from itertools import permutations
from typing import TypeVar, cast
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from aiortc.exceptions import InvalidStateError
from aiortc.rtcdatachannel import RTCDataChannel, RTCDataChannelParameters
from aiortc.rtcsctptransport import (
    LOCAL_MAX_MESSAGE_SIZE,
    REMOTE_DEFAULT_MAX_MESSAGE_SIZE,
    SCTP_DATA_FIRST_FRAG,
    SCTP_DATA_LAST_FRAG,
    SCTP_DATA_UNORDERED,
    SCTP_MAX_PACKET_LENGTH,
    USERDATA_MAX_LENGTH,
    AbortChunk,
    Chunk,
    CookieAckChunk,
    CookieEchoChunk,
    DataChunk,
    ErrorChunk,
    ForwardTsnChunk,
    HeartbeatAckChunk,
    HeartbeatChunk,
    InboundStream,
    InitAckChunk,
    InitChunk,
    ReconfigChunk,
    RTCSctpCapabilities,
    RTCSctpTransport,
    SackChunk,
    ShutdownAckChunk,
    ShutdownChunk,
    ShutdownCompleteChunk,
    StreamAddOutgoingParam,
    StreamResetOutgoingParam,
    StreamResetResponseParam,
    parse_packet,
    serialize_packet,
    tsn_minus_one,
    tsn_plus_one,
)

from .utils import (
    ClosedDtlsTransport,
    asynctest,
    dummy_dtls_transport_pair,
    load,
    set_loss_pattern,
)

T = TypeVar("T")


@contextlib.asynccontextmanager
async def client_and_server() -> AsyncGenerator[
    tuple[RTCSctpTransport, RTCSctpTransport], None
]:
    # The controlled side resolves to the DTLS client in auto mode.
    async with dummy_dtls_transport_pair() as (server_transport, client_transport):
        client = RTCSctpTransport(client_transport)
        server = RTCSctpTransport(server_transport)
        assert client.is_server is False
        assert server.is_server is True

        try:
            yield client, server
        finally:
            await client.stop()
            await server.stop()
            assert client._association_state == RTCSctpTransport.State.CLOSED
            assert client.state == "closed"
            assert server._association_state == RTCSctpTransport.State.CLOSED
            assert server.state == "closed"


@contextlib.asynccontextmanager
async def client_standalone() -> AsyncGenerator[RTCSctpTransport, None]:
    async with dummy_dtls_transport_pair() as (_, client_transport):
        client = RTCSctpTransport(client_transport)
        assert client.is_server is False

        try:
            yield client
        finally:
            await client.stop()


async def noop_init() -> None:
    pass


async def noop_send_chunk(chunk: Chunk) -> None:
    pass


def outstanding_tsns(client: RTCSctpTransport) -> list[int]:
    return [chunk.tsn for chunk in client._sent_queue]


def queued_tsns(client: RTCSctpTransport) -> list[int]:
    return [chunk.tsn for chunk in client._outbound_queue]


def track_channels(transport: RTCSctpTransport) -> list[RTCDataChannel]:
    channels = []

    @transport.on("datachannel")
    def on_datachannel(channel: RTCDataChannel) -> None:
        channels.append(channel)

    return channels


async def wait_for_outcome(client: RTCSctpTransport, server: RTCSctpTransport) -> None:
    final = [RTCSctpTransport.State.ESTABLISHED, RTCSctpTransport.State.CLOSED]
    for i in range(100):
        if client._association_state in final and server._association_state in final:
            break
        await asyncio.sleep(0.1)


class SctpPacketTest(TestCase):
    def test_unknown_chunk_is_ignored(self) -> None:
        unknown = Chunk(body=b"extension")
        unknown.type = 255
        packet = serialize_packet(5000, 5001, 1234, unknown)
        self.assertEqual(parse_packet(packet), (5000, 5001, 1234, []))

    def roundtrip_packet(self, data: bytes, cls: type[T]) -> T:
        source_port, destination_port, verification_tag, chunks = parse_packet(data)
        self.assertEqual(source_port, 5000)
        self.assertEqual(destination_port, 5000)
        self.assertEqual(len(chunks), 1)
        output = serialize_packet(
            source_port, destination_port, verification_tag, chunks[0]
        )
        self.assertEqual(output, data)

        self.assertIsInstance(chunks[0], cls)
        return cast(T, chunks[0])

    def test_parse_init(self) -> None:
        data = load("sctp_init.bin")
        chunk = self.roundtrip_packet(data, InitChunk)

        self.assertEqual(chunk.type, 1)
        self.assertEqual(chunk.flags, 0)
        self.assertEqual(len(chunk.body), 82)
        self.assertEqual(repr(chunk), "InitChunk(flags=0)")

    def test_parse_init_invalid_checksum(self) -> None:
        data = load("sctp_init.bin")
        data = data[0:8] + b"\x01\x02\x03\x04" + data[12:]
        with self.assertRaises(ValueError) as cm:
            self.roundtrip_packet(data, None)
        self.assertEqual(str(cm.exception), "SCTP packet has invalid checksum")

    def test_parse_init_truncated_packet_header(self) -> None:
        data = load("sctp_init.bin")[0:10]
        with self.assertRaises(ValueError) as cm:
            self.roundtrip_packet(data, None)
        self.assertEqual(str(cm.exception), "SCTP packet length is less than 16 bytes")

    def test_parse_cookie_echo(self) -> None:
        data = load("sctp_cookie_echo.bin")
        chunk = self.roundtrip_packet(data, CookieEchoChunk)

        self.assertEqual(chunk.type, 10)
        self.assertEqual(chunk.flags, 0)
        self.assertEqual(len(chunk.body), 8)

    def test_parse_abort(self) -> None:
        data = load("sctp_abort.bin")
        chunk = self.roundtrip_packet(data, AbortChunk)

        self.assertEqual(chunk.type, 6)
        self.assertEqual(chunk.flags, 0)
        self.assertEqual(
            chunk.params, [(13, b"Expected B-bit for TSN=4ce1f17f, SID=0001, SSN=0000")]
        )

    def test_parse_data(self) -> None:
        data = load("sctp_data.bin")
        chunk = self.roundtrip_packet(data, DataChunk)

        self.assertEqual(chunk.type, 0)
        self.assertEqual(chunk.flags, 3)
        self.assertEqual(chunk.tsn, 2584679421)
        self.assertEqual(chunk.stream_id, 1)
        self.assertEqual(chunk.stream_seq, 1)
        self.assertEqual(chunk.protocol, 51)
        self.assertEqual(chunk.user_data, b"ping")
        self.assertEqual(
            repr(chunk), "DataChunk(flags=3, tsn=2584679421, stream_id=1, stream_seq=1)"
        )

    def test_serialize_data_max_length(self) -> None:
        """
        A DATA chunk carrying the maximum amount of user data must fit
        within the maximum SCTP packet size.
        """
        chunk = DataChunk()
        chunk.user_data = b"M" * USERDATA_MAX_LENGTH
        packet = serialize_packet(5000, 5000, 0, chunk)
        self.assertLessEqual(len(packet), SCTP_MAX_PACKET_LENGTH)

    def test_parse_data_padding(self) -> None:
        data = load("sctp_data_padding.bin")
        chunk = self.roundtrip_packet(data, DataChunk)

        self.assertEqual(chunk.type, 0)
        self.assertEqual(chunk.flags, 3)
        self.assertEqual(chunk.tsn, 2584679421)
        self.assertEqual(chunk.stream_id, 1)
        self.assertEqual(chunk.stream_seq, 1)
        self.assertEqual(chunk.protocol, 51)
        self.assertEqual(chunk.user_data, b"M")
        self.assertEqual(
            repr(chunk), "DataChunk(flags=3, tsn=2584679421, stream_id=1, stream_seq=1)"
        )

    def test_parse_data_truncated_chunk_value(self) -> None:
        data = load("sctp_data_truncated.bin")
        with self.assertRaises(ValueError) as cm:
            self.roundtrip_packet(data, None)
        self.assertEqual(
            str(cm.exception), "SCTP chunk has an invalid length of 8 bytes"
        )

    def test_parse_data_zero_length(self) -> None:
        data = load("sctp_data_zero_length.bin")
        with self.assertRaises(ValueError) as cm:
            self.roundtrip_packet(data, None)
        self.assertEqual(
            str(cm.exception), "SCTP chunk has an invalid length of 0 bytes"
        )

    def test_parse_error(self) -> None:
        data = load("sctp_error.bin")
        chunk = self.roundtrip_packet(data, ErrorChunk)

        self.assertEqual(chunk.type, 9)
        self.assertEqual(chunk.flags, 0)
        self.assertEqual(chunk.params, [(1, b"\x30\x39\x00\x00")])

    def test_parse_forward_tsn(self) -> None:
        data = load("sctp_forward_tsn.bin")
        chunk = self.roundtrip_packet(data, ForwardTsnChunk)

        self.assertEqual(chunk.type, 192)
        self.assertEqual(chunk.flags, 0)
        self.assertEqual(chunk.cumulative_tsn, 1234)
        self.assertEqual(chunk.streams, [(12, 34)])
        self.assertEqual(
            repr(chunk), "ForwardTsnChunk(cumulative_tsn=1234, streams=[(12, 34)])"
        )

    def test_parse_heartbeat(self) -> None:
        data = load("sctp_heartbeat.bin")
        chunk = self.roundtrip_packet(data, HeartbeatChunk)

        self.assertEqual(chunk.type, 4)
        self.assertEqual(chunk.flags, 0)
        self.assertEqual(
            chunk.params,
            [
                (
                    1,
                    b"\xb5o\xaaZvZ\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00{\x10\x00\x00"
                    b"\x004\xeb\x07F\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
                )
            ],
        )

    def test_parse_reconfig_reset_out(self) -> None:
        data = load("sctp_reconfig_reset_out.bin")
        chunk = self.roundtrip_packet(data, ReconfigChunk)

        self.assertEqual(chunk.type, 130)
        self.assertEqual(chunk.flags, 0)
        self.assertEqual(
            chunk.params, [(13, b"\x8b\xd8\n[\xe4\x8b\xecs\x8b\xd8\n^\x00\x01")]
        )

        # Outgoing SSN Reset Request Parameter
        param_data = chunk.params[0][1]
        param = StreamResetOutgoingParam.parse(param_data)
        self.assertEqual(param.request_sequence, 2346191451)
        self.assertEqual(param.response_sequence, 3834375283)
        self.assertEqual(param.last_tsn, 2346191454)
        self.assertEqual(param.streams, [1])
        self.assertEqual(bytes(param), param_data)

    def test_parse_reconfig_add_out(self) -> None:
        data = load("sctp_reconfig_add_out.bin")
        chunk = self.roundtrip_packet(data, ReconfigChunk)

        self.assertEqual(chunk.type, 130)
        self.assertEqual(chunk.flags, 0)
        self.assertEqual(chunk.params, [(17, b"\xca\x02\xf60\x00\x10\x00\x00")])

        # Add Outgoing Streams Request Parameter
        param_data = chunk.params[0][1]
        param = StreamAddOutgoingParam.parse(param_data)
        self.assertEqual(param.request_sequence, 3389191728)
        self.assertEqual(param.new_streams, 16)
        self.assertEqual(bytes(param), param_data)

    def test_parse_reconfig_response(self) -> None:
        data = load("sctp_reconfig_response.bin")
        chunk = self.roundtrip_packet(data, ReconfigChunk)

        self.assertEqual(chunk.type, 130)
        self.assertEqual(chunk.flags, 0)
        self.assertEqual(chunk.params, [(16, b"\x91S\x1fT\x00\x00\x00\x01")])

        # Re-configuration Response Parameter
        param_data = chunk.params[0][1]
        param = StreamResetResponseParam.parse(param_data)
        self.assertEqual(param.response_sequence, 2438143828)
        self.assertEqual(param.result, 1)
        self.assertEqual(bytes(param), param_data)

    def test_reconfig_parameter_dataclass_contracts(self) -> None:
        cases = (
            (
                StreamResetOutgoingParam(1, 2, 3, [4, 5]),
                bytes.fromhex("00000001000000020000000300040005"),
                ("request_sequence", "response_sequence", "last_tsn", "streams"),
                "StreamResetOutgoingParam(request_sequence=1, response_sequence=2, "
                "last_tsn=3, streams=[4, 5])",
            ),
            (
                StreamResetResponseParam(1, 2),
                bytes.fromhex("0000000100000002"),
                ("response_sequence", "result"),
                "StreamResetResponseParam(response_sequence=1, result=2)",
            ),
            (
                StreamAddOutgoingParam(1, 2),
                bytes.fromhex("0000000100020000"),
                ("request_sequence", "new_streams"),
                "StreamAddOutgoingParam(request_sequence=1, new_streams=2)",
            ),
        )
        for param, wire, field_names, expected_repr in cases:
            with self.subTest(parameter=type(param).__name__):
                self.assertEqual(bytes(param), wire)
                self.assertEqual(type(param).parse(wire), param)
                self.assertEqual(
                    tuple(field.name for field in fields(param)), field_names
                )
                self.assertEqual(repr(param), expected_repr)
        first = StreamResetOutgoingParam(1, 2, 3)
        second = StreamResetOutgoingParam(1, 2, 3)
        first.streams.append(4)
        self.assertEqual(second.streams, [])
        self.assertNotEqual(first, second)

    def test_parse_sack(self) -> None:
        data = load("sctp_sack.bin")
        chunk = self.roundtrip_packet(data, SackChunk)

        self.assertEqual(chunk.type, 3)
        self.assertEqual(chunk.flags, 0)
        self.assertEqual(chunk.cumulative_tsn, 2222939037)
        self.assertEqual(chunk.gaps, [(2, 2), (4, 4)])
        self.assertEqual(chunk.duplicates, [2222939041])
        self.assertEqual(
            repr(chunk),
            "SackChunk(flags=0, advertised_rwnd=128160, cumulative_tsn=2222939037, "
            "gaps=[(2, 2), (4, 4)])",
        )

    def test_parse_shutdown(self) -> None:
        data = load("sctp_shutdown.bin")
        chunk = self.roundtrip_packet(data, ShutdownChunk)

        self.assertEqual(
            repr(chunk), "ShutdownChunk(flags=0, cumulative_tsn=2696426712)"
        )
        self.assertEqual(chunk.type, 7)
        self.assertEqual(chunk.flags, 0)
        self.assertEqual(chunk.cumulative_tsn, 2696426712)

    def test_parse_shutdown_ack(self) -> None:
        data = load("sctp_shutdown_ack.bin")
        chunk = self.roundtrip_packet(data, ShutdownAckChunk)

        self.assertEqual(chunk.type, 8)


class ChunkFactory:
    def __init__(self, tsn: int = 1) -> None:
        self.tsn = tsn
        self.stream_seq = 0

    def create(self, frags: list[bytes], ordered: bool = True) -> list[DataChunk]:
        chunks = []

        for i, frag in enumerate(frags):
            flags = 0
            if not ordered:
                flags |= SCTP_DATA_UNORDERED
            if i == 0:
                flags |= SCTP_DATA_FIRST_FRAG
            if i == len(frags) - 1:
                flags |= SCTP_DATA_LAST_FRAG

            chunk = DataChunk(flags=flags)
            chunk.protocol = 123
            chunk.stream_id = 456
            if ordered:
                chunk.stream_seq = self.stream_seq
            chunk.tsn = self.tsn
            chunk.user_data = frag
            chunks.append(chunk)

            self.tsn += 1

        if ordered:
            self.stream_seq += 1

        return chunks


class SctpStreamTest(TestCase):
    def setUp(self) -> None:
        self.factory = ChunkFactory()

    def test_duplicate(self) -> None:
        stream = InboundStream()
        chunks = self.factory.create([b"foo", b"bar", b"baz"])

        # feed first chunk
        stream.add_chunk(chunks[0])
        self.assertEqual(stream.reassembly, [chunks[0]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[0]])
        self.assertEqual(stream.sequence_number, 0)

        # feed first chunk again
        with self.assertRaises(AssertionError) as cm:
            stream.add_chunk(chunks[0])
        self.assertEqual(str(cm.exception), "duplicate chunk in reassembly")

    def test_fragment_permutations_across_serial_rollover(self) -> None:
        for tsns in (
            (0, 1, 2, 3),
            (0x7FFFFFFE, 0x7FFFFFFF, 0x80000000, 0x80000001),
            (0xFFFFFFFE, 0xFFFFFFFF, 0, 1),
        ):
            for order in permutations(range(4)):
                with self.subTest(tsns=tsns, order=order):
                    stream = InboundStream()
                    chunks = ChunkFactory().create([b"one", b"two", b"three", b"four"])
                    for chunk, tsn in zip(chunks, tsns):
                        chunk.tsn = tsn
                    for index in order:
                        stream.add_chunk(chunks[index])
                        with self.assertRaisesRegex(
                            AssertionError, "duplicate chunk in reassembly"
                        ):
                            stream.add_chunk(chunks[index])
                    self.assertEqual(
                        list(stream.pop_messages()), [(456, 123, b"onetwothreefour")]
                    )
                    self.assertEqual(stream.reassembly, [])
                    self.assertEqual(stream.sequence_number, 1)

    def test_whole_in_order(self) -> None:
        stream = InboundStream()
        chunks = self.factory.create([b"foo"]) + self.factory.create([b"bar"])

        # feed first unfragmented
        stream.add_chunk(chunks[0])
        self.assertEqual(stream.reassembly, [chunks[0]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [(456, 123, b"foo")])
        self.assertEqual(stream.reassembly, [])
        self.assertEqual(stream.sequence_number, 1)

        # feed second unfragmented
        stream.add_chunk(chunks[1])
        self.assertEqual(stream.reassembly, [chunks[1]])
        self.assertEqual(stream.sequence_number, 1)

        self.assertEqual(list(stream.pop_messages()), [(456, 123, b"bar")])
        self.assertEqual(stream.reassembly, [])
        self.assertEqual(stream.sequence_number, 2)

    def test_whole_out_of_order(self) -> None:
        stream = InboundStream()
        chunks = (
            self.factory.create([b"foo"])
            + self.factory.create([b"bar"])
            + self.factory.create([b"baz", b"qux"])
        )

        # feed second unfragmented
        stream.add_chunk(chunks[1])
        self.assertEqual(stream.reassembly, [chunks[1]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[1]])
        self.assertEqual(stream.sequence_number, 0)

        # feed third partial
        stream.add_chunk(chunks[2])
        self.assertEqual(stream.reassembly, [chunks[1], chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[1], chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        # feed first unfragmented
        stream.add_chunk(chunks[0])
        self.assertEqual(stream.reassembly, [chunks[0], chunks[1], chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(
            list(stream.pop_messages()), [(456, 123, b"foo"), (456, 123, b"bar")]
        )
        self.assertEqual(stream.reassembly, [chunks[2]])
        self.assertEqual(stream.sequence_number, 2)

    def test_fragments_in_order(self) -> None:
        stream = InboundStream()
        chunks = self.factory.create([b"foo", b"bar", b"baz"])

        # feed first chunk
        stream.add_chunk(chunks[0])
        self.assertEqual(stream.reassembly, [chunks[0]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[0]])
        self.assertEqual(stream.sequence_number, 0)

        # feed second chunk
        stream.add_chunk(chunks[1])
        self.assertEqual(stream.reassembly, [chunks[0], chunks[1]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[0], chunks[1]])
        self.assertEqual(stream.sequence_number, 0)

        # feed third chunk
        stream.add_chunk(chunks[2])
        self.assertEqual(stream.reassembly, [chunks[0], chunks[1], chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [(456, 123, b"foobarbaz")])
        self.assertEqual(stream.reassembly, [])
        self.assertEqual(stream.sequence_number, 1)

    def test_fragments_out_of_order(self) -> None:
        stream = InboundStream()
        chunks = self.factory.create([b"foo", b"bar", b"baz"])

        # feed third chunk
        stream.add_chunk(chunks[2])
        self.assertEqual(stream.reassembly, [chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        # feed first chunk
        stream.add_chunk(chunks[0])
        self.assertEqual(stream.reassembly, [chunks[0], chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[0], chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        # feed second chunk
        stream.add_chunk(chunks[1])
        self.assertEqual(stream.reassembly, [chunks[0], chunks[1], chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [(456, 123, b"foobarbaz")])
        self.assertEqual(stream.reassembly, [])
        self.assertEqual(stream.sequence_number, 1)

    def test_unordered_no_fragments(self) -> None:
        stream = InboundStream()
        chunks = (
            self.factory.create([b"foo"], ordered=False)
            + self.factory.create([b"bar"], ordered=False)
            + self.factory.create([b"baz"], ordered=False)
        )

        # feed second unfragmented
        stream.add_chunk(chunks[1])
        self.assertEqual(stream.reassembly, [chunks[1]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [(456, 123, b"bar")])
        self.assertEqual(stream.reassembly, [])
        self.assertEqual(stream.sequence_number, 0)

        # feed third unfragmented
        stream.add_chunk(chunks[2])
        self.assertEqual(stream.reassembly, [chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [(456, 123, b"baz")])
        self.assertEqual(stream.reassembly, [])
        self.assertEqual(stream.sequence_number, 0)

        # feed first unfragmented
        stream.add_chunk(chunks[0])
        self.assertEqual(stream.reassembly, [chunks[0]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [(456, 123, b"foo")])
        self.assertEqual(stream.reassembly, [])
        self.assertEqual(stream.sequence_number, 0)

    def test_unordered_with_fragments(self) -> None:
        stream = InboundStream()
        chunks = (
            self.factory.create([b"foo", b"bar"], ordered=False)
            + self.factory.create([b"baz"], ordered=False)
            + self.factory.create([b"qux", b"quux", b"corge"], ordered=False)
        )

        # feed second fragment of first message
        stream.add_chunk(chunks[1])
        self.assertEqual(stream.reassembly, [chunks[1]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[1]])
        self.assertEqual(stream.sequence_number, 0)

        # feed second message
        stream.add_chunk(chunks[2])
        self.assertEqual(stream.reassembly, [chunks[1], chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [(456, 123, b"baz")])
        self.assertEqual(stream.reassembly, [chunks[1]])
        self.assertEqual(stream.sequence_number, 0)

        # feed first fragment of third message
        stream.add_chunk(chunks[3])
        self.assertEqual(stream.reassembly, [chunks[1], chunks[3]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[1], chunks[3]])
        self.assertEqual(stream.sequence_number, 0)

        # feed third fragment of third message
        stream.add_chunk(chunks[5])
        self.assertEqual(stream.reassembly, [chunks[1], chunks[3], chunks[5]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[1], chunks[3], chunks[5]])
        self.assertEqual(stream.sequence_number, 0)

        # feed second fragment of third message
        stream.add_chunk(chunks[4])
        self.assertEqual(
            stream.reassembly, [chunks[1], chunks[3], chunks[4], chunks[5]]
        )
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [(456, 123, b"quxquuxcorge")])
        self.assertEqual(stream.reassembly, [chunks[1]])
        self.assertEqual(stream.sequence_number, 0)

        # feed first fragment of first message
        stream.add_chunk(chunks[0])
        self.assertEqual(stream.reassembly, [chunks[0], chunks[1]])
        self.assertEqual(stream.sequence_number, 0)

        self.assertEqual(list(stream.pop_messages()), [(456, 123, b"foobar")])
        self.assertEqual(stream.reassembly, [])
        self.assertEqual(stream.sequence_number, 0)

    def test_prune_chunks(self) -> None:
        stream = InboundStream()
        factory = ChunkFactory(tsn=100)
        chunks = factory.create([b"foo", b"bar"]) + factory.create([b"baz", b"qux"])

        for i in [1, 2]:
            stream.add_chunk(chunks[i])
            self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[1], chunks[2]])
        self.assertEqual(stream.sequence_number, 0)

        stream.sequence_number = 2
        self.assertEqual(list(stream.pop_messages()), [])
        self.assertEqual(stream.reassembly, [chunks[1], chunks[2]])
        self.assertEqual(stream.sequence_number, 2)

        self.assertEqual(stream.prune_chunks(101), 3)
        self.assertEqual(stream.reassembly, [chunks[2]])
        self.assertEqual(stream.sequence_number, 2)
        self.assertEqual(stream.prune_chunks(102), 3)
        self.assertEqual(stream.reassembly, [])
        self.assertEqual(stream.prune_chunks(102), 0)


class SctpUtilTest(TestCase):
    def test_tsn_minus_one(self) -> None:
        self.assertEqual(tsn_minus_one(0), 4294967295)
        self.assertEqual(tsn_minus_one(1), 0)
        self.assertEqual(tsn_minus_one(4294967294), 4294967293)
        self.assertEqual(tsn_minus_one(4294967295), 4294967294)

    def test_tsn_plus_one(self) -> None:
        self.assertEqual(tsn_plus_one(0), 1)
        self.assertEqual(tsn_plus_one(1), 2)
        self.assertEqual(tsn_plus_one(4294967294), 4294967295)
        self.assertEqual(tsn_plus_one(4294967295), 0)


class RTCSctpTransportTest(TestCase):
    @asynctest
    async def test_init_ack_without_cookie_cancels_init_before_abort_send(self) -> None:
        async with client_standalone() as client:
            channel = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )

            async def send_data(data: bytes) -> None:
                if data[12] == 6:
                    await asyncio.sleep(3.05)

            with patch.object(
                client.transport, "_send_data", side_effect=send_data
            ) as send:
                await client.start(client.getCapabilities(), 5000)
                chunk = InitAckChunk()
                chunk.initiate_tag = 0x01020304
                await client._handle_data(
                    serialize_packet(5000, 5000, client._local_verification_tag, chunk)
                )
                self.assertEqual(
                    [call.args[0][12] for call in send.await_args_list], [1, 6]
                )
                self.assertEqual(
                    send.await_args_list[1].args[0][:8],
                    bytes.fromhex("1388138801020304"),
                )
                self.assertEqual(
                    send.await_args_list[1].args[0][12:],
                    bytes.fromhex("0600000e0002000a0000000100070000"),
                )
                self.assertEqual(client.state, "closed")
                self.assertEqual(channel.readyState, "closed")
                self.assertIsNone(client._t1_handle)

    @asynctest
    async def test_init_ack_without_cookie_closes_when_abort_send_fails(self) -> None:
        async with client_standalone() as client:
            channel = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )
            with patch.object(
                client.transport, "_send_data", side_effect=[None, ConnectionError]
            ) as send:
                await client.start(client.getCapabilities(), 5000)
                chunk = InitAckChunk()
                chunk.initiate_tag = 0x01020304
                await client._handle_data(
                    serialize_packet(5000, 5000, client._local_verification_tag, chunk)
                )
                self.assertEqual(send.await_count, 2)
                self.assertEqual(
                    send.await_args.args[0][12:],
                    bytes.fromhex("0600000e0002000a0000000100070000"),
                )
                self.assertEqual(client.state, "closed")
                self.assertEqual(channel.readyState, "closed")
                self.assertIsNone(client._t1_handle)

    @asynctest
    async def test_init_ack_without_cookie_closes_when_abort_send_cancelled(
        self,
    ) -> None:
        async with client_standalone() as client:
            channel = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )
            started = asyncio.Event()

            async def send_data(data: bytes) -> None:
                if data[12] == 6:
                    started.set()
                    await asyncio.Event().wait()

            with patch.object(client.transport, "_send_data", side_effect=send_data):
                await client.start(client.getCapabilities(), 5000)
                chunk = InitAckChunk()
                chunk.initiate_tag = 0x01020304
                task = asyncio.create_task(
                    client._handle_data(
                        serialize_packet(
                            5000, 5000, client._local_verification_tag, chunk
                        )
                    )
                )
                try:
                    await asyncio.wait_for(started.wait(), 1)
                finally:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                self.assertEqual(client.state, "closed")
                self.assertEqual(channel.readyState, "closed")
                self.assertIsNone(client._t1_handle)

    @asynctest
    async def test_lost_forward_tsn_retransmits_until_acknowledged(self) -> None:
        async with client_and_server() as (client, server):
            client._local_tsn = 1
            client._last_sacked_tsn = 0
            client._advanced_peer_ack_tsn = 0
            local = RTCDataChannel(
                client,
                RTCDataChannelParameters(
                    label="partial", negotiated=True, id=100, maxRetransmits=0
                ),
            )
            remote = RTCDataChannel(
                server,
                RTCDataChannelParameters(label="partial", negotiated=True, id=100),
            )
            reliable = RTCDataChannel(
                client,
                RTCDataChannelParameters(label="reliable", negotiated=True, id=102),
            )
            remote_reliable = RTCDataChannel(
                server,
                RTCDataChannelParameters(label="reliable", negotiated=True, id=102),
            )
            received: asyncio.Queue[bytes] = asyncio.Queue()
            remote.on("message", received.put_nowait)
            remote_reliable.on("message", received.put_nowait)
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)
            with (
                patch.object(
                    client.transport, "_send_data", new_callable=AsyncMock
                ) as tx,
                patch.object(
                    server.transport, "_send_data", new_callable=AsyncMock
                ) as rx,
            ):
                local.send(b"M" * 4640)
                await asyncio.sleep(0)
                self.assertEqual(tx.await_count, 3)
                for expiry in range(2):
                    client._t3_cancel()
                    client._t3_expired()
                    await asyncio.sleep(0)
                    self.assertEqual(tx.await_count, 4 + expiry)
                    self.assertEqual(
                        tx.await_args.args[0][12:],
                        bytes.fromhex("c000000c0000000400640000"),
                    )
                    self.assertIsNotNone(client._t3_handle)
                    self.assertEqual(client._flight_size, 0)
                reliable.send(b"other")
                await asyncio.sleep(0)
                self.assertEqual(tx.await_count, 6)
                reliable_data = tx.await_args.args[0]
                self.assertEqual(
                    reliable_data[12:],
                    bytes.fromhex("00030015000000050066000000000035")
                    + b"other\x00\x00\x00",
                )
                client._t3_cancel()
                client._t3_expired()
                await asyncio.sleep(0)
                self.assertEqual(
                    [call.args[0][12:] for call in tx.await_args_list[6:]],
                    [bytes.fromhex("c000000c0000000400640000"), reliable_data[12:]],
                )
                self.assertEqual(client._flight_size, 5)
                self.assertEqual(client._rto, 3.0)
                timer = client._t3_handle
                self.assertIsNotNone(timer)
                await server._handle_data(tx.await_args_list[6].args[0])
                await client._handle_data(rx.await_args.args[0])
                self.assertEqual(client._flight_size, 5)
                self.assertIs(client._t3_handle, timer)
                await server._handle_data(tx.await_args_list[7].args[0])
                await client._handle_data(rx.await_args.args[0])
                self.assertEqual(await asyncio.wait_for(received.get(), 1), b"other")
                self.assertIsNone(client._t3_handle)
                self.assertEqual(tx.await_count, 8)
                local.send(b"after")
                await asyncio.sleep(0)
                self.assertEqual(
                    tx.await_args.args[0][12:],
                    bytes.fromhex("00030015000000060064000100000035")
                    + b"after\x00\x00\x00",
                )
                await server._handle_data(tx.await_args.args[0])
                await client._handle_data(rx.await_args.args[0])
                self.assertEqual(await asyncio.wait_for(received.get(), 1), b"after")
                self.assertTrue(received.empty())
                self.assertEqual(client._flight_size, 0)
                self.assertIsNone(client._t3_handle)

    @asynctest
    async def test_forward_tsn_retains_skips_across_new_abandonment(self) -> None:
        for ordered in (True, False):
            for first_tsn, first_ssn, last_tsn in (
                (1, 0, 4),
                (0xFFFFFFFD, 0xFFFF, 0),
            ):
                with self.subTest(ordered=ordered, first_tsn=first_tsn):
                    async with client_and_server() as (client, server):
                        client._local_tsn = first_tsn
                        client._last_sacked_tsn = (first_tsn - 1) & 0xFFFFFFFF
                        client._advanced_peer_ack_tsn = client._last_sacked_tsn
                        server._advertised_rwnd = 131072
                        local = RTCDataChannel(
                            client,
                            RTCDataChannelParameters(
                                label="partial",
                                negotiated=True,
                                id=100,
                                ordered=ordered,
                                maxRetransmits=0,
                            ),
                        )
                        remote = RTCDataChannel(
                            server,
                            RTCDataChannelParameters(
                                label="partial", negotiated=True, id=100
                            ),
                        )
                        other = RTCDataChannel(
                            client,
                            RTCDataChannelParameters(
                                label="other", negotiated=True, id=102, maxRetransmits=0
                            ),
                        )
                        remote_other = RTCDataChannel(
                            server,
                            RTCDataChannelParameters(
                                label="other", negotiated=True, id=102
                            ),
                        )
                        received: asyncio.Queue[bytes] = asyncio.Queue()
                        remote.on("message", received.put_nowait)
                        remote_other.on("message", received.put_nowait)
                        await server.start(client.getCapabilities(), client.port)
                        await client.start(server.getCapabilities(), server.port)
                        await wait_for_outcome(client, server)
                        if ordered:
                            client._outbound_stream_seq[100] = first_ssn
                            server._get_inbound_stream(100).sequence_number = first_ssn
                        with (
                            patch.object(
                                client.transport, "_send_data", new_callable=AsyncMock
                            ) as tx,
                            patch.object(
                                server.transport, "_send_data", new_callable=AsyncMock
                            ) as rx,
                        ):
                            local.send(b"M" * 4640)
                            await asyncio.sleep(0)
                            self.assertEqual(tx.await_count, 3)
                            await server._handle_data(tx.await_args_list[0].args[0])
                            self.assertEqual(server._advertised_rwnd, 129912)
                            client._t3_cancel()
                            client._t3_expired()
                            await asyncio.sleep(0)
                            self.assertEqual(tx.await_count, 4)
                            self.assertEqual(
                                tx.await_args.args[0][12:],
                                bytes.fromhex("c000000c" if ordered else "c0000008")
                                + last_tsn.to_bytes(4, "big")
                                + (
                                    bytes.fromhex("0064") + first_ssn.to_bytes(2, "big")
                                    if ordered
                                    else b""
                                ),
                            )
                            local.send(b"lost-again")
                            other.send(b"lost-other")
                            await asyncio.sleep(0)
                            self.assertEqual(tx.await_count, 6)
                            self.assertEqual(client._flight_size, 20)
                            latest_tsn = last_tsn + 2
                            latest_ssn = (first_ssn + 1) & 0xFFFF
                            wire = (
                                bytes.fromhex("c0000010" if ordered else "c000000c")
                                + latest_tsn.to_bytes(4, "big")
                                + (
                                    bytes.fromhex("0064")
                                    + latest_ssn.to_bytes(2, "big")
                                    if ordered
                                    else b""
                                )
                                + bytes.fromhex("00660000")
                            )
                            for expiry in range(2):
                                client._t3_cancel()
                                client._t3_expired()
                                await asyncio.sleep(0)
                                self.assertEqual(tx.await_count, 7 + expiry)
                                self.assertEqual(tx.await_args.args[0][12:], wire)
                                self.assertEqual(client._flight_size, 0)
                                self.assertEqual(client._rto, 3.0)
                                self.assertIsNotNone(client._t3_handle)
                            await server._handle_data(tx.await_args.args[0])
                            self.assertEqual(server._advertised_rwnd, 131072)
                            await client._handle_data(rx.await_args.args[0])
                            self.assertIsNone(client._t3_handle)
                            self.assertEqual(tx.await_count, 8)
                            local.send(b"after")
                            other.send(b"other")
                            await asyncio.sleep(0)
                            self.assertEqual(
                                [call.args[0][12:] for call in tx.await_args_list[8:]],
                                [
                                    bytes.fromhex("00030015" if ordered else "00070015")
                                    + (latest_tsn + 1).to_bytes(4, "big")
                                    + bytes.fromhex("0064")
                                    + (
                                        ((first_ssn + 2) & 0xFFFF).to_bytes(2, "big")
                                        if ordered
                                        else b"\x00\x00"
                                    )
                                    + bytes.fromhex("00000035")
                                    + b"after\x00\x00\x00",
                                    bytes.fromhex("00030015")
                                    + (latest_tsn + 2).to_bytes(4, "big")
                                    + bytes.fromhex("0066000100000035")
                                    + b"other\x00\x00\x00",
                                ],
                            )
                            for call in tx.await_args_list[8:]:
                                await server._handle_data(call.args[0])
                                await client._handle_data(rx.await_args.args[0])
                            self.assertEqual(
                                [
                                    await asyncio.wait_for(received.get(), 1),
                                    await asyncio.wait_for(received.get(), 1),
                                ],
                                [b"after", b"other"],
                            )
                            self.assertTrue(received.empty())
                            self.assertEqual(client._flight_size, 0)
                            self.assertEqual(outstanding_tsns(client), [])
                            self.assertEqual(queued_tsns(client), [])
                            self.assertIsNone(client._t3_handle)

    @asynctest
    async def test_forward_tsn_can_skip_more_than_half_ssn_space(self) -> None:
        async with client_and_server() as (client, server):
            client._local_tsn = 1
            client._last_sacked_tsn = 0
            client._advanced_peer_ack_tsn = 0
            local = RTCDataChannel(
                client,
                RTCDataChannelParameters(label="partial", negotiated=True, id=100),
            )
            remote = RTCDataChannel(
                server,
                RTCDataChannelParameters(label="partial", negotiated=True, id=100),
            )
            received: list[bytes] = []
            remote.on("message", received.append)
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)
            with (
                patch.object(
                    client.transport, "_send_data", new_callable=AsyncMock
                ) as tx,
                patch.object(
                    server.transport, "_send_data", new_callable=AsyncMock
                ) as rx,
            ):
                client._local_tsn = 32770
                client._advanced_peer_ack_tsn = 32769
                client._outbound_stream_seq[100] = 32769
                client._forward_tsn_streams[100] = 32768
                client._update_advanced_peer_ack_point()
                await client._transmit()
                self.assertEqual(
                    tx.await_args.args[0][12:],
                    bytes.fromhex("c000000c0000800100648000"),
                )
                await server._handle_data(tx.await_args.args[0])
                await client._handle_data(rx.await_args.args[0])
                local.send(b"first")
                local.send(b"second")
                await asyncio.sleep(0)
                first = tx.await_args_list[-2].args[0]
                second = tx.await_args_list[-1].args[0]
                self.assertEqual(first[20:24], bytes.fromhex("00648001"))
                self.assertEqual(second[20:24], bytes.fromhex("00648002"))
                await server._handle_data(second)
                with self.subTest(contract="hold later message"):
                    self.assertEqual(received, [])
                await server._handle_data(first)
                self.assertEqual(received, [b"first", b"second"])
                await client._handle_data(rx.await_args.args[0])
                self.assertEqual(client._flight_size, 0)
                self.assertIsNone(client._t3_handle)

    @asynctest
    async def test_aged_forward_tsn_skip_does_not_rewind_stream(self) -> None:
        for max_retransmits, delivered_count in ((None, 32769), (0, 32769), (0, 65533)):
            with self.subTest(
                max_retransmits=max_retransmits, delivered_count=delivered_count
            ):
                async with client_and_server() as (client, server):
                    client._local_tsn = 1
                    client._last_sacked_tsn = 0
                    client._advanced_peer_ack_tsn = 0
                    local = RTCDataChannel(
                        client,
                        RTCDataChannelParameters(
                            label="progress",
                            negotiated=True,
                            id=100,
                            maxRetransmits=max_retransmits,
                        ),
                    )
                    remote = RTCDataChannel(
                        server,
                        RTCDataChannelParameters(
                            label="progress", negotiated=True, id=100
                        ),
                    )
                    other = RTCDataChannel(
                        client,
                        RTCDataChannelParameters(
                            label="other", negotiated=True, id=102, maxRetransmits=0
                        ),
                    )
                    remote_other = RTCDataChannel(
                        server,
                        RTCDataChannelParameters(
                            label="other", negotiated=True, id=102
                        ),
                    )
                    received: list[bytes] = []
                    remote.on("message", received.append)
                    remote_other.on("message", received.append)
                    await server.start(client.getCapabilities(), client.port)
                    await client.start(server.getCapabilities(), server.port)
                    await wait_for_outcome(client, server)
                    with (
                        patch.object(
                            client.transport, "_send_data", new_callable=AsyncMock
                        ) as tx,
                        patch.object(
                            server.transport, "_send_data", new_callable=AsyncMock
                        ) as rx,
                    ):
                        await client._send(100, 53, b"old", max_retransmits=0)
                        client._t3_cancel()
                        client._t3_expired()
                        await asyncio.sleep(0)
                        first_forward = tx.await_args.args[0]
                        self.assertEqual(
                            first_forward[12:],
                            bytes.fromhex("c000000c0000000100640000"),
                        )
                        other.send(b"lost-other")
                        await asyncio.sleep(0)
                        client._t3_cancel()
                        client._t3_expired()
                        await asyncio.sleep(0)
                        second_forward = tx.await_args.args[0]
                        self.assertEqual(
                            second_forward[12:],
                            bytes.fromhex("c0000010000000020064000000660000"),
                        )
                        await server._handle_data(first_forward)
                        await client._handle_data(rx.await_args.args[0])
                        client._t3_cancel()
                        with patch.object(server, "_mark_received", return_value=False):
                            for sequence in range(1, delivered_count + 1):
                                chunk = DataChunk()
                                chunk.flags = SCTP_DATA_FIRST_FRAG | SCTP_DATA_LAST_FRAG
                                chunk.tsn = sequence + 2
                                chunk.stream_id = 100
                                chunk.stream_seq = sequence
                                chunk.protocol = 53
                                chunk.user_data = b"x"
                                chunk._abandoned = False
                                chunk._acked = False
                                chunk._book_size = 1
                                chunk._expiry = None
                                chunk._max_retransmits = max_retransmits
                                chunk._misses = 0
                                chunk._retransmit = False
                                chunk._sent_count = 1
                                chunk._sent_time = time.time()
                                client._sent_queue.append(chunk)
                                await server._receive_data_chunk(chunk)
                        self.assertEqual(received, [b"x"] * delivered_count)
                        received.clear()
                        client._flight_size = delivered_count
                        client._local_tsn = delivered_count + 3
                        client._outbound_stream_seq[100] = delivered_count + 1
                        server._sack_misordered = set(range(3, delivered_count + 3))
                        await server._send_sack()
                        self.assertEqual(
                            rx.await_args.args[0][12:],
                            bytes.fromhex(
                                "0300001400000001001000000001000000028002"
                                if delivered_count == 32769
                                else "030000140000000100100000000100000002fffe"
                            ),
                        )
                        await client._handle_data(rx.await_args.args[0])
                        self.assertEqual(client._flight_size, 0)
                        await server._handle_data(second_forward)
                        await client._handle_data(rx.await_args.args[0])
                        self.assertEqual(received, [])
                        local.send(b"first")
                        local.send(b"second")
                        await asyncio.sleep(0)
                        first = tx.await_args_list[-2].args[0]
                        second = tx.await_args_list[-1].args[0]
                        self.assertEqual(
                            first[12:],
                            bytes.fromhex(
                                "00030015000080040064800200000035"
                                if delivered_count == 32769
                                else "00030015000100000064fffe00000035"
                            )
                            + b"first\x00\x00\x00",
                        )
                        self.assertEqual(
                            second[20:24],
                            bytes.fromhex(
                                "00648003" if delivered_count == 32769 else "0064ffff"
                            ),
                        )
                        await server._handle_data(second)
                        with self.subTest(contract="hold later message"):
                            self.assertEqual(received, [])
                        await server._handle_data(first)
                        self.assertEqual(received, [b"first", b"second"])
                        await client._handle_data(rx.await_args.args[0])
                        other.send(b"other")
                        await asyncio.sleep(0)
                        await server._handle_data(tx.await_args.args[0])
                        await client._handle_data(rx.await_args.args[0])
                        self.assertEqual(received, [b"first", b"second", b"other"])
                        self.assertEqual(client._flight_size, 0)
                        self.assertIsNone(client._t3_handle)
                        await client._send(100, 53, b"lost-new", max_retransmits=0)
                        client._t3_cancel()
                        client._t3_expired()
                        await asyncio.sleep(0)
                        self.assertEqual(
                            tx.await_args.args[0][12:],
                            bytes.fromhex(
                                "c000000c0000800700648004"
                                if delivered_count == 32769
                                else "c000000c0001000300640000"
                            ),
                        )
                        await server._handle_data(tx.await_args.args[0])
                        await client._handle_data(rx.await_args.args[0])
                        local.send(b"after")
                        await asyncio.sleep(0)
                        await server._handle_data(tx.await_args.args[0])
                        await client._handle_data(rx.await_args.args[0])
                        self.assertEqual(
                            received, [b"first", b"second", b"other", b"after"]
                        )
                        self.assertEqual(client._flight_size, 0)
                        self.assertIsNone(client._t3_handle)

    @asynctest
    async def test_forward_tsn_after_long_idle_stream(self) -> None:
        for refresh_delivery in (False, True):
            with self.subTest(refresh_delivery=refresh_delivery):
                async with client_and_server() as (client, server):
                    client._local_tsn = 1
                    client._last_sacked_tsn = client._advanced_peer_ack_tsn = 0
                    local = RTCDataChannel(
                        client,
                        RTCDataChannelParameters(
                            label="idle", negotiated=True, id=100, maxRetransmits=0
                        ),
                    )
                    remote = RTCDataChannel(
                        server,
                        RTCDataChannelParameters(label="idle", negotiated=True, id=100),
                    )
                    RTCDataChannel(
                        client,
                        RTCDataChannelParameters(label="busy", negotiated=True, id=102),
                    )
                    RTCDataChannel(
                        server,
                        RTCDataChannelParameters(label="busy", negotiated=True, id=102),
                    )
                    received: list[bytes] = []
                    remote.on("message", received.append)
                    await server.start(client.getCapabilities(), client.port)
                    await client.start(server.getCapabilities(), server.port)
                    await wait_for_outcome(client, server)
                    with (
                        patch.object(
                            client.transport, "_send_data", new_callable=AsyncMock
                        ) as tx,
                        patch.object(
                            server.transport, "_send_data", new_callable=AsyncMock
                        ) as rx,
                    ):
                        local.send(b"old")
                        await asyncio.sleep(0)
                        await server._handle_data(tx.await_args.args[0])
                        await client._handle_data(rx.await_args.args[0])
                        self.assertEqual(received, [b"old"])
                        self.assertEqual(client._last_sacked_tsn, 1)
                        self.assertEqual(server._last_received_tsn, 1)
                        self.assertEqual(client._advanced_peer_ack_tsn, 1)
                        self.assertEqual(client._flight_size, 0)
                        self.assertEqual(list(client._sent_queue), [])
                        self.assertEqual(list(client._outbound_queue), [])
                        self.assertEqual(client._forward_tsn_streams, {})
                        self.assertIsNone(client._forward_tsn_chunk)
                        self.assertIsNone(client._t3_handle)
                        self.assertEqual(server._sack_misordered, set())
                        self.assertEqual(server._sack_duplicates, [])
                        self.assertEqual(server._advertised_rwnd, 1048576)
                        self.assertTrue(
                            all(
                                not stream.reassembly
                                for stream in server._inbound_streams.values()
                            )
                        )

                        # This checkpoint represents 0x80000001 fully ACKed DATA
                        # chunks on stream 102, not billions of simulated packets.
                        client._local_tsn = 0x80000003
                        client._last_sacked_tsn = 0x80000002
                        client._advanced_peer_ack_tsn = 0x80000002
                        client._outbound_stream_seq[102] = 1
                        server._last_received_tsn = 0x80000002
                        server._get_inbound_stream(102).sequence_number = 1
                        if refresh_delivery:
                            local.send(b"fresh")
                            await asyncio.sleep(0)
                            await server._handle_data(tx.await_args.args[0])
                            await client._handle_data(rx.await_args.args[0])
                            self.assertEqual(received, [b"old", b"fresh"])
                            self.assertEqual(client._flight_size, 0)
                            self.assertIsNone(client._t3_handle)

                        local.send(b"lost")
                        await asyncio.sleep(0)
                        client._t3_cancel()
                        client._t3_expired()
                        await asyncio.sleep(0)
                        forward = tx.await_args.args[0]
                        self.assertEqual(
                            forward[12:],
                            bytes.fromhex(
                                "c000000c8000000400640002"
                                if refresh_delivery
                                else "c000000c8000000300640001"
                            ),
                        )
                        await server._handle_data(forward)
                        await client._handle_data(rx.await_args.args[0])
                        local.send(b"after")
                        await asyncio.sleep(0)
                        after = tx.await_args.args[0]
                        self.assertEqual(
                            after[12:28],
                            bytes.fromhex(
                                "00030015800000050064000300000035"
                                if refresh_delivery
                                else "00030015800000040064000200000035"
                            ),
                        )
                        await server._handle_data(after)
                        await client._handle_data(rx.await_args.args[0])
                        self.assertEqual(
                            received,
                            [b"old", b"fresh", b"after"]
                            if refresh_delivery
                            else [b"old", b"after"],
                        )
                        self.assertEqual(client._flight_size, 0)
                        self.assertEqual(list(client._sent_queue), [])
                        self.assertIsNone(client._t3_handle)
                        self.assertEqual(server._inbound_streams[100].reassembly, [])
                        self.assertEqual(server._advertised_rwnd, 1048576)

    @asynctest
    async def test_lost_reset_response_does_not_block_acknowledged_boundary(
        self,
    ) -> None:
        for ack_before_reset in (True, False):
            with self.subTest(ack_before_reset=ack_before_reset):
                async with client_and_server() as (client, server):
                    client._local_tsn = 1
                    client._last_sacked_tsn = client._advanced_peer_ack_tsn = 0
                    local = RTCDataChannel(
                        client,
                        RTCDataChannelParameters(
                            label="closing", negotiated=True, id=100
                        ),
                    )
                    remote = RTCDataChannel(
                        server,
                        RTCDataChannelParameters(
                            label="closing", negotiated=True, id=100
                        ),
                    )
                    other = RTCDataChannel(
                        client,
                        RTCDataChannelParameters(
                            label="other", negotiated=True, id=102, maxRetransmits=0
                        ),
                    )
                    remote_other = RTCDataChannel(
                        server,
                        RTCDataChannelParameters(
                            label="other", negotiated=True, id=102
                        ),
                    )
                    received: list[bytes] = []
                    remote_other.on("message", received.append)
                    await server.start(client.getCapabilities(), client.port)
                    await client.start(server.getCapabilities(), server.port)
                    await wait_for_outcome(client, server)
                    with (
                        patch.object(
                            client.transport, "_send_data", new_callable=AsyncMock
                        ) as tx,
                        patch.object(
                            server.transport, "_send_data", new_callable=AsyncMock
                        ) as rx,
                    ):
                        local.send(b"old")
                        await asyncio.sleep(0)
                        await server._handle_data(tx.await_args.args[0])
                        sack = rx.await_args.args[0]
                        if ack_before_reset:
                            await client._handle_data(sack)
                        local.close()
                        await asyncio.sleep(0)
                        tx_pos = rx_pos = 0
                        held_response = b""
                        for _ in range(4):
                            await asyncio.sleep(0)
                            while tx_pos < tx.await_count:
                                packet = tx.await_args_list[tx_pos].args[0]
                                tx_pos += 1
                                if packet[12] == 130:
                                    await server._handle_data(packet)
                            while rx_pos < rx.await_count:
                                packet = rx.await_args_list[rx_pos].args[0]
                                rx_pos += 1
                                if packet[12] != 130:
                                    continue
                                if not held_response and any(
                                    kind == 16
                                    for chunk in parse_packet(packet)[3]
                                    if isinstance(chunk, ReconfigChunk)
                                    for kind, value in chunk.params
                                ):
                                    held_response = packet
                                else:
                                    await client._handle_data(packet)
                        self.assertTrue(held_response)
                        self.assertEqual(local.readyState, "closing")
                        self.assertEqual(remote.readyState, "closed")
                        if not ack_before_reset:
                            await client._handle_data(sack)
                        self.assertEqual(client._last_sacked_tsn, 1)
                        self.assertEqual(client._forward_tsn_streams, {})
                        self.assertEqual(client._flight_size, 0)
                        with self.assertRaises(ValueError):
                            RTCDataChannel(
                                client,
                                RTCDataChannelParameters(
                                    label="too-early", negotiated=True, id=100
                                ),
                            )

                        other.send(b"lost")
                        await asyncio.sleep(0)
                        client._t3_cancel()
                        client._t3_expired()
                        await asyncio.sleep(0)
                        forward = tx.await_args.args[0]
                        self.assertEqual(
                            forward[12:], bytes.fromhex("c000000c0000000200660000")
                        )
                        other.send(b"after")
                        await asyncio.sleep(0)
                        await server._handle_data(tx.await_args.args[0])
                        self.assertEqual(received, [])
                        await client._handle_data(rx.await_args.args[0])
                        await server._handle_data(forward)
                        await client._handle_data(rx.await_args.args[0])
                        self.assertEqual(received, [b"after"])
                        self.assertEqual(client._flight_size, 0)
                        self.assertEqual(list(client._sent_queue), [])
                        self.assertIsNone(client._t3_handle)
                        self.assertEqual(local.readyState, "closing")
                        await client._handle_data(held_response)
                        self.assertEqual(local.readyState, "closed")
                        replacement = RTCDataChannel(
                            client,
                            RTCDataChannelParameters(
                                label="reused", negotiated=True, id=100
                            ),
                        )
                        self.assertEqual(replacement.readyState, "open")

    @asynctest
    async def test_reset_retires_forward_tsn_skips_before_channel_reuse(self) -> None:
        for close_from_sender in (True, False):
            for control_state, old_delivered in (
                ("queued", False),
                ("transmitted", False),
                ("future", False),
                ("future", True),
            ):
                with self.subTest(
                    close_from_sender=close_from_sender,
                    control_state=control_state,
                    old_delivered=old_delivered,
                ):
                    async with client_and_server() as (client, server):
                        client._local_tsn = 1
                        client._last_sacked_tsn = 0
                        client._advanced_peer_ack_tsn = 0
                        local = RTCDataChannel(
                            client,
                            RTCDataChannelParameters(
                                label="old", negotiated=True, id=100, maxRetransmits=0
                            ),
                        )
                        remote = RTCDataChannel(
                            server,
                            RTCDataChannelParameters(
                                label="old", negotiated=True, id=100
                            ),
                        )
                        other = RTCDataChannel(
                            client,
                            RTCDataChannelParameters(
                                label="other", negotiated=True, id=102, maxRetransmits=0
                            ),
                        )
                        remote_other = RTCDataChannel(
                            server,
                            RTCDataChannelParameters(
                                label="other", negotiated=True, id=102
                            ),
                        )
                        received: list[bytes] = []
                        remote_other.on("message", received.append)
                        await server.start(client.getCapabilities(), client.port)
                        await client.start(server.getCapabilities(), server.port)
                        await wait_for_outcome(client, server)
                        with (
                            patch.object(
                                client.transport, "_send_data", new_callable=AsyncMock
                            ) as tx,
                            patch.object(
                                server.transport, "_send_data", new_callable=AsyncMock
                            ) as rx,
                        ):
                            local.send(b"old")
                            await asyncio.sleep(0)
                            if not old_delivered:
                                client._t3_cancel()
                                client._t3_expired()
                                await asyncio.sleep(0)
                            old_packet = tx.await_args.args[0]
                            self.assertEqual(
                                old_packet[12:],
                                bytes.fromhex("00030013000000010064000000000035")
                                + b"old\x00"
                                if old_delivered
                                else bytes.fromhex("c000000c0000000100640000"),
                            )
                            await server._handle_data(old_packet)

                            async def abandon_other() -> None:
                                other.send(b"lost-other")
                                await asyncio.sleep(0)
                                client._t3_cancel()
                                client._t3_expired()

                            (local if close_from_sender else remote).close()
                            tx_pos = rx_pos = 0
                            reset_response = b""
                            delayed_forward = b""
                            for _ in range(4):
                                await asyncio.sleep(0)
                                while tx_pos < tx.await_count:
                                    packet = tx.await_args_list[tx_pos].args[0]
                                    tx_pos += 1
                                    if packet[12] == 130:
                                        await server._handle_data(packet)
                                while rx_pos < rx.await_count:
                                    packet = rx.await_args_list[rx_pos].args[0]
                                    rx_pos += 1
                                    if packet[12] != 130:
                                        continue
                                    is_response = any(
                                        kind == 16
                                        for chunk in parse_packet(packet)[3]
                                        if isinstance(chunk, ReconfigChunk)
                                        for kind, body in chunk.params
                                    )
                                    if is_response:
                                        reset_response = packet
                                        if control_state != "future":
                                            await abandon_other()
                                        if control_state == "transmitted":
                                            await asyncio.sleep(0)
                                            delayed_forward = tx.await_args.args[0]
                                            with self.subTest(contract="reset fence"):
                                                self.assertEqual(
                                                    delayed_forward[12:],
                                                    bytes.fromhex(
                                                        "c000000c0000000100640000"
                                                    ),
                                                )
                                    await client._handle_data(packet)
                            self.assertEqual(local.readyState, "closed")
                            self.assertEqual(remote.readyState, "closed")

                            replacement = RTCDataChannel(
                                client,
                                RTCDataChannelParameters(
                                    label="new", negotiated=True, id=100
                                ),
                            )
                            remote_replacement = RTCDataChannel(
                                server,
                                RTCDataChannelParameters(
                                    label="new", negotiated=True, id=100
                                ),
                            )
                            remote_replacement.on("message", received.append)
                            self.assertTrue(reset_response)
                            await client._handle_data(reset_response)
                            await client._handle_data(reset_response)
                            self.assertEqual(replacement.readyState, "open")
                            if delayed_forward:
                                await server._handle_data(delayed_forward)
                            if control_state == "future":
                                await abandon_other()
                                await asyncio.sleep(0)
                            forward = [
                                call.args[0]
                                for call in tx.await_args_list
                                if call.args[0][12] == 192
                            ][-1]
                            with self.subTest(contract="wire"):
                                self.assertEqual(
                                    forward[12:],
                                    bytes.fromhex("c000000c0000000200660000"),
                                )
                            for _ in range(2):
                                client._t3_cancel()
                                client._t3_expired()
                                await asyncio.sleep(0)
                                with self.subTest(contract="retransmitted wire"):
                                    self.assertEqual(
                                        tx.await_args.args[0][12:],
                                        bytes.fromhex("c000000c0000000200660000"),
                                    )
                            await server._handle_data(forward)
                            await server._handle_data(old_packet)
                            replacement.send(b"first")
                            replacement.send(b"second")
                            await asyncio.sleep(0)
                            first = tx.await_args_list[-2].args[0]
                            second = tx.await_args_list[-1].args[0]
                            self.assertEqual(first[20:24], bytes.fromhex("00640000"))
                            self.assertEqual(second[20:24], bytes.fromhex("00640001"))
                            await server._handle_data(second)
                            with self.subTest(contract="hold later message"):
                                self.assertEqual(received, [])
                            await server._handle_data(first)
                            with self.subTest(contract="ordered delivery"):
                                self.assertEqual(received, [b"first", b"second"])
                            await client._handle_data(rx.await_args.args[0])
                            other.send(b"other")
                            await asyncio.sleep(0)
                            await server._handle_data(tx.await_args.args[0])
                            self.assertEqual(received[-1:], [b"other"])
                            await client._handle_data(rx.await_args.args[0])
                            self.assertEqual(client._flight_size, 0)
                            self.assertIsNone(client._t3_handle)

    @asynctest
    async def test_reset_waits_for_successful_response(self) -> None:
        for result, offset in ((6, 0), (2, 0), (1, 1)):
            with self.subTest(result=result, offset=offset):
                async with client_and_server() as (client, server):
                    local = RTCDataChannel(
                        client,
                        RTCDataChannelParameters(label="old", negotiated=True, id=100),
                    )
                    RTCDataChannel(
                        server,
                        RTCDataChannelParameters(label="old", negotiated=True, id=100),
                    )
                    await server.start(client.getCapabilities(), client.port)
                    await client.start(server.getCapabilities(), server.port)
                    await wait_for_outcome(client, server)
                    with patch.object(
                        client.transport, "_send_data", new_callable=AsyncMock
                    ) as tx:
                        local.close()
                        await asyncio.sleep(0)
                        request_chunk = parse_packet(tx.await_args.args[0])[3][0]
                        assert isinstance(request_chunk, ReconfigChunk)
                        request = StreamResetOutgoingParam.parse(
                            request_chunk.params[0][1]
                        )
                        response = ReconfigChunk()
                        response.params = [
                            (
                                16,
                                bytes(
                                    StreamResetResponseParam(
                                        response_sequence=tsn_plus_one(
                                            request.request_sequence
                                        )
                                        if offset
                                        else request.request_sequence,
                                        result=result,
                                    )
                                ),
                            )
                        ]
                        await client._handle_data(
                            serialize_packet(
                                5000, 5000, client._local_verification_tag, response
                            )
                        )
                        self.assertEqual(local.readyState, "closing")
                        with self.assertRaises(ValueError):
                            RTCDataChannel(
                                client,
                                RTCDataChannelParameters(
                                    label="new", negotiated=True, id=100
                                ),
                            )

    @asynctest
    async def test_reset_defers_until_old_forward_tsn_is_consumed(self) -> None:
        async with client_and_server() as (client, server):
            client._local_tsn = 1
            client._last_sacked_tsn = 0
            client._advanced_peer_ack_tsn = 0
            local = RTCDataChannel(
                client,
                RTCDataChannelParameters(
                    label="old", negotiated=True, id=100, maxRetransmits=0
                ),
            )
            remote = RTCDataChannel(
                server, RTCDataChannelParameters(label="old", negotiated=True, id=100)
            )
            other = RTCDataChannel(
                client,
                RTCDataChannelParameters(
                    label="other", negotiated=True, id=102, maxRetransmits=0
                ),
            )
            remote_other = RTCDataChannel(
                server, RTCDataChannelParameters(label="other", negotiated=True, id=102)
            )
            received: list[bytes] = []
            remote_other.on("message", received.append)
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)
            with (
                patch.object(
                    client.transport, "_send_data", new_callable=AsyncMock
                ) as tx,
                patch.object(
                    server.transport, "_send_data", new_callable=AsyncMock
                ) as rx,
            ):
                local.send(b"old")
                await asyncio.sleep(0)
                client._t3_cancel()
                client._t3_expired()
                await asyncio.sleep(0)
                first_forward = tx.await_args.args[0]
                other.send(b"old-other")
                await asyncio.sleep(0)
                client._t3_cancel()
                client._t3_expired()
                await asyncio.sleep(0)
                second_forward = tx.await_args.args[0]
                self.assertEqual(
                    second_forward[12:],
                    bytes.fromhex("c0000010000000020064000000660000"),
                )
                local.close()
                await asyncio.sleep(0)
                reset_request = tx.await_args.args[0]
                await server._handle_data(reset_request)
                early_response = rx.await_args.args[0]
                with self.subTest(contract="reset boundary"):
                    self.assertEqual(early_response[-4:], bytes.fromhex("00000006"))
                    self.assertEqual(remote.readyState, "open")
                await client._handle_data(early_response)
                await server._handle_data(reset_request)
                await client._handle_data(rx.await_args.args[0])
                await server._handle_data(first_forward)
                with self.subTest(contract="partial progress"):
                    self.assertEqual(remote.readyState, "open")
                await server._handle_data(second_forward)
                tx_pos = rx_pos = 0
                for _ in range(4):
                    await asyncio.sleep(0)
                    while tx_pos < tx.await_count:
                        packet = tx.await_args_list[tx_pos].args[0]
                        tx_pos += 1
                        if packet[12] == 130 and packet != reset_request:
                            await server._handle_data(packet)
                    while rx_pos < rx.await_count:
                        packet = rx.await_args_list[rx_pos].args[0]
                        rx_pos += 1
                        if packet[12] == 130:
                            await client._handle_data(packet)
                self.assertEqual(local.readyState, "closed")
                self.assertEqual(remote.readyState, "closed")
                replacement = RTCDataChannel(
                    client,
                    RTCDataChannelParameters(label="new", negotiated=True, id=100),
                )
                remote_replacement = RTCDataChannel(
                    server,
                    RTCDataChannelParameters(label="new", negotiated=True, id=100),
                )
                remote_replacement.on("message", received.append)
                for delayed in (second_forward, first_forward, second_forward):
                    await server._handle_data(delayed)
                replacement.send(b"first")
                replacement.send(b"second")
                await asyncio.sleep(0)
                first = tx.await_args_list[-2].args[0]
                second = tx.await_args_list[-1].args[0]
                await server._handle_data(second)
                self.assertEqual(received, [])
                await server._handle_data(first)
                self.assertEqual(received, [b"first", b"second"])
                await client._handle_data(rx.await_args.args[0])
                other.send(b"other")
                await asyncio.sleep(0)
                await server._handle_data(tx.await_args.args[0])
                await client._handle_data(rx.await_args.args[0])
                self.assertEqual(received, [b"first", b"second", b"other"])
                self.assertEqual(client._flight_size, 0)
                self.assertIsNone(client._t3_handle)

    @asynctest
    async def test_partial_forward_tsn_ack_does_not_rewind_delivered_stream(
        self,
    ) -> None:
        async with client_and_server() as (client, server):
            client._local_tsn = 1
            client._last_sacked_tsn = 0
            client._advanced_peer_ack_tsn = 0
            server._advertised_rwnd = 131072
            local = RTCDataChannel(
                client,
                RTCDataChannelParameters(
                    label="partial", negotiated=True, id=100, maxRetransmits=0
                ),
            )
            remote = RTCDataChannel(
                server,
                RTCDataChannelParameters(label="partial", negotiated=True, id=100),
            )
            other = RTCDataChannel(
                client,
                RTCDataChannelParameters(
                    label="other", negotiated=True, id=102, maxRetransmits=0
                ),
            )
            RTCDataChannel(
                server, RTCDataChannelParameters(label="other", negotiated=True, id=102)
            )
            reliable = RTCDataChannel(
                client,
                RTCDataChannelParameters(label="reliable", negotiated=True, id=104),
            )
            remote_reliable = RTCDataChannel(
                server,
                RTCDataChannelParameters(label="reliable", negotiated=True, id=104),
            )
            received: asyncio.Queue[bytes] = asyncio.Queue()
            remote.on("message", received.put_nowait)
            remote_reliable.on("message", received.put_nowait)
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)
            with (
                patch.object(
                    client.transport, "_send_data", new_callable=AsyncMock
                ) as tx,
                patch.object(
                    server.transport, "_send_data", new_callable=AsyncMock
                ) as rx,
            ):
                local.send(b"lost-a")
                await asyncio.sleep(0)
                client._t3_cancel()
                client._t3_expired()
                await asyncio.sleep(0)
                first_forward = tx.await_args.args[0]
                self.assertEqual(
                    first_forward[12:], bytes.fromhex("c000000c0000000100640000")
                )
                other.send(b"lost-b")
                await asyncio.sleep(0)
                client._t3_cancel()
                client._t3_expired()
                await asyncio.sleep(0)
                second_forward = tx.await_args.args[0]
                self.assertEqual(
                    second_forward[12:],
                    bytes.fromhex("c0000010000000020064000000660000"),
                )
                local.send(b"one")
                reliable.send(b"other")
                await asyncio.sleep(0)
                self.assertEqual(tx.await_count, 6)
                one = tx.await_args_list[4].args[0]
                reliable_data = tx.await_args_list[5].args[0]
                self.assertEqual(client._flight_size, 8)
                await server._handle_data(first_forward)
                await server._handle_data(one)
                self.assertEqual(await asyncio.wait_for(received.get(), 1), b"one")
                partial_ack = rx.await_args.args[0]
                self.assertEqual(
                    partial_ack[12:],
                    bytes.fromhex("0300001400000001000200000001000000020002"),
                )
                timer = client._t3_handle
                for repeats in range(2):
                    await client._handle_data(partial_ack)
                    self.assertEqual(tx.await_count, 7 + repeats)
                    self.assertEqual(tx.await_args.args[0][12:], second_forward[12:])
                    self.assertIs(client._t3_handle, timer)
                    self.assertEqual(client._flight_size, 5)
                await server._handle_data(tx.await_args.args[0])
                self.assertEqual(
                    rx.await_args.args[0][12:],
                    bytes.fromhex("03000010000000030002000000000000"),
                )
                await client._handle_data(rx.await_args.args[0])
                self.assertEqual(client._flight_size, 5)
                self.assertEqual(tx.await_count, 8)
                timer = client._t3_handle
                await client._handle_data(partial_ack)
                self.assertEqual(tx.await_count, 8)
                self.assertIs(client._t3_handle, timer)
                for delayed in (first_forward, second_forward):
                    await server._handle_data(delayed)
                    self.assertEqual(
                        rx.await_args.args[0][12:],
                        bytes.fromhex("03000010000000030002000000000000"),
                    )
                    self.assertTrue(received.empty())
                local.send(b"after")
                await asyncio.sleep(0)
                self.assertEqual(
                    tx.await_args.args[0][12:],
                    bytes.fromhex("00030015000000050064000200000035")
                    + b"after\x00\x00\x00",
                )
                await server._handle_data(reliable_data)
                await server._handle_data(tx.await_args.args[0])
                await client._handle_data(rx.await_args.args[0])
                self.assertEqual(
                    [
                        await asyncio.wait_for(received.get(), 1),
                        await asyncio.wait_for(received.get(), 1),
                    ],
                    [b"other", b"after"],
                )
                self.assertTrue(received.empty())
                self.assertEqual(client._flight_size, 0)
                self.assertIsNone(client._t3_handle)
                other.send(b"lost-again")
                await asyncio.sleep(0)
                client._t3_cancel()
                client._t3_expired()
                await asyncio.sleep(0)
                self.assertEqual(
                    tx.await_args.args[0][12:],
                    bytes.fromhex("c000000c0000000600660001"),
                )
                await server._handle_data(tx.await_args.args[0])
                await client._handle_data(rx.await_args.args[0])
                self.assertIsNone(client._t3_handle)
                self.assertEqual(client._flight_size, 0)
                self.assertEqual(outstanding_tsns(client), [])
                self.assertEqual(queued_tsns(client), [])

    @asynctest
    async def test_init_ack_without_cookie_aborts_association(self) -> None:
        for params in ([], [(0x8008, b"\x82")]):
            with self.subTest(params=params):
                async with client_standalone() as client:
                    channel = RTCDataChannel(
                        client,
                        RTCDataChannelParameters(label="chat", negotiated=True, id=100),
                    )
                    with patch.object(
                        client.transport, "_send_data", new_callable=AsyncMock
                    ) as send:
                        await client.start(client.getCapabilities(), 5000)
                        chunk = InitAckChunk()
                        chunk.initiate_tag = 0x01020304
                        chunk.advertised_rwnd = 131072
                        chunk.outbound_streams = 16
                        chunk.inbound_streams = 16
                        chunk.initial_tsn = 100
                        chunk.params = params
                        await client._handle_data(
                            serialize_packet(
                                5000, 5000, client._local_verification_tag, chunk
                            )
                        )
                        self.assertEqual(send.await_count, 2)
                        self.assertEqual(
                            send.await_args.args[0][:8],
                            bytes.fromhex("1388138801020304"),
                        )
                        self.assertEqual(
                            send.await_args.args[0][12:],
                            bytes.fromhex("0600000e0002000a0000000100070000"),
                        )
                        self.assertEqual(client.state, "closed")
                        self.assertEqual(channel.readyState, "closed")
                        self.assertIsNone(client._t1_handle)

    @asynctest
    async def test_init_ack_echoes_cookie_between_optional_parameters(self) -> None:
        async with client_standalone() as client:
            channel = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )
            with patch.object(
                client.transport, "_send_data", new_callable=AsyncMock
            ) as send:
                await client.start(client.getCapabilities(), 5000)
                chunk = InitAckChunk()
                chunk.initiate_tag = 0x01020304
                chunk.advertised_rwnd = 131072
                chunk.outbound_streams = 16
                chunk.inbound_streams = 16
                chunk.initial_tsn = 100
                chunk.params = [(0x8008, b"\x82"), (7, b"cookie"), (0xC000, b"")]
                await client._handle_data(
                    serialize_packet(5000, 5000, client._local_verification_tag, chunk)
                )
                self.assertEqual(send.await_count, 2)
                self.assertEqual(
                    send.await_args.args[0][:8], bytes.fromhex("1388138801020304")
                )
                self.assertEqual(
                    send.await_args.args[0][12:], b"\x0a\x00\x00\x0acookie\x00\x00"
                )
                self.assertEqual(channel.readyState, "connecting")
                self.assertIsNotNone(client._t1_handle)
                await client._handle_data(
                    serialize_packet(
                        5000, 5000, client._local_verification_tag, CookieAckChunk()
                    )
                )
                self.assertEqual(client.state, "connected")
                self.assertEqual(channel.readyState, "open")
                self.assertIsNone(client._t1_handle)

    @asynctest
    async def test_half_range_data_is_ignored_before_reassembly(self) -> None:
        for initial_tsn in (0, 0x80000000, 0xFFFFFFFE):
            for flags in (
                SCTP_DATA_FIRST_FRAG,
                SCTP_DATA_FIRST_FRAG | SCTP_DATA_LAST_FRAG,
            ):
                with self.subTest(initial_tsn=initial_tsn, flags=flags):
                    async with client_and_server() as (client, server):
                        client._local_tsn = initial_tsn
                        client._last_sacked_tsn = tsn_minus_one(initial_tsn)
                        client._advanced_peer_ack_tsn = tsn_minus_one(initial_tsn)
                        local = RTCDataChannel(
                            client,
                            RTCDataChannelParameters(
                                label="chat", negotiated=True, id=0
                            ),
                        )
                        remote = RTCDataChannel(
                            server,
                            RTCDataChannelParameters(
                                label="chat", negotiated=True, id=0
                            ),
                        )
                        received: asyncio.Queue[str] = asyncio.Queue()
                        messages: list[str] = []
                        remote.on("message", received.put_nowait)
                        remote.on("message", messages.append)
                        await server.start(client.getCapabilities(), client.port)
                        await client.start(server.getCapabilities(), server.port)
                        await wait_for_outcome(client, server)
                        local.send("seed")
                        self.assertEqual(
                            await asyncio.wait_for(received.get(), 1), "seed"
                        )

                        stream = server._inbound_streams[0]
                        before_tsn = server._last_received_tsn
                        before_window = server._advertised_rwnd
                        before_sequence = stream.sequence_number
                        chunk = DataChunk(flags=flags)
                        chunk.tsn = (before_tsn + 0x80000000) % 0x100000000
                        chunk.stream_id = 0
                        chunk.stream_seq = before_sequence
                        chunk.protocol = 51
                        chunk.user_data = b"ambiguous"
                        packet = serialize_packet(
                            client.port,
                            server.port,
                            server._local_verification_tag,
                            chunk,
                        )
                        with patch.object(
                            server.transport,
                            "_send_data",
                            wraps=server.transport._send_data,
                        ) as send:
                            await server._handle_data(packet)
                            await server._handle_data(packet)
                        self.assertEqual(messages, ["seed"])
                        self.assertEqual(server._last_received_tsn, before_tsn)
                        self.assertEqual(server._advertised_rwnd, before_window)
                        self.assertEqual(server._sack_misordered, set())
                        self.assertEqual(server._sack_duplicates, [])
                        self.assertEqual(stream.reassembly, [])
                        self.assertEqual(stream.sequence_number, before_sequence)
                        self.assertTrue(received.empty())
                        send.assert_not_awaited()

                        local.send("valid")
                        local.send("following")
                        self.assertEqual(
                            await asyncio.wait_for(received.get(), 1), "valid"
                        )
                        self.assertEqual(
                            await asyncio.wait_for(received.get(), 1), "following"
                        )
                        self.assertTrue(received.empty())
                        self.assertEqual(messages, ["seed", "valid", "following"])
                        self.assertEqual(
                            server._last_received_tsn,
                            (initial_tsn + 2) % 0x100000000,
                        )
                        self.assertEqual(server._advertised_rwnd, before_window)
                        self.assertEqual(stream.reassembly, [])

    @asynctest
    async def test_reconfig_subclasses_preserve_wire_types(self) -> None:
        class Reset(StreamResetOutgoingParam):
            pass

        class Response(StreamResetResponseParam):
            pass

        class Add(StreamAddOutgoingParam):
            pass

        cases = (
            (
                Reset(1, 2, 3),
                bytes.fromhex("82000014000d0010000000010000000200000003"),
            ),
            (Response(1, 2), bytes.fromhex("820000100010000c0000000100000002")),
            (Add(1, 2), bytes.fromhex("820000100011000c0000000100020000")),
        )
        async with client_and_server() as (client, server):
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)
            with patch.object(
                client.transport, "_send_data", new_callable=AsyncMock
            ) as send:
                for param, expected in cases:
                    with self.subTest(parameter=type(param).__name__):
                        send.reset_mock()
                        await client._send_reconfig_param(param)
                        send.assert_awaited_once()
                        self.assertEqual(send.await_args.args[0][12:], expected)
                        self.assertEqual(type(param).parse(bytes(param)), param)

    @asynctest
    async def test_reliability_open_and_empty_messages(self) -> None:
        cases = (
            (
                RTCDataChannelParameters(
                    label="chat", protocol="raw", ordered=False, maxRetransmits=3
                ),
                bytes.fromhex("03810000000000030004000363686174726177"),
            ),
            (
                RTCDataChannelParameters(
                    label="chat", protocol="raw", maxPacketLifeTime=60000
                ),
                bytes.fromhex("030200000000ea600004000363686174726177"),
            ),
        )
        for parameters, expected_open in cases:
            with self.subTest(parameters=parameters):
                async with client_and_server() as (client, server):
                    incoming = asyncio.get_running_loop().create_future()
                    server.on("datachannel", incoming.set_result)
                    await server.start(client.getCapabilities(), client.port)
                    await client.start(server.getCapabilities(), server.port)
                    await wait_for_outcome(client, server)
                    with patch.object(
                        client.transport,
                        "_send_data",
                        wraps=client.transport._send_data,
                    ) as send:
                        await client.start(server.getCapabilities(), server.port)
                        send.assert_not_awaited()
                        local = RTCDataChannel(client, parameters)
                        opened = asyncio.Event()
                        local.on("open", opened.set)
                        remote = await asyncio.wait_for(incoming, 1)
                        await asyncio.wait_for(opened.wait(), 1)
                        open_messages = [
                            chunk.user_data
                            for call in send.await_args_list
                            for chunk in parse_packet(call.args[0])[3]
                            if isinstance(chunk, DataChunk) and chunk.protocol == 50
                        ]
                        self.assertEqual(open_messages, [expected_open])
                    self.assertEqual(remote.label, "chat")
                    self.assertEqual(remote.protocol, "raw")
                    self.assertEqual(remote.ordered, parameters.ordered)
                    self.assertEqual(remote.maxRetransmits, parameters.maxRetransmits)
                    self.assertEqual(
                        remote.maxPacketLifeTime, parameters.maxPacketLifeTime
                    )
                    received: asyncio.Queue[str | bytes] = asyncio.Queue()
                    remote.on("message", received.put_nowait)
                    for message in ("", b"", "text", b"binary"):
                        local.send(message)
                        self.assertEqual(
                            await asyncio.wait_for(received.get(), 1), message
                        )
                    self.assertTrue(received.empty())
                    self.assertEqual(local.bufferedAmount, 0)

    @asynctest
    async def test_close_before_connect_preserves_other_channels(self) -> None:
        for negotiated in (False, True):
            with self.subTest(negotiated=negotiated):
                async with client_and_server() as (client, server):
                    discarded = RTCDataChannel(
                        client,
                        RTCDataChannelParameters(
                            label="discarded",
                            negotiated=negotiated,
                            id=2 if negotiated else None,
                        ),
                    )
                    survivor = RTCDataChannel(
                        client, RTCDataChannelParameters(label="survivor")
                    )
                    discarded.close()
                    discarded.close()
                    self.assertEqual(discarded.readyState, "closed")
                    incoming = asyncio.get_running_loop().create_future()
                    server.on("datachannel", incoming.set_result)
                    channels = track_channels(server)
                    opened = asyncio.Event()
                    survivor.on("open", opened.set)
                    await server.start(client.getCapabilities(), client.port)
                    await client.start(server.getCapabilities(), server.port)
                    remote = await asyncio.wait_for(incoming, 1)
                    await asyncio.wait_for(opened.wait(), 1)
                    self.assertEqual(
                        [channel.label for channel in channels], ["survivor"]
                    )
                    received: asyncio.Queue[str] = asyncio.Queue()
                    remote.on("message", received.put_nowait)
                    survivor.send("after close")
                    self.assertEqual(
                        await asyncio.wait_for(received.get(), 1), "after close"
                    )
                    self.assertEqual(discarded.readyState, "closed")

    @asynctest
    async def test_abort_closes_established_channel(self) -> None:
        async with client_and_server() as (client, server):
            local = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", negotiated=True, id=0)
            )
            remote = RTCDataChannel(
                server, RTCDataChannelParameters(label="chat", negotiated=True, id=0)
            )
            received: asyncio.Queue[str] = asyncio.Queue()
            remote.on("message", received.put_nowait)
            closed = asyncio.Event()
            local.on("close", closed.set)
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)
            local.send("before abort")
            self.assertEqual(await asyncio.wait_for(received.get(), 1), "before abort")
            await server._send_chunk(AbortChunk())
            await asyncio.wait_for(closed.wait(), 1)
            self.assertEqual(client.state, "closed")
            self.assertEqual(local.readyState, "closed")
            self.assertIsNone(client._t1_handle)
            self.assertIsNone(client._t2_handle)
            self.assertIsNone(client._t3_handle)
            with self.assertRaises(InvalidStateError):
                local.send("after abort")

    @asynctest
    async def test_unknown_reconfig_parameter_does_not_hide_stream_add(self) -> None:
        async with client_and_server() as (client, server):
            client._outbound_streams_count = 16
            local = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", negotiated=True, id=0)
            )
            remote = RTCDataChannel(
                server, RTCDataChannelParameters(label="chat", negotiated=True, id=0)
            )
            received: asyncio.Queue[str] = asyncio.Queue()
            remote.on("message", received.put_nowait)
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)
            self.assertEqual(server.maxChannels, 16)

            chunk = ReconfigChunk()
            chunk.params = [
                (0xFFFF, b"extension"),
                (17, bytes(StreamAddOutgoingParam(request_sequence=42, new_streams=2))),
            ]
            with patch.object(
                server.transport, "_send_data", wraps=server.transport._send_data
            ) as send:
                await server._handle_data(
                    serialize_packet(
                        client.port, server.port, server._local_verification_tag, chunk
                    )
                )
                self.assertEqual(server.maxChannels, 18)
                self.assertEqual(
                    [call.args[0][12:] for call in send.await_args_list],
                    [bytes.fromhex("820000100010000c0000002a00000001")],
                )
            local.send("after reconfiguration")
            self.assertEqual(
                await asyncio.wait_for(received.get(), 1), "after reconfiguration"
            )
            self.assertTrue(received.empty())

    @asynctest
    async def test_duplicate_cookie_echo_keeps_open_channels_usable(self) -> None:
        async with client_and_server() as (client, server):
            local = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )
            remote = RTCDataChannel(
                server, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )
            ordinary = RTCDataChannel(server, RTCDataChannelParameters(label="in-band"))
            opened = MagicMock()
            remote.on("open", opened)
            received: asyncio.Queue[str] = asyncio.Queue()
            remote.on("message", received.put_nowait)
            with patch.object(
                client.transport, "_send_data", wraps=client.transport._send_data
            ) as send:
                await server.start(client.getCapabilities(), client.port)
                await client.start(server.getCapabilities(), server.port)
                await wait_for_outcome(client, server)
                await asyncio.sleep(0.1)
                echoes = [
                    call.args[0]
                    for call in send.await_args_list
                    if call.args[0][12] == 10
                ]
            self.assertEqual(len(echoes), 1)
            self.assertEqual(ordinary.readyState, "open")
            opened.assert_called_once_with()
            with patch.object(
                server.transport, "_send_data", wraps=server.transport._send_data
            ) as send:
                await server._handle_data(echoes[0])
                self.assertEqual(
                    [call.args[0][12:] for call in send.await_args_list],
                    [b"\x0b\x00\x00\x04"],
                )
            self.assertEqual(remote.readyState, "open")
            self.assertEqual(ordinary.readyState, "open")
            opened.assert_called_once_with()
            local.send("after duplicate cookie")
            self.assertEqual(
                await asyncio.wait_for(received.get(), 1), "after duplicate cookie"
            )
            self.assertTrue(received.empty())

    @asynctest
    async def test_abandon_fragment_after_first_fragment_acknowledged(self) -> None:
        async with client_and_server() as (client, server):
            client._local_tsn = 1
            client._last_sacked_tsn = 0
            client._advanced_peer_ack_tsn = 0
            server._advertised_rwnd = 131072
            local = RTCDataChannel(
                client,
                RTCDataChannelParameters(
                    label="chat", negotiated=True, id=100, maxRetransmits=0
                ),
            )
            remote = RTCDataChannel(
                server, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )
            received: asyncio.Queue[bytes] = asyncio.Queue()
            remote.on("message", received.put_nowait)
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)

            with (
                patch.object(
                    client.transport, "_send_data", new_callable=AsyncMock
                ) as tx,
                patch.object(
                    server.transport, "_send_data", new_callable=AsyncMock
                ) as rx,
            ):
                local.send(b"M" * 2320)
                await asyncio.sleep(0)
                self.assertEqual(
                    [call.args[0][12:] for call in tx.await_args_list],
                    [
                        bytes.fromhex("00020498000000010064000000000035") + b"M" * 1160,
                        bytes.fromhex("00010498000000020064000000000035") + b"M" * 1160,
                    ],
                )
                await server._handle_data(tx.await_args_list[0].args[0])
                self.assertEqual(
                    rx.await_args.args[0][12:],
                    bytes.fromhex("03000010000000010001fb7800000000"),
                )
                await client._handle_data(rx.await_args.args[0])
                client._t3_cancel()
                client._t3_expired()
                await asyncio.sleep(0)
                self.assertEqual(tx.await_count, 3)
                self.assertEqual(
                    tx.await_args.args[0][12:],
                    bytes.fromhex("c000000c0000000200640000"),
                )
                await server._handle_data(tx.await_args.args[0])
                self.assertEqual(
                    rx.await_args.args[0][12:],
                    bytes.fromhex("03000010000000020002000000000000"),
                )
                await client._handle_data(rx.await_args.args[0])
                self.assertTrue(received.empty())
                local.send(b"after")
                await asyncio.sleep(0)
                self.assertEqual(
                    tx.await_args.args[0][12:],
                    bytes.fromhex("00030015000000030064000100000035")
                    + b"after\x00\x00\x00",
                )
                await server._handle_data(tx.await_args.args[0])
                self.assertEqual(await asyncio.wait_for(received.get(), 1), b"after")
                self.assertTrue(received.empty())

    @asynctest
    async def test_abandon_message_with_queued_fragments(self) -> None:
        for ordered in (True, False):
            for timed in (False, True):
                for first_tsn, acknowledge_prefix, last_tsn in (
                    (1, False, 4),
                    (1, True, 6),
                    (0xFFFFFFFD, False, 0),
                    (0xFFFFFFFD, True, 2),
                ):
                    with self.subTest(
                        ordered=ordered,
                        timed=timed,
                        first_tsn=first_tsn,
                        acknowledge_prefix=acknowledge_prefix,
                    ):
                        async with client_and_server() as (client, server):
                            client._local_tsn = first_tsn
                            client._last_sacked_tsn = (first_tsn - 1) & 0xFFFFFFFF
                            client._advanced_peer_ack_tsn = client._last_sacked_tsn
                            server._advertised_rwnd = 131072
                            local = RTCDataChannel(
                                client,
                                RTCDataChannelParameters(
                                    label="partial",
                                    negotiated=True,
                                    id=100,
                                    ordered=ordered,
                                    maxPacketLifeTime=10000 if timed else None,
                                    maxRetransmits=None if timed else 0,
                                ),
                            )
                            remote = RTCDataChannel(
                                server,
                                RTCDataChannelParameters(
                                    label="partial", negotiated=True, id=100
                                ),
                            )
                            reliable = RTCDataChannel(
                                client,
                                RTCDataChannelParameters(
                                    label="reliable", negotiated=True, id=102
                                ),
                            )
                            remote_reliable = RTCDataChannel(
                                server,
                                RTCDataChannelParameters(
                                    label="reliable", negotiated=True, id=102
                                ),
                            )
                            received: asyncio.Queue[bytes] = asyncio.Queue()
                            remote.on("message", received.put_nowait)
                            remote_reliable.on("message", received.put_nowait)
                            await server.start(client.getCapabilities(), client.port)
                            await client.start(server.getCapabilities(), server.port)
                            await wait_for_outcome(client, server)

                            with (
                                patch.object(
                                    client.transport,
                                    "_send_data",
                                    new_callable=AsyncMock,
                                ) as tx,
                                patch.object(
                                    server.transport,
                                    "_send_data",
                                    new_callable=AsyncMock,
                                ) as rx,
                            ):
                                fragments = 6 if acknowledge_prefix else 4
                                local.send(b"M" * 1160 * fragments)
                                await asyncio.sleep(0)
                                self.assertEqual(tx.await_count, 3)
                                self.assertEqual(client._flight_size, 3480)
                                self.assertEqual(local.bufferedAmount, 0)
                                await server._handle_data(tx.await_args_list[0].args[0])
                                self.assertEqual(server._advertised_rwnd, 129912)
                                if acknowledge_prefix:
                                    await client._handle_data(rx.await_args.args[0])
                                    self.assertEqual(tx.await_count, 5)
                                    self.assertEqual(client._flight_size, 4640)
                                sent_count = tx.await_count
                                self.assertEqual(
                                    queued_tsns(client),
                                    [
                                        (first_tsn + offset) & 0xFFFFFFFF
                                        for offset in range(sent_count, fragments)
                                    ],
                                )
                                local.send(b"after")
                                reliable.send(b"other")
                                await asyncio.sleep(0)
                                self.assertEqual(local.bufferedAmount, 5)
                                self.assertEqual(reliable.bufferedAmount, 5)
                                self.assertEqual(tx.await_count, sent_count)

                                client._t3_cancel()
                                with patch(
                                    "aiortc.rtcsctptransport.time.time",
                                    return_value=time.time() + 20,
                                ):
                                    client._t3_expired()
                                await asyncio.sleep(0)
                                forward = (
                                    bytes.fromhex("c000000c")
                                    + last_tsn.to_bytes(4, "big")
                                    + bytes.fromhex("00640000")
                                    if ordered
                                    else bytes.fromhex("c0000008")
                                    + last_tsn.to_bytes(4, "big")
                                )
                                self.assertEqual(
                                    [
                                        call.args[0][12:]
                                        for call in tx.await_args_list[sent_count:]
                                    ],
                                    [forward],
                                )
                                self.assertEqual(outstanding_tsns(client), [])
                                self.assertEqual(queued_tsns(client), [])
                                self.assertEqual(client._flight_size, 0)
                                self.assertEqual(client._cwnd, 1160)
                                self.assertEqual(
                                    client._advanced_peer_ack_tsn, last_tsn
                                )
                                self.assertIsNotNone(client._t3_handle)

                                await server._handle_data(tx.await_args.args[0])
                                self.assertEqual(
                                    rx.await_args.args[0][12:],
                                    bytes.fromhex("03000010")
                                    + last_tsn.to_bytes(4, "big")
                                    + bytes.fromhex("0002000000000000"),
                                )
                                self.assertTrue(received.empty())
                                self.assertEqual(server._advertised_rwnd, 131072)
                                await client._handle_data(rx.await_args.args[0])
                                self.assertEqual(client._last_sacked_tsn, last_tsn)
                                self.assertEqual(local.bufferedAmount, 0)
                                self.assertEqual(reliable.bufferedAmount, 0)
                                self.assertEqual(client._flight_size, 10)
                                later = tx.await_args_list[sent_count + 1 :]
                                self.assertEqual(
                                    [call.args[0][12:] for call in later],
                                    [
                                        bytes.fromhex(
                                            "00030015" if ordered else "00070015"
                                        )
                                        + ((last_tsn + 1) & 0xFFFFFFFF).to_bytes(
                                            4, "big"
                                        )
                                        + bytes.fromhex(
                                            "0064000100000035"
                                            if ordered
                                            else "0064000000000035"
                                        )
                                        + b"after\x00\x00\x00",
                                        bytes.fromhex("00030015")
                                        + ((last_tsn + 2) & 0xFFFFFFFF).to_bytes(
                                            4, "big"
                                        )
                                        + bytes.fromhex("0066000000000035")
                                        + b"other\x00\x00\x00",
                                    ],
                                )
                                for call in later:
                                    await server._handle_data(call.args[0])
                                    await client._handle_data(rx.await_args.args[0])
                                self.assertEqual(
                                    [await received.get(), await received.get()],
                                    [b"after", b"other"],
                                )
                                self.assertTrue(received.empty())
                                self.assertEqual(client._flight_size, 0)
                                self.assertEqual(outstanding_tsns(client), [])
                                self.assertEqual(queued_tsns(client), [])
                                self.assertEqual(server._advertised_rwnd, 131072)
                                self.assertIsNone(client._t3_handle)

    @asynctest
    async def test_abandon_queued_fragments_behind_reliable_message(self) -> None:
        async with client_and_server() as (client, server):
            client._local_tsn = 1
            client._last_sacked_tsn = 0
            client._advanced_peer_ack_tsn = 0
            reliable = RTCDataChannel(
                client,
                RTCDataChannelParameters(label="reliable", negotiated=True, id=102),
            )
            remote_reliable = RTCDataChannel(
                server,
                RTCDataChannelParameters(label="reliable", negotiated=True, id=102),
            )
            local = RTCDataChannel(
                client,
                RTCDataChannelParameters(
                    label="partial", negotiated=True, id=100, maxRetransmits=0
                ),
            )
            remote = RTCDataChannel(
                server,
                RTCDataChannelParameters(label="partial", negotiated=True, id=100),
            )
            received: asyncio.Queue[bytes] = asyncio.Queue()
            remote_reliable.on("message", received.put_nowait)
            remote.on("message", received.put_nowait)
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)
            with (
                patch.object(
                    client.transport, "_send_data", new_callable=AsyncMock
                ) as tx,
                patch.object(
                    server.transport, "_send_data", new_callable=AsyncMock
                ) as rx,
            ):
                reliable.send(b"P" * 1160)
                local.send(b"M" * 4640)
                local.send(b"after")
                await asyncio.sleep(0)
                self.assertEqual(tx.await_count, 3)
                self.assertEqual(queued_tsns(client), [4, 5])
                client._t3_cancel()
                client._t3_expired()
                await asyncio.sleep(0)
                self.assertEqual(
                    [call.args[0][12:] for call in tx.await_args_list[3:]],
                    [bytes.fromhex("00030498000000010066000000000035") + b"P" * 1160],
                )
                self.assertEqual(client._advanced_peer_ack_tsn, 0)
                self.assertEqual(client._flight_size, 1160)
                await server._handle_data(tx.await_args.args[0])
                await client._handle_data(rx.await_args.args[0])
                self.assertEqual(
                    [call.args[0][12:] for call in tx.await_args_list[4:]],
                    [
                        bytes.fromhex("c000000c0000000500640000"),
                        bytes.fromhex("00030015000000060064000100000035")
                        + b"after\x00\x00\x00",
                    ],
                )
                self.assertEqual(client._advanced_peer_ack_tsn, 5)
                self.assertEqual(client._flight_size, 5)
                self.assertIsNotNone(client._t3_handle)
                for call in tx.await_args_list[4:]:
                    await server._handle_data(call.args[0])
                    await client._handle_data(rx.await_args.args[0])
                self.assertEqual(
                    [await received.get(), await received.get()],
                    [b"P" * 1160, b"after"],
                )
                self.assertTrue(received.empty())
                self.assertEqual(client._last_sacked_tsn, 6)
                self.assertEqual(client._flight_size, 0)
                self.assertEqual(outstanding_tsns(client), [])
                self.assertEqual(queued_tsns(client), [])
                self.assertEqual(local.bufferedAmount, 0)
                self.assertIsNone(client._t3_handle)

    @asynctest
    async def test_fast_abandon_message_with_queued_fragments(self) -> None:
        async with client_and_server() as (client, server):
            client._local_tsn = 1
            client._last_sacked_tsn = 0
            client._advanced_peer_ack_tsn = 0
            server._advertised_rwnd = 131072
            local = RTCDataChannel(
                client,
                RTCDataChannelParameters(
                    label="partial", negotiated=True, id=100, maxRetransmits=0
                ),
            )
            remote = RTCDataChannel(
                server,
                RTCDataChannelParameters(label="partial", negotiated=True, id=100),
            )
            received: asyncio.Queue[bytes] = asyncio.Queue()
            remote.on("message", received.put_nowait)
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)
            with (
                patch.object(
                    client.transport, "_send_data", new_callable=AsyncMock
                ) as tx,
                patch.object(
                    server.transport, "_send_data", new_callable=AsyncMock
                ) as rx,
            ):
                local.send(b"M" * 8120)
                local.send(b"after")
                await asyncio.sleep(0)
                timer = client._t3_handle
                self.assertEqual(tx.await_count, 3)
                for index in range(1, 4):
                    await server._handle_data(tx.await_args_list[index].args[0])
                    await client._handle_data(rx.await_args.args[0])
                self.assertEqual(
                    [call.args[0][12:] for call in tx.await_args_list[5:]],
                    [
                        bytes.fromhex("c000000c0000000700640000"),
                        bytes.fromhex("00030015000000080064000100000035")
                        + b"after\x00\x00\x00",
                    ],
                )
                self.assertEqual(client._flight_size, 5)
                self.assertEqual(outstanding_tsns(client), [8])
                self.assertEqual(queued_tsns(client), [])
                self.assertIs(client._t3_handle, timer)
                self.assertTrue(received.empty())
                for call in tx.await_args_list[5:]:
                    await server._handle_data(call.args[0])
                    await client._handle_data(rx.await_args.args[0])
                self.assertEqual(await asyncio.wait_for(received.get(), 1), b"after")
                self.assertTrue(received.empty())
                self.assertEqual(server._advertised_rwnd, 131072)
                self.assertEqual(client._flight_size, 0)
                self.assertEqual(client._last_sacked_tsn, 8)
                self.assertEqual(outstanding_tsns(client), [])
                self.assertIsNone(client._t3_handle)

    @asynctest
    async def test_abandoned_fragments_late_ack_preserves_other_flight_bytes(
        self,
    ) -> None:
        for gap_ack in (False, True):
            with self.subTest(gap_ack=gap_ack):
                async with client_and_server() as (client, server):
                    client._local_tsn = 1
                    client._last_sacked_tsn = 0
                    client._advanced_peer_ack_tsn = 0
                    server._advertised_rwnd = 131072
                    reliable = RTCDataChannel(
                        client,
                        RTCDataChannelParameters(
                            label="reliable", negotiated=True, id=102
                        ),
                    )
                    remote_reliable = RTCDataChannel(
                        server,
                        RTCDataChannelParameters(
                            label="reliable", negotiated=True, id=102
                        ),
                    )
                    local = RTCDataChannel(
                        client,
                        RTCDataChannelParameters(
                            label="partial", negotiated=True, id=100, maxRetransmits=0
                        ),
                    )
                    remote = RTCDataChannel(
                        server,
                        RTCDataChannelParameters(
                            label="partial", negotiated=True, id=100
                        ),
                    )
                    received: asyncio.Queue[bytes] = asyncio.Queue()
                    remote_reliable.on("message", received.put_nowait)
                    remote.on("message", received.put_nowait)
                    await server.start(client.getCapabilities(), client.port)
                    await client.start(server.getCapabilities(), server.port)
                    await wait_for_outcome(client, server)
                    with (
                        patch.object(
                            client.transport, "_send_data", new_callable=AsyncMock
                        ) as tx,
                        patch.object(
                            server.transport, "_send_data", new_callable=AsyncMock
                        ) as rx,
                    ):
                        reliable.send(b"P" * 1160)
                        local.send(b"M" * 8120)
                        local.send(b"after")
                        await asyncio.sleep(0)
                        for index in range(2, 5):
                            await server._handle_data(tx.await_args_list[index].args[0])
                            await client._handle_data(rx.await_args.args[0])
                        self.assertEqual(
                            [call.args[0][12:] for call in tx.await_args_list[5:]],
                            [
                                bytes.fromhex("00030498000000010066000000000035")
                                + b"P" * 1160,
                                bytes.fromhex("00030015000000090064000100000035")
                                + b"after\x00\x00\x00",
                            ],
                        )
                        self.assertEqual(client._flight_size, 1165)
                        self.assertEqual(client._advanced_peer_ack_tsn, 0)
                        await server._handle_data(tx.await_args_list[1].args[0])
                        if gap_ack:
                            await client._handle_data(rx.await_args.args[0])
                            self.assertEqual(client._flight_size, 1165)
                        await server._handle_data(tx.await_args_list[5].args[0])
                        await client._handle_data(rx.await_args.args[0])
                        self.assertEqual(client._flight_size, 5)
                        self.assertEqual(
                            [call.args[0][12:] for call in tx.await_args_list[7:]],
                            [bytes.fromhex("c000000c0000000800640000")],
                        )
                        await server._handle_data(tx.await_args_list[7].args[0])
                        await client._handle_data(rx.await_args.args[0])
                        self.assertEqual(client._flight_size, 5)
                        self.assertEqual(server._advertised_rwnd, 131072)
                        await server._handle_data(tx.await_args_list[6].args[0])
                        await client._handle_data(rx.await_args.args[0])
                        self.assertEqual(
                            [await received.get(), await received.get()],
                            [b"P" * 1160, b"after"],
                        )
                        self.assertTrue(received.empty())
                        self.assertEqual(client._last_sacked_tsn, 9)
                        self.assertEqual(client._flight_size, 0)
                        self.assertEqual(outstanding_tsns(client), [])
                        self.assertEqual(queued_tsns(client), [])
                        self.assertEqual(local.bufferedAmount, 0)
                        self.assertIsNone(client._t3_handle)

    @asynctest
    async def test_fast_abandon_sends_forward_tsn_without_restarting_t3(self) -> None:
        async with client_and_server() as (client, server):
            client._local_tsn = 1
            client._last_sacked_tsn = 0
            client._advanced_peer_ack_tsn = 0
            server._advertised_rwnd = 131072
            lost = RTCDataChannel(
                client,
                RTCDataChannelParameters(
                    label="unreliable", negotiated=True, id=100, maxRetransmits=0
                ),
            )
            remote_lost = RTCDataChannel(
                server,
                RTCDataChannelParameters(label="unreliable", negotiated=True, id=100),
            )
            reliable = RTCDataChannel(
                client,
                RTCDataChannelParameters(label="reliable", negotiated=True, id=102),
            )
            remote_reliable = RTCDataChannel(
                server,
                RTCDataChannelParameters(label="reliable", negotiated=True, id=102),
            )
            received: asyncio.Queue[bytes] = asyncio.Queue()
            remote_lost.on("message", received.put_nowait)
            remote_reliable.on("message", received.put_nowait)
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)

            with (
                patch.object(
                    client.transport, "_send_data", new_callable=AsyncMock
                ) as tx,
                patch.object(
                    server.transport, "_send_data", new_callable=AsyncMock
                ) as rx,
            ):
                lost.send(b"X" * 1160)
                for payload in (b"A", b"B", b"C"):
                    reliable.send(payload * 1160)
                await asyncio.sleep(0)
                timer = client._t3_handle
                self.assertIsNotNone(timer)
                self.assertEqual(tx.await_count, 3)
                for index in range(1, 4):
                    await server._handle_data(tx.await_args_list[index].args[0])
                    self.assertEqual(
                        rx.await_args.args[0][12:],
                        bytes.fromhex("030000140000000000020000000100000002")
                        + (index + 1).to_bytes(2, "big"),
                    )
                    await client._handle_data(rx.await_args.args[0])
                self.assertEqual(
                    [call.args[0][12:] for call in tx.await_args_list],
                    [
                        bytes.fromhex("00030498000000010064000000000035") + b"X" * 1160,
                        bytes.fromhex("00030498000000020066000000000035") + b"A" * 1160,
                        bytes.fromhex("00030498000000030066000100000035") + b"B" * 1160,
                        bytes.fromhex("00030498000000040066000200000035") + b"C" * 1160,
                        bytes.fromhex("c000000c0000000100640000"),
                    ],
                )
                self.assertIs(client._t3_handle, timer)
                await server._handle_data(tx.await_args.args[0])
                self.assertEqual(
                    rx.await_args.args[0][12:],
                    bytes.fromhex("03000010000000040002000000000000"),
                )
                await client._handle_data(rx.await_args.args[0])
                self.assertIsNone(client._t3_handle)
                self.assertEqual(
                    [await asyncio.wait_for(received.get(), 1) for _ in range(3)],
                    [b"A" * 1160, b"B" * 1160, b"C" * 1160],
                )
                self.assertTrue(received.empty())

    @asynctest
    async def test_close_multiple_channels_repeatedly(self) -> None:
        async with client_and_server() as (client, server):
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)
            channels = [
                RTCDataChannel(client, RTCDataChannelParameters(label=label))
                for label in ("first", "second")
            ]
            await asyncio.sleep(0.1)
            self.assertEqual(
                [channel.readyState for channel in channels], ["open", "open"]
            )
            closed = [asyncio.Event(), asyncio.Event()]
            for channel, event in zip(channels, closed):
                channel.on("close", event.set)
                channel.close()
                channel.close()
            await asyncio.gather(
                *(asyncio.wait_for(event.wait(), 1) for event in closed)
            )
            self.assertEqual(
                [channel.readyState for channel in channels], ["closed", "closed"]
            )
            for channel in channels:
                channel.close()

    @asynctest
    async def test_duplicate_ack_and_unknown_channel_control(self) -> None:
        async with client_and_server() as (client, server):
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)
            await wait_for_outcome(client, server)
            channel = RTCDataChannel(client, RTCDataChannelParameters(label="chat"))
            opened = MagicMock()
            channel.on("open", opened)
            await asyncio.sleep(0.1)
            self.assertEqual(channel.readyState, "open")
            opened.assert_called_once_with()
            for sequence, control in enumerate((b"\x02", b"\xff"), start=1):
                chunk = DataChunk(flags=SCTP_DATA_FIRST_FRAG | SCTP_DATA_LAST_FRAG)
                chunk.tsn = tsn_plus_one(client._last_received_tsn)
                chunk.stream_id = channel.id
                chunk.stream_seq = sequence
                chunk.protocol = 50
                chunk.user_data = control
                await client._handle_data(
                    serialize_packet(
                        server.port, client.port, client._local_verification_tag, chunk
                    )
                )
            self.assertEqual(channel.readyState, "open")
            opened.assert_called_once_with()

    @asynctest
    async def test_expired_unordered_message_advances_tsn_without_stream_sequence(
        self,
    ) -> None:
        async with client_standalone() as client:
            client._local_tsn = 1
            client._last_sacked_tsn = 0
            client._advanced_peer_ack_tsn = 0
            with patch.object(client, "_send_chunk", new_callable=AsyncMock) as send:
                await client._send(
                    123, 456, b"expired", max_retransmits=0, ordered=False
                )
                self.assertEqual(outstanding_tsns(client), [1])
                client._t3_cancel()
                client._t3_expired()
                await asyncio.sleep(0)
                self.assertEqual(outstanding_tsns(client), [])
                self.assertEqual(send.await_count, 2)
                forward = send.await_args_list[1].args[0]
                self.assertIsInstance(forward, ForwardTsnChunk)
                self.assertEqual(forward.cumulative_tsn, 1)
                self.assertEqual(forward.streams, [])

    @contextlib.contextmanager
    def assertTimerPreserved(self, client: RTCSctpTransport) -> Generator[None]:
        previous_timer = client._t3_handle
        yield
        self.assertIsNotNone(client._t3_handle)
        self.assertEqual(client._t3_handle, previous_timer)

    @contextlib.contextmanager
    def assertTimerRestarted(self, client: RTCSctpTransport) -> Generator[None]:
        previous_timer = client._t3_handle
        yield
        self.assertIsNotNone(client._t3_handle)
        self.assertNotEqual(client._t3_handle, previous_timer)

    @contextlib.contextmanager
    def assertTimerStopped(self, client: RTCSctpTransport) -> Generator[None]:
        yield
        self.assertIsNone(client._t3_handle)

    @asynctest
    async def test_construct(self) -> None:
        async with dummy_dtls_transport_pair() as (client_transport, server_transport):
            sctpTransport = RTCSctpTransport(client_transport)
            self.assertEqual(sctpTransport.transport, client_transport)
            self.assertEqual(sctpTransport.port, 5000)
            sctpTransport.setTransport(server_transport)
            self.assertIs(sctpTransport.transport, server_transport)
            self.assertEqual(sctpTransport.port, 5000)

    @asynctest
    async def test_construct_invalid_dtls_transport_state(self) -> None:
        dtlsTransport = ClosedDtlsTransport()
        with self.assertRaises(InvalidStateError):
            RTCSctpTransport(dtlsTransport)  # type: ignore

    @asynctest
    async def test_connect_broken_transport(self) -> None:
        """
        Transport with 100% loss never connects.
        """
        loss_pattern = [True]

        async with client_and_server() as (client, server):
            client._rto = 0.1
            set_loss_pattern(client.transport.transport, loss_pattern)

            server._rto = 0.1
            set_loss_pattern(server.transport.transport, loss_pattern)

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(client._association_state, RTCSctpTransport.State.CLOSED)
            self.assertEqual(client.state, "closed")
            self.assertEqual(server._association_state, RTCSctpTransport.State.CLOSED)
            self.assertEqual(server.state, "closed")

    @asynctest
    async def test_connect_one_side_initiates(self) -> None:
        for initiator in ["client", "server"]:
            async with client_and_server() as (client, server):
                if initiator == "client":
                    server._init = noop_init  # type: ignore
                else:
                    client._init = noop_init  # type: ignore

                await server.start(client.getCapabilities(), client.port)
                await client.start(server.getCapabilities(), server.port)

                await wait_for_outcome(client, server)
                self.assertEqual(
                    client._association_state, RTCSctpTransport.State.ESTABLISHED
                )
                self.assertEqual(
                    server._association_state, RTCSctpTransport.State.ESTABLISHED
                )

    @asynctest
    async def test_connect_peer_ignores_init(self) -> None:
        async with client_and_server() as (client, server):
            real_receive_chunk = client._receive_chunk

            async def mock_receive_chunk(chunk: Chunk) -> None:
                if not isinstance(chunk, InitChunk):
                    await real_receive_chunk(chunk)

            client._receive_chunk = mock_receive_chunk  # type: ignore

            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertIsNone(server._t1_handle)

    @asynctest
    async def test_connect_lossy_transport(self) -> None:
        """
        Transport with 25% loss eventually connects.
        """
        loss_pattern = [True, False, False, False]

        async with client_and_server() as (client, server):
            client._rto = 0.1
            set_loss_pattern(client.transport.transport, loss_pattern)

            server._rto = 0.1
            set_loss_pattern(server.transport.transport, loss_pattern)

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client.state, "connected")
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server.state, "connected")

            # transmit data
            server_queue: asyncio.Queue[tuple[int, int, bytes]] = asyncio.Queue()

            async def mock_receive(stream_id: int, pp_id: int, data: bytes) -> None:
                await server_queue.put((stream_id, pp_id, data))

            server._receive = mock_receive  # type: ignore

            for i in range(20):
                message = (123, i, b"ping")
                await client._send(*message)
                received = await server_queue.get()
                self.assertEqual(received, message)

    @asynctest
    async def test_connect_client_limits_streams(self) -> None:
        async with client_and_server() as (client, server):
            client._inbound_streams_max = 2048
            client._outbound_streams_count = 256
            self.assertEqual(client.maxChannels, None)
            self.assertEqual(server.maxChannels, None)

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(client.maxChannels, 256)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._inbound_streams_count, 2048)
            self.assertEqual(client._outbound_streams_count, 256)
            self.assertEqual(client._remote_extensions, [192, 130])
            self.assertEqual(server.maxChannels, 256)
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._inbound_streams_count, 256)
            self.assertEqual(server._outbound_streams_count, 2048)
            self.assertEqual(server._remote_extensions, [192, 130])

            # client requests additional outbound streams
            param = StreamAddOutgoingParam(
                request_sequence=client._reconfig_request_seq, new_streams=16
            )
            await client._send_reconfig_param(param)

            await asyncio.sleep(0.1)

            self.assertEqual(server.maxChannels, 272)
            self.assertEqual(server._inbound_streams_count, 272)
            self.assertEqual(server._outbound_streams_count, 2048)

    @asynctest
    async def test_connect_server_limits_streams(self) -> None:
        async with client_and_server() as (client, server):
            server._inbound_streams_max = 2048
            server._outbound_streams_count = 256

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(client.maxChannels, 256)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._inbound_streams_count, 256)
            self.assertEqual(client._outbound_streams_count, 2048)
            self.assertEqual(client._remote_extensions, [192, 130])
            self.assertEqual(server.maxChannels, 256)
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._inbound_streams_count, 2048)
            self.assertEqual(server._outbound_streams_count, 256)
            self.assertEqual(server._remote_extensions, [192, 130])

            await asyncio.sleep(0.1)

    @asynctest
    async def test_connect_then_client_creates_data_channel(self) -> None:
        async with client_and_server() as (client, server):
            self.assertFalse(client.is_server)
            self.assertEqual(client.maxChannels, None)
            self.assertTrue(server.is_server)
            self.assertEqual(server.maxChannels, None)

            client_channels = track_channels(client)
            server_channels = track_channels(server)

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(client.maxChannels, 65535)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._inbound_streams_count, 65535)
            self.assertEqual(client._outbound_streams_count, 65535)
            self.assertEqual(client._remote_extensions, [192, 130])
            self.assertEqual(server.maxChannels, 65535)
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._inbound_streams_count, 65535)
            self.assertEqual(server._outbound_streams_count, 65535)
            self.assertEqual(server._remote_extensions, [192, 130])

            # create data channel
            channel = RTCDataChannel(client, RTCDataChannelParameters(label="chat"))
            self.assertEqual(channel.id, None)
            self.assertEqual(channel.label, "chat")

            await asyncio.sleep(0.1)
            self.assertEqual(channel.id, 0)
            self.assertEqual(channel.label, "chat")
            self.assertEqual(len(client_channels), 0)
            self.assertEqual(len(server_channels), 1)
            self.assertEqual(server_channels[0].id, 0)
            self.assertEqual(server_channels[0].label, "chat")

    @asynctest
    async def test_connect_then_send_message_too_large(self) -> None:
        async with client_and_server() as (client, server):
            # connect, the server only accepts messages of up to 1000 bytes
            await server.start(client.getCapabilities(), client.port)
            await client.start(RTCSctpCapabilities(maxMessageSize=1000), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )

            # create data channel
            channel = RTCDataChannel(client, RTCDataChannelParameters(label="chat"))
            await asyncio.sleep(0.1)
            self.assertEqual(channel.readyState, "open")

            # a message which fits is accepted
            channel.send(b"M" * 1000)

            # a larger message is rejected and is not buffered
            buffered_amount = channel.bufferedAmount
            with self.assertRaises(ValueError) as cm:
                channel.send(b"M" * 1001)
            self.assertEqual(
                str(cm.exception),
                "Cannot send message of 1001 bytes, the remote party only supports "
                "messages up to 1000 bytes",
            )
            self.assertEqual(channel.bufferedAmount, buffered_amount)

    @asynctest
    async def test_connect_then_send_message_unlimited(self) -> None:
        async with client_and_server() as (client, server):
            # connect, the server accepts messages of any size
            await server.start(client.getCapabilities(), client.port)
            await client.start(RTCSctpCapabilities(maxMessageSize=0), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )

            # create data channel
            channel = RTCDataChannel(client, RTCDataChannelParameters(label="chat"))
            await asyncio.sleep(0.1)
            self.assertEqual(channel.readyState, "open")

            # a large message is accepted
            channel.send(b"M" * (2 * LOCAL_MAX_MESSAGE_SIZE))

    @asynctest
    async def test_connect_without_remote_capabilities(self) -> None:
        async with client_and_server() as (client, server):
            # connect, the server did not advertise a maximum message size
            await server.start(client.getCapabilities(), client.port)
            await client.start(None, server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(
                client._remote_max_message_size, REMOTE_DEFAULT_MAX_MESSAGE_SIZE
            )

    @asynctest
    async def test_connect_then_client_creates_data_channel_with_custom_id(
        self,
    ) -> None:
        async with client_and_server() as (client, server):
            client_channels = track_channels(client)
            server_channels = track_channels(server)

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._inbound_streams_count, 65535)
            self.assertEqual(client._outbound_streams_count, 65535)
            self.assertEqual(client._remote_extensions, [192, 130])
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._inbound_streams_count, 65535)
            self.assertEqual(server._outbound_streams_count, 65535)
            self.assertEqual(server._remote_extensions, [192, 130])

            # create data channel
            channel = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", id=100)
            )
            self.assertEqual(channel.id, 100)
            self.assertEqual(channel.label, "chat")

            # create second data channel
            channel2 = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", id=101)
            )
            self.assertEqual(channel2.id, 101)
            self.assertEqual(channel2.label, "chat")

            await asyncio.sleep(0.1)
            self.assertEqual(channel.id, 100)
            self.assertEqual(channel.label, "chat")
            self.assertEqual(channel2.id, 101)
            self.assertEqual(channel2.label, "chat")
            self.assertEqual(len(client_channels), 0)
            self.assertEqual(len(server_channels), 2)
            self.assertEqual(server_channels[0].id, 100)
            self.assertEqual(server_channels[0].label, "chat")
            self.assertEqual(server_channels[1].id, 101)
            self.assertEqual(server_channels[1].label, "chat")

    @asynctest
    async def test_connect_then_client_creates_data_channel_with_custom_id_and_then_normal(  # noqa: E501
        self,
    ) -> None:
        async with client_and_server() as (client, server):
            client_channels = track_channels(client)
            server_channels = track_channels(server)

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._inbound_streams_count, 65535)
            self.assertEqual(client._outbound_streams_count, 65535)
            self.assertEqual(client._remote_extensions, [192, 130])
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._inbound_streams_count, 65535)
            self.assertEqual(server._outbound_streams_count, 65535)
            self.assertEqual(server._remote_extensions, [192, 130])

            # create data channel
            channel = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", id=1)
            )
            self.assertEqual(channel.id, 1)
            self.assertEqual(channel.label, "chat")

            # create second data channel
            channel2 = RTCDataChannel(client, RTCDataChannelParameters(label="chat"))
            self.assertEqual(channel2.id, None)
            self.assertEqual(channel2.label, "chat")

            await asyncio.sleep(0.1)
            self.assertEqual(channel.id, 1)
            self.assertEqual(channel.label, "chat")
            self.assertEqual(channel2.id, 0)
            self.assertEqual(channel2.label, "chat")
            self.assertEqual(len(client_channels), 0)
            self.assertEqual(len(server_channels), 2)
            self.assertEqual(server_channels[0].id, 1)
            self.assertEqual(server_channels[0].label, "chat")
            self.assertEqual(server_channels[1].id, 0)
            self.assertEqual(server_channels[1].label, "chat")

    @asynctest
    async def test_connect_then_client_creates_second_data_channel_with_custom_already_used_id(  # noqa: E501
        self,
    ) -> None:
        async with client_and_server() as (client, server):
            client_channels = track_channels(client)
            server_channels = track_channels(server)

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._inbound_streams_count, 65535)
            self.assertEqual(client._outbound_streams_count, 65535)
            self.assertEqual(client._remote_extensions, [192, 130])
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._inbound_streams_count, 65535)
            self.assertEqual(server._outbound_streams_count, 65535)
            self.assertEqual(server._remote_extensions, [192, 130])

            # create data channel
            channel = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", id=100)
            )
            self.assertEqual(channel.id, 100)
            self.assertEqual(channel.label, "chat")

            # create second data channel with the same id
            self.assertRaises(
                ValueError,
                lambda: RTCDataChannel(
                    client, RTCDataChannelParameters(label="chat", id=100)
                ),
            )

            await asyncio.sleep(0.1)
            self.assertEqual(channel.id, 100)
            self.assertEqual(channel.label, "chat")
            self.assertEqual(len(client_channels), 0)
            self.assertEqual(len(server_channels), 1)
            self.assertEqual(server_channels[0].id, 100)
            self.assertEqual(server_channels[0].label, "chat")

    @asynctest
    async def test_connect_then_client_creates_negotiated_data_channel_without_id(
        self,
    ) -> None:
        async with client_and_server() as (client, server):
            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._inbound_streams_count, 65535)
            self.assertEqual(client._outbound_streams_count, 65535)
            self.assertEqual(client._remote_extensions, [192, 130])
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._inbound_streams_count, 65535)
            self.assertEqual(server._outbound_streams_count, 65535)
            self.assertEqual(server._remote_extensions, [192, 130])

            # create data channel
            self.assertRaises(
                ValueError,
                lambda: RTCDataChannel(
                    client, RTCDataChannelParameters(label="chat", negotiated=True)
                ),
            )

            await asyncio.sleep(0.1)

    @asynctest
    async def test_connect_then_client_and_server_creates_negotiated_data_channel(
        self,
    ) -> None:
        async with client_and_server() as (client, server):
            client_channels = track_channels(client)
            server_channels = track_channels(server)

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._inbound_streams_count, 65535)
            self.assertEqual(client._outbound_streams_count, 65535)
            self.assertEqual(client._remote_extensions, [192, 130])
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._inbound_streams_count, 65535)
            self.assertEqual(server._outbound_streams_count, 65535)
            self.assertEqual(server._remote_extensions, [192, 130])

            # create data channel for client
            channel_client = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )
            self.assertEqual(channel_client.id, 100)
            self.assertEqual(channel_client.label, "chat")

            # create data channel for server
            channel_server = RTCDataChannel(
                server, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )
            self.assertEqual(channel_server.id, 100)
            self.assertEqual(channel_server.label, "chat")

            await asyncio.sleep(0.1)
            self.assertEqual(channel_client.id, 100)
            self.assertEqual(channel_client.label, "chat")
            self.assertEqual(channel_server.id, 100)
            self.assertEqual(channel_server.label, "chat")
            # both arrays should be 0 as they track data channels created by event
            #   which is not the case in out-of-band
            self.assertEqual(len(client_channels), 0)
            self.assertEqual(len(server_channels), 0)

    @asynctest
    async def test_connect_then_client_creates_negotiated_data_channel_with_used_id(
        self,
    ) -> None:
        async with client_and_server() as (client, server):
            client_channels = track_channels(client)
            server_channels = track_channels(server)

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._inbound_streams_count, 65535)
            self.assertEqual(client._outbound_streams_count, 65535)
            self.assertEqual(client._remote_extensions, [192, 130])
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._inbound_streams_count, 65535)
            self.assertEqual(server._outbound_streams_count, 65535)
            self.assertEqual(server._remote_extensions, [192, 130])

            # create data channel for client
            channel_client = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )
            self.assertEqual(channel_client.id, 100)
            self.assertEqual(channel_client.label, "chat")

            self.assertRaises(
                ValueError,
                lambda: RTCDataChannel(
                    client,
                    RTCDataChannelParameters(label="chat", negotiated=True, id=100),
                ),
            )

            await asyncio.sleep(0.1)
            self.assertEqual(channel_client.id, 100)
            self.assertEqual(channel_client.label, "chat")
            # both arrays should be 0 as they track data channels created by event
            #   which is not the case in out-of-band
            self.assertEqual(len(client_channels), 0)
            self.assertEqual(len(server_channels), 0)

    @asynctest
    async def test_connect_then_client_and_server_creates_negotiated_data_channel_before_transport(  # noqa: E501
        self,
    ) -> None:
        async with client_and_server() as (client, server):
            client_channels = track_channels(client)
            server_channels = track_channels(server)

            self.assertEqual(client._association_state, RTCSctpTransport.State.CLOSED)
            self.assertEqual(server._association_state, RTCSctpTransport.State.CLOSED)

            # create data channel for client
            channel_client = RTCDataChannel(
                client, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )
            self.assertEqual(channel_client.id, 100)
            self.assertEqual(channel_client.label, "chat")
            self.assertEqual(channel_client.readyState, "connecting")

            # create data channel for server
            channel_server = RTCDataChannel(
                server, RTCDataChannelParameters(label="chat", negotiated=True, id=100)
            )
            self.assertEqual(channel_server.id, 100)
            self.assertEqual(channel_server.label, "chat")
            self.assertEqual(channel_server.readyState, "connecting")

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._inbound_streams_count, 65535)
            self.assertEqual(client._outbound_streams_count, 65535)
            self.assertEqual(client._remote_extensions, [192, 130])
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._inbound_streams_count, 65535)
            self.assertEqual(server._outbound_streams_count, 65535)
            self.assertEqual(server._remote_extensions, [192, 130])

            self.assertEqual(channel_client.readyState, "open")
            self.assertEqual(channel_server.readyState, "open")

            await asyncio.sleep(0.1)
            self.assertEqual(channel_client.id, 100)
            self.assertEqual(channel_client.label, "chat")
            self.assertEqual(channel_server.id, 100)
            self.assertEqual(channel_server.label, "chat")
            # both arrays should be 0 as they track data channels created by event
            #   which is not the case in out-of-band
            self.assertEqual(len(client_channels), 0)
            self.assertEqual(len(server_channels), 0)

    @asynctest
    async def test_connect_then_server_creates_data_channel(self) -> None:
        async with client_and_server() as (client, server):
            client_channels = track_channels(client)
            server_channels = track_channels(server)

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._remote_extensions, [192, 130])
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._remote_extensions, [192, 130])

            # create data channel
            channel = RTCDataChannel(server, RTCDataChannelParameters(label="chat"))
            self.assertEqual(channel.id, None)
            self.assertEqual(channel.label, "chat")

            await asyncio.sleep(0.1)
            self.assertEqual(len(client_channels), 1)
            self.assertEqual(client_channels[0].id, 1)
            self.assertEqual(client_channels[0].label, "chat")
            self.assertEqual(len(server_channels), 0)

    @patch("aiortc.rtcsctptransport.logger.isEnabledFor")
    @asynctest
    async def test_connect_with_logging(self, mock_is_enabled_for: MagicMock) -> None:
        mock_is_enabled_for.return_value = True

        async with client_and_server() as (client, server):
            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )

    @asynctest
    async def test_connect_with_partial_reliability(self) -> None:
        async with client_and_server() as (client, server):
            client._local_partial_reliability = True
            server._local_partial_reliability = False

            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(client._remote_extensions, [130])
            self.assertEqual(client._remote_partial_reliability, False)
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(server._remote_extensions, [192, 130])
            self.assertEqual(server._remote_partial_reliability, True)

    @asynctest
    async def test_abrupt_disconnect(self) -> None:
        """
        Abrupt disconnect causes sending ABORT chunk to fail.
        """
        async with client_and_server() as (client, server):
            # connect
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await wait_for_outcome(client, server)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.ESTABLISHED
            )
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.ESTABLISHED
            )

            # break connection
            await client.transport.stop()
            await server.transport.stop()
            await asyncio.sleep(0)  # FIXME

    @asynctest
    async def test_garbage(self) -> None:
        async with client_and_server() as (_, server):
            await server.start(RTCSctpCapabilities(maxMessageSize=65536), 5000)
            await server._handle_data(b"garbage")

            # check outcome
            await asyncio.sleep(0.1)
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.COOKIE_WAIT
            )

    @asynctest
    async def test_bad_verification_tag(self) -> None:
        # verification tag is 12345 instead of 0
        data = load("sctp_init_bad_verification.bin")

        async with client_and_server() as (_, server):
            await server.start(RTCSctpCapabilities(maxMessageSize=65536), 5000)
            await server._handle_data(data)

            # check outcome
            await asyncio.sleep(0.1)
            self.assertEqual(
                server._association_state, RTCSctpTransport.State.COOKIE_WAIT
            )

    @asynctest
    async def test_bad_cookie(self) -> None:
        async with client_and_server() as (client, server):
            # corrupt cookie
            real_send_chunk = client._send_chunk

            async def mock_send_chunk(chunk: Chunk) -> None:
                if isinstance(chunk, CookieEchoChunk):
                    chunk.body = b"garbage"
                return await real_send_chunk(chunk)

            client._send_chunk = mock_send_chunk  # type: ignore
            server._init = noop_init  # type: ignore

            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await asyncio.sleep(0.1)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.COOKIE_ECHOED
            )
            self.assertEqual(server._association_state, RTCSctpTransport.State.CLOSED)

    @asynctest
    async def test_maybe_abandon(self) -> None:
        async with client_standalone() as client:
            client._local_tsn = 0
            client._send_chunk = noop_send_chunk  # type: ignore

            # send 3 chunks
            await client._send(123, 456, b"M" * USERDATA_MAX_LENGTH * 3)
            self.assertEqual(outstanding_tsns(client), [0, 1, 2])
            self.assertEqual(queued_tsns(client), [])
            for chunk in client._outbound_queue:
                self.assertEqual(chunk._abandoned, False)

            # try abandon middle chunk
            client._maybe_abandon(client._sent_queue[1])
            for chunk in client._outbound_queue:
                self.assertEqual(chunk._abandoned, False)

    @asynctest
    async def test_maybe_abandon_max_retransmits(self) -> None:
        async with client_standalone() as client:
            client._local_tsn = 1
            client._last_sacked_tsn = 0
            client._advanced_peer_ack_tsn = 0
            client._send_chunk = noop_send_chunk  # type: ignore

            # send 3 chunks
            await client._send(
                123, 456, b"M" * USERDATA_MAX_LENGTH * 3, max_retransmits=0
            )
            self.assertEqual(outstanding_tsns(client), [1, 2, 3])
            self.assertEqual(queued_tsns(client), [])
            self.assertEqual(client._local_tsn, 4)
            self.assertEqual(client._advanced_peer_ack_tsn, 0)
            for chunk in client._outbound_queue:
                self.assertEqual(chunk._abandoned, False)

            # try abandon middle chunk
            client._maybe_abandon(client._sent_queue[1])
            for chunk in client._outbound_queue:
                self.assertEqual(chunk._abandoned, True)

            # try abandon middle chunk (again)
            client._maybe_abandon(client._sent_queue[1])
            for chunk in client._outbound_queue:
                self.assertEqual(chunk._abandoned, True)

            # update advanced peer ack point
            client._update_advanced_peer_ack_point()
            self.assertEqual(outstanding_tsns(client), [])
            self.assertEqual(queued_tsns(client), [])
            self.assertEqual(client._advanced_peer_ack_tsn, 3)

            # check forward TSN
            self.assertIsNotNone(client._forward_tsn_chunk)
            self.assertEqual(client._forward_tsn_chunk.cumulative_tsn, 3)
            self.assertEqual(client._forward_tsn_chunk.streams, [(123, 0)])

            # transmit
            client._t3_cancel()
            await client._transmit()
            self.assertIsNone(client._forward_tsn_chunk)
            self.assertIsNotNone(client._t3_handle)

    @asynctest
    async def test_stale_cookie(self) -> None:
        mock_timestamp_values = [0, 61]

        def mock_timestamp() -> int:
            return mock_timestamp_values.pop(0)

        async with client_and_server() as (client, server):
            server._get_timestamp = mock_timestamp  # type: ignore
            server._init = noop_init  # type: ignore
            await server.start(client.getCapabilities(), client.port)
            await client.start(server.getCapabilities(), server.port)

            # check outcome
            await asyncio.sleep(0.1)
            self.assertEqual(client._association_state, RTCSctpTransport.State.CLOSED)
            self.assertEqual(server._association_state, RTCSctpTransport.State.CLOSED)

    @asynctest
    async def test_receive_data(self) -> None:
        async with client_standalone() as client:
            client._last_received_tsn = 0

            # receive chunk
            chunk = DataChunk(flags=(SCTP_DATA_FIRST_FRAG | SCTP_DATA_LAST_FRAG))
            chunk.user_data = b"foo"
            chunk.tsn = 1
            await client._receive_chunk(chunk)

            self.assertEqual(client._sack_needed, True)
            self.assertEqual(client._sack_duplicates, [])
            self.assertEqual(client._sack_misordered, set())
            self.assertEqual(client._last_received_tsn, 1)
            client._sack_needed = False

            # receive chunk again
            await client._receive_chunk(chunk)
            self.assertEqual(client._sack_needed, True)
            self.assertEqual(client._sack_duplicates, [1])
            self.assertEqual(client._sack_misordered, set())
            self.assertEqual(client._last_received_tsn, 1)

    @asynctest
    async def test_receive_data_out_of_order(self) -> None:
        async with client_standalone() as client:
            client._last_received_tsn = 0

            # build chunks
            chunks = []

            chunk = DataChunk(flags=SCTP_DATA_FIRST_FRAG)
            chunk.user_data = b"foo"
            chunk.tsn = 1
            chunks.append(chunk)

            chunk = DataChunk()
            chunk.user_data = b"bar"
            chunk.tsn = 2
            chunks.append(chunk)

            chunk = DataChunk(flags=SCTP_DATA_LAST_FRAG)
            chunk.user_data = b"baz"
            chunk.tsn = 3
            chunks.append(chunk)

            # receive first chunk
            await client._receive_chunk(chunks[0])
            self.assertEqual(client._sack_needed, True)
            self.assertEqual(client._sack_duplicates, [])
            self.assertEqual(client._sack_misordered, set())
            self.assertEqual(client._last_received_tsn, 1)
            client._sack_needed = False

            # receive last chunk
            await client._receive_chunk(chunks[2])
            self.assertEqual(client._sack_needed, True)
            self.assertEqual(client._sack_duplicates, [])
            self.assertEqual(client._sack_misordered, set([3]))
            self.assertEqual(client._last_received_tsn, 1)
            client._sack_needed = False

            # receive middle chunk
            await client._receive_chunk(chunks[1])
            self.assertEqual(client._sack_needed, True)
            self.assertEqual(client._sack_duplicates, [])
            self.assertEqual(client._sack_misordered, set([]))
            self.assertEqual(client._last_received_tsn, 3)
            client._sack_needed = False

            # receive last chunk again
            await client._receive_chunk(chunks[2])
            self.assertEqual(client._sack_needed, True)
            self.assertEqual(client._sack_duplicates, [3])
            self.assertEqual(client._sack_misordered, set([]))
            self.assertEqual(client._last_received_tsn, 3)
            client._sack_needed = False

    @asynctest
    async def test_receive_forward_tsn(self) -> None:
        received = []

        async def mock_receive(stream_id: int, pp_id: int, data: bytes) -> None:
            received.append((stream_id, pp_id, data))

        async with client_standalone() as client:
            client._last_received_tsn = 101
            client._receive = mock_receive  # type: ignore

            factory = ChunkFactory(tsn=102)
            chunks = (
                factory.create([b"foo"])
                + factory.create([b"baz"])
                + factory.create([b"qux"])
                + factory.create([b"quux"])
                + factory.create([b"corge"])
                + factory.create([b"grault"])
            )

            # receive chunks with gaps
            for i in [0, 2, 3, 5]:
                await client._receive_chunk(chunks[i])

            self.assertEqual(client._sack_needed, True)
            self.assertEqual(client._sack_duplicates, [])
            self.assertEqual(client._sack_misordered, set([104, 105, 107]))
            self.assertEqual(client._last_received_tsn, 102)
            self.assertEqual(received, [(456, 123, b"foo")])
            received.clear()
            client._sack_needed = False

            # receive forward tsn
            chunk = ForwardTsnChunk()
            chunk.cumulative_tsn = 103
            chunk.streams = [(456, 1)]
            await client._receive_chunk(chunk)
            self.assertEqual(client._sack_needed, True)
            self.assertEqual(client._sack_duplicates, [])
            self.assertEqual(client._sack_misordered, set([107]))
            self.assertEqual(client._last_received_tsn, 105)
            self.assertEqual(received, [(456, 123, b"qux"), (456, 123, b"quux")])
            received.clear()
            client._sack_needed = False

            # receive forward tsn again
            await client._receive_chunk(chunk)
            self.assertEqual(client._sack_needed, True)
            self.assertEqual(client._sack_duplicates, [])
            self.assertEqual(client._sack_misordered, set([107]))
            self.assertEqual(client._last_received_tsn, 105)
            self.assertEqual(received, [])
            client._sack_needed = False

            # receive chunk
            await client._receive_chunk(chunks[4])

            self.assertEqual(client._sack_needed, True)
            self.assertEqual(client._sack_duplicates, [])
            self.assertEqual(client._sack_misordered, set())
            self.assertEqual(client._last_received_tsn, 107)
            self.assertEqual(received, [(456, 123, b"corge"), (456, 123, b"grault")])
            received.clear()
            client._sack_needed = False

            chunk = ForwardTsnChunk()
            chunk.cumulative_tsn = 108
            await client._receive_chunk(chunk)
            self.assertEqual(client._last_received_tsn, 108)
            self.assertEqual(client._sack_misordered, set())
            self.assertEqual(received, [])

    @asynctest
    async def test_receive_heartbeat(self) -> None:
        ack = None

        async def mock_send_chunk(chunk: Chunk) -> None:
            nonlocal ack
            assert isinstance(chunk, HeartbeatAckChunk)
            ack = chunk

        async with client_standalone() as client:
            client._send_chunk = mock_send_chunk  # type: ignore

            # receive heartbeat
            chunk = HeartbeatChunk()
            chunk.params.append((1, b"\x01\x02\x03\x04"))
            await client._receive_chunk(chunk)

            # check response
            self.assertIsInstance(ack, HeartbeatAckChunk)
            self.assertEqual(ack.params, [(1, b"\x01\x02\x03\x04")])

    @asynctest
    async def test_receive_sack_discard(self) -> None:
        async with client_standalone() as client:
            client._last_received_tsn = 0

            # receive sack
            sack_point = client._last_sacked_tsn
            chunk = SackChunk()
            chunk.cumulative_tsn = tsn_minus_one(sack_point)
            await client._receive_chunk(chunk)

            # sack point must not changed
            self.assertEqual(client._last_sacked_tsn, sack_point)

    @asynctest
    async def test_receive_shutdown(self) -> None:
        async with client_standalone() as client:
            client._last_received_tsn = 0
            client._send_chunk = noop_send_chunk  # type: ignore
            client._set_state(RTCSctpTransport.State.ESTABLISHED)

            # receive shutdown
            chunk = ShutdownChunk()
            chunk.cumulative_tsn = tsn_minus_one(client._last_sacked_tsn)
            await client._receive_chunk(chunk)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.SHUTDOWN_ACK_SENT
            )

            # receive shutdown complete
            chunk2 = ShutdownCompleteChunk()
            await client._receive_chunk(chunk2)
            self.assertEqual(client._association_state, RTCSctpTransport.State.CLOSED)

    @asynctest
    async def test_mark_received(self) -> None:
        async with client_standalone() as client:
            client._last_received_tsn = 0

            # receive 1
            self.assertFalse(client._mark_received(1))
            self.assertEqual(client._last_received_tsn, 1)
            self.assertEqual(client._sack_misordered, set())

            # receive 3
            self.assertFalse(client._mark_received(3))
            self.assertEqual(client._last_received_tsn, 1)
            self.assertEqual(client._sack_misordered, set([3]))

            # receive 4
            self.assertFalse(client._mark_received(4))
            self.assertEqual(client._last_received_tsn, 1)
            self.assertEqual(client._sack_misordered, set([3, 4]))

            # receive 6
            self.assertFalse(client._mark_received(6))
            self.assertEqual(client._last_received_tsn, 1)
            self.assertEqual(client._sack_misordered, set([3, 4, 6]))

            # receive 2
            self.assertFalse(client._mark_received(2))
            self.assertEqual(client._last_received_tsn, 4)
            self.assertEqual(client._sack_misordered, set([6]))

    @asynctest
    async def test_send_sack(self) -> None:
        sack = None

        async def mock_send_chunk(chunk: Chunk) -> None:
            nonlocal sack
            assert isinstance(chunk, SackChunk)
            sack = chunk

        async with client_standalone() as client:
            client._last_received_tsn = 123
            client._send_chunk = mock_send_chunk  # type: ignore

            await client._send_sack()
            self.assertIsNotNone(sack)
            self.assertEqual(sack.duplicates, [])
            self.assertEqual(sack.gaps, [])
            self.assertEqual(sack.cumulative_tsn, 123)

    @asynctest
    async def test_send_sack_with_duplicates(self) -> None:
        sack = None

        async def mock_send_chunk(chunk: Chunk) -> None:
            nonlocal sack
            assert isinstance(chunk, SackChunk)
            sack = chunk

        async with client_standalone() as client:
            client._last_received_tsn = 123
            client._sack_duplicates = [125, 127]
            client._send_chunk = mock_send_chunk  # type: ignore

            await client._send_sack()
            self.assertIsNotNone(sack)
            self.assertEqual(sack.duplicates, [125, 127])
            self.assertEqual(sack.gaps, [])
            self.assertEqual(sack.cumulative_tsn, 123)

    @asynctest
    async def test_send_sack_with_gaps(self) -> None:
        sack = None

        async def mock_send_chunk(chunk: SackChunk) -> None:
            nonlocal sack
            sack = chunk

        async with client_standalone() as client:
            client._last_received_tsn = 12
            client._sack_misordered = {14, 15, 17}
            client._send_chunk = mock_send_chunk  # type: ignore

            await client._send_sack()
            self.assertIsNotNone(sack)
            self.assertEqual(sack.duplicates, [])
            self.assertEqual(sack.gaps, [(2, 3), (5, 5)])
            self.assertEqual(sack.cumulative_tsn, 12)

    @asynctest
    async def test_send_data(self) -> None:
        async with client_standalone() as client:
            client._local_tsn = 0
            client._send_chunk = noop_send_chunk  # type: ignore

            # no data
            await client._transmit()
            self.assertIsNone(client._t3_handle)
            self.assertEqual(outstanding_tsns(client), [])
            self.assertEqual(queued_tsns(client), [])
            self.assertEqual(client._outbound_stream_seq, {})

            # 1 chunk
            await client._send(123, 456, b"M" * USERDATA_MAX_LENGTH)
            self.assertIsNotNone(client._t3_handle)
            self.assertEqual(outstanding_tsns(client), [0])
            self.assertEqual(queued_tsns(client), [])
            self.assertEqual(client._outbound_stream_seq, {123: 1})

    @asynctest
    async def test_send_data_unordered(self) -> None:
        async with client_standalone() as client:
            client._local_tsn = 0
            client._send_chunk = noop_send_chunk  # type: ignore

            # 1 chunk
            await client._send(123, 456, b"M" * USERDATA_MAX_LENGTH, ordered=False)
            self.assertIsNotNone(client._t3_handle)
            self.assertEqual(outstanding_tsns(client), [0])
            self.assertEqual(queued_tsns(client), [])
            self.assertEqual(client._outbound_stream_seq, {})

    @asynctest
    async def test_send_data_congestion_control(self) -> None:
        sent_tsns = []

        async def mock_send_chunk(chunk: Chunk) -> None:
            assert isinstance(chunk, DataChunk)
            sent_tsns.append(chunk.tsn)

        async with client_standalone() as client:
            client._cwnd = 4 * USERDATA_MAX_LENGTH
            client._last_sacked_tsn = 4294967295
            client._advanced_peer_ack_tsn = 4294967295
            client._local_tsn = 0
            client._ssthresh = 4 * USERDATA_MAX_LENGTH
            client._send_chunk = mock_send_chunk  # type: ignore

            # queue 16 chunks, but cwnd only allows 4
            await client._send(123, 456, b"M" * USERDATA_MAX_LENGTH * 16)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3])
            self.assertEqual(outstanding_tsns(client), [0, 1, 2, 3])
            self.assertEqual(
                queued_tsns(client), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            )

            # SACK comes in acknowledging 2 chunks
            sack = SackChunk()
            sack.cumulative_tsn = 1
            await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6])
            self.assertEqual(outstanding_tsns(client), [2, 3, 4, 5, 6])
            self.assertEqual(queued_tsns(client), [7, 8, 9, 10, 11, 12, 13, 14, 15])

            # SACK comes in acknowledging 2 more chunks
            sack = SackChunk()
            sack.cumulative_tsn = 3
            await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7, 8])
            self.assertEqual(outstanding_tsns(client), [4, 5, 6, 7, 8])
            self.assertEqual(queued_tsns(client), [9, 10, 11, 12, 13, 14, 15])

            # SACK comes in acknowledging 2 more chunks
            sack = SackChunk()
            sack.cumulative_tsn = 5
            await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            self.assertEqual(outstanding_tsns(client), [6, 7, 8, 9, 10])
            self.assertEqual(queued_tsns(client), [11, 12, 13, 14, 15])

            # SACK comes in acknowledging 2 more chunks
            sack = SackChunk()
            sack.cumulative_tsn = 7
            await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 6 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 6 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
            self.assertEqual(outstanding_tsns(client), [8, 9, 10, 11, 12, 13])
            self.assertEqual(queued_tsns(client), [14, 15])

            # SACK comes in acknowledging 2 more chunks
            sack = SackChunk()
            sack.cumulative_tsn = 9
            await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 6 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 6 * USERDATA_MAX_LENGTH)
            self.assertEqual(
                sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            )
            self.assertEqual(outstanding_tsns(client), [10, 11, 12, 13, 14, 15])
            self.assertEqual(queued_tsns(client), [])

    @asynctest
    async def test_send_data_slow_start(self) -> None:
        sent_tsns = []

        async def mock_send_chunk(chunk: Chunk) -> None:
            assert isinstance(chunk, DataChunk)
            sent_tsns.append(chunk.tsn)

        async with client_standalone() as client:
            client._last_sacked_tsn = 4294967295
            client._advanced_peer_ack_tsn = 4294967295
            client._local_tsn = 0
            client._ssthresh = 131072
            client._send_chunk = mock_send_chunk  # type: ignore

            # queue 8 chunks, but cwnd only allows 3
            with self.assertTimerRestarted(client):
                await client._send(123, 456, b"M" * USERDATA_MAX_LENGTH * 8)

            self.assertEqual(client._cwnd, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2])
            self.assertEqual(outstanding_tsns(client), [0, 1, 2])
            self.assertEqual(queued_tsns(client), [3, 4, 5, 6, 7])

            # SACK comes in acknowledging 2 chunks
            sack = SackChunk()
            sack.cumulative_tsn = 1
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5])
            self.assertEqual(outstanding_tsns(client), [2, 3, 4, 5])
            self.assertEqual(queued_tsns(client), [6, 7])

            # SACK sack comes in acknowledging 2 more chunks
            sack = SackChunk()
            sack.cumulative_tsn = 3
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(outstanding_tsns(client), [4, 5, 6, 7])
            self.assertEqual(queued_tsns(client), [])

            # SACK comes in acknowledging 2 more chunks
            sack = SackChunk()
            sack.cumulative_tsn = 5
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 2 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(outstanding_tsns(client), [6, 7])
            self.assertEqual(queued_tsns(client), [])

            # SACK comes in acknowledging final chunks
            sack = SackChunk()
            sack.cumulative_tsn = 7
            with self.assertTimerStopped(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 0)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(outstanding_tsns(client), [])
            self.assertEqual(queued_tsns(client), [])

    @asynctest
    async def test_send_data_with_gap(self) -> None:
        sent_tsns = []

        async def mock_send_chunk(chunk: Chunk) -> None:
            assert isinstance(chunk, DataChunk)
            sent_tsns.append(chunk.tsn)

        async with client_standalone() as client:
            client._last_sacked_tsn = 4294967295
            client._advanced_peer_ack_tsn = 4294967295
            client._local_tsn = 0
            client._ssthresh = 131072
            client._send_chunk = mock_send_chunk  # type: ignore

            # queue 8 chunks, but cwnd only allows 3
            with self.assertTimerRestarted(client):
                await client._send(123, 456, b"M" * USERDATA_MAX_LENGTH * 8)

            self.assertEqual(client._cwnd, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2])
            self.assertEqual(outstanding_tsns(client), [0, 1, 2])
            self.assertEqual(queued_tsns(client), [3, 4, 5, 6, 7])

            # SACK comes in acknowledging chunks 0 and 2
            sack = SackChunk()
            sack.cumulative_tsn = 0
            sack.gaps = [(2, 2)]  # TSN 1 is missing
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5])
            self.assertEqual(outstanding_tsns(client), [1, 2, 3, 4, 5])
            self.assertEqual(queued_tsns(client), [6, 7])

            # SACK comes in acknowledging chunks 1 and 3
            sack = SackChunk()
            sack.cumulative_tsn = 3
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(outstanding_tsns(client), [4, 5, 6, 7])
            self.assertEqual(queued_tsns(client), [])

            # SACK comes in acknowledging 2 more chunks
            sack = SackChunk()
            sack.cumulative_tsn = 5
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 2 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(outstanding_tsns(client), [6, 7])
            self.assertEqual(queued_tsns(client), [])

            # SACK comes in acknowledging final chunks
            sack = SackChunk()
            sack.cumulative_tsn = 7
            with self.assertTimerStopped(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 5 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 0)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(outstanding_tsns(client), [])
            self.assertEqual(queued_tsns(client), [])

    @asynctest
    async def test_send_data_with_gap_1_retransmit(self) -> None:
        sent_tsns = []

        async def mock_send_chunk(chunk: Chunk) -> None:
            assert isinstance(chunk, DataChunk)
            sent_tsns.append(chunk.tsn)

        async with client_standalone() as client:
            client._last_sacked_tsn = 4294967295
            client._advanced_peer_ack_tsn = 4294967295
            client._local_tsn = 0
            client._ssthresh = 131072
            client._send_chunk = mock_send_chunk  # type: ignore

            # queue 8 chunks, but cwnd only allows 3
            with self.assertTimerRestarted(client):
                await client._send(123, 456, b"M" * USERDATA_MAX_LENGTH * 8)

            self.assertEqual(client._cwnd, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2])
            self.assertEqual(outstanding_tsns(client), [0, 1, 2])
            self.assertEqual(queued_tsns(client), [3, 4, 5, 6, 7])

            # SACK comes in acknowledging chunks 0 and 2
            sack = SackChunk()
            sack.cumulative_tsn = 0
            sack.gaps = [(2, 2)]  # TSN 1 is missing
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5])
            self.assertEqual(outstanding_tsns(client), [1, 2, 3, 4, 5])
            self.assertEqual(queued_tsns(client), [6, 7])

            # SACK comes in acknowledging chunks 3 and 4
            sack = SackChunk()
            sack.cumulative_tsn = 0
            sack.gaps = [(2, 4)]  # TSN 1 is missing
            with self.assertTimerPreserved(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(outstanding_tsns(client), [1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(queued_tsns(client), [])

            # SACK comes in acknowledging 2 more chunks
            sack = SackChunk()
            sack.cumulative_tsn = 0
            sack.gaps = [(2, 6)]  # TSN 1 is missing
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, 7)
            self.assertEqual(client._flight_size, 2 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7, 1])
            self.assertEqual(outstanding_tsns(client), [1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(queued_tsns(client), [])

            # SACK comes in acknowledging final chunks
            sack = SackChunk()
            sack.cumulative_tsn = 7
            with self.assertTimerStopped(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 0)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7, 1])
            self.assertEqual(outstanding_tsns(client), [])
            self.assertEqual(queued_tsns(client), [])

    @asynctest
    async def test_send_data_with_gap_2_retransmit(self) -> None:
        sent_tsns = []

        async def mock_send_chunk(chunk: Chunk) -> None:
            assert isinstance(chunk, DataChunk)
            sent_tsns.append(chunk.tsn)

        async with client_standalone() as client:
            client._last_sacked_tsn = 4294967295
            client._advanced_peer_ack_tsn = 4294967295
            client._local_tsn = 0
            client._ssthresh = 131072
            client._send_chunk = mock_send_chunk  # type: ignore

            # queue 8 chunks, but cwnd only allows 3
            with self.assertTimerRestarted(client):
                await client._send(123, 456, b"M" * USERDATA_MAX_LENGTH * 8)

            self.assertEqual(client._cwnd, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2])
            self.assertEqual(outstanding_tsns(client), [0, 1, 2])
            self.assertEqual(queued_tsns(client), [3, 4, 5, 6, 7])

            # SACK comes in acknowledging chunk 2
            sack = SackChunk()
            sack.cumulative_tsn = 4294967295
            sack.gaps = [(3, 3)]  # TSN 0 and 1 are missing
            with self.assertTimerPreserved(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3])
            self.assertEqual(outstanding_tsns(client), [0, 1, 2, 3])
            self.assertEqual(queued_tsns(client), [4, 5, 6, 7])

            # SACK comes in acknowledging chunk 3
            sack = SackChunk()
            sack.cumulative_tsn = 4294967295
            sack.gaps = [(3, 4)]  # TSN 0 and 1 are missing
            with self.assertTimerPreserved(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4])
            self.assertEqual(outstanding_tsns(client), [0, 1, 2, 3, 4])
            self.assertEqual(queued_tsns(client), [5, 6, 7])

            # SACK comes in acknowledging chunk 4
            sack = SackChunk()
            sack.cumulative_tsn = 4294967295
            sack.gaps = [(3, 5)]  # TSN 0 and 1 are missing
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, 4)
            self.assertEqual(client._flight_size, 2 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 0, 1])
            self.assertEqual(outstanding_tsns(client), [0, 1, 2, 3, 4])
            self.assertEqual(queued_tsns(client), [5, 6, 7])

            # SACK comes in acknowledging all chunks up to 4
            sack = SackChunk()
            sack.cumulative_tsn = 4
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 0, 1, 5, 6, 7])
            self.assertEqual(outstanding_tsns(client), [5, 6, 7])
            self.assertEqual(queued_tsns(client), [])

            # SACK comes in acknowledging final chunks
            sack = SackChunk()
            sack.cumulative_tsn = 7
            with self.assertTimerStopped(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 0)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 0, 1, 5, 6, 7])
            self.assertEqual(outstanding_tsns(client), [])
            self.assertEqual(queued_tsns(client), [])

    @asynctest
    async def test_send_data_with_gap_3_retransmit(self) -> None:
        sent_tsns = []

        async def mock_send_chunk(chunk: Chunk) -> None:
            assert isinstance(chunk, DataChunk)
            sent_tsns.append(chunk.tsn)

        async with client_standalone() as client:
            client._last_sacked_tsn = 4294967295
            client._advanced_peer_ack_tsn = 4294967295
            client._local_tsn = 0
            client._ssthresh = 131072
            client._send_chunk = mock_send_chunk  # type: ignore

            # queue 8 chunks, but cwnd only allows 3
            with self.assertTimerRestarted(client):
                await client._send(123, 456, b"M" * USERDATA_MAX_LENGTH * 8)

            self.assertEqual(client._cwnd, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2])
            self.assertEqual(outstanding_tsns(client), [0, 1, 2])
            self.assertEqual(queued_tsns(client), [3, 4, 5, 6, 7])

            # SACK comes in acknowledging chunks 0 and 1
            sack = SackChunk()
            sack.cumulative_tsn = 1
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5])
            self.assertEqual(outstanding_tsns(client), [2, 3, 4, 5])
            self.assertEqual(queued_tsns(client), [6, 7])

            # SACK comes in acknowledging chunk 5
            sack = SackChunk()
            sack.cumulative_tsn = 1
            sack.gaps = [(4, 4)]  # TSN 2, 3 and 4 are missing
            with self.assertTimerPreserved(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6])
            self.assertEqual(outstanding_tsns(client), [2, 3, 4, 5, 6])
            self.assertEqual(queued_tsns(client), [7])

            # SACK comes in acknowledging chunk 6
            sack = SackChunk()
            sack.cumulative_tsn = 1
            sack.gaps = [(4, 5)]  # TSN 2, 3 and 4 are missing
            with self.assertTimerPreserved(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(outstanding_tsns(client), [2, 3, 4, 5, 6, 7])
            self.assertEqual(queued_tsns(client), [])

            # artificially raise flight size to hit cwnd
            client._flight_size += 2 * USERDATA_MAX_LENGTH

            # SACK comes in acknowledging chunk 7
            sack = SackChunk()
            sack.cumulative_tsn = 1
            sack.gaps = [(4, 6)]  # TSN 2, 3 and 4 are missing
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, 7)
            self.assertEqual(client._flight_size, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7, 2, 3])
            self.assertEqual(outstanding_tsns(client), [2, 3, 4, 5, 6, 7])
            self.assertEqual(queued_tsns(client), [])

            # SACK comes in acknowledging all chunks up to 3, and 5, 6, 7
            sack = SackChunk()
            sack.cumulative_tsn = 3
            sack.gaps = [(2, 4)]  # TSN 4 is missing
            with self.assertTimerRestarted(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, 7)
            self.assertEqual(client._flight_size, 3 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4])
            self.assertEqual(outstanding_tsns(client), [4, 5, 6, 7])
            self.assertEqual(queued_tsns(client), [])

            # SACK comes in ackowledging all chunks
            sack = SackChunk()
            sack.cumulative_tsn = 7
            with self.assertTimerStopped(client):
                await client._receive_chunk(sack)

            self.assertEqual(client._cwnd, 4 * USERDATA_MAX_LENGTH)
            self.assertEqual(client._fast_recovery_exit, None)
            self.assertEqual(client._flight_size, 2 * USERDATA_MAX_LENGTH)
            self.assertEqual(sent_tsns, [0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4])
            self.assertEqual(outstanding_tsns(client), [])
            self.assertEqual(queued_tsns(client), [])

    @asynctest
    async def test_t2_expired_when_shutdown_ack_sent(self) -> None:
        async with client_standalone() as client:
            client._send_chunk = noop_send_chunk  # type: ignore

            chunk = ShutdownAckChunk()

            # fails once
            client._set_state(RTCSctpTransport.State.SHUTDOWN_ACK_SENT)
            client._t2_start(chunk)
            client._t2_expired()
            self.assertEqual(client._t2_failures, 1)
            self.assertIsNotNone(client._t2_handle)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.SHUTDOWN_ACK_SENT
            )

            # fails 10 times
            client._t2_failures = 9
            client._t2_expired()
            self.assertEqual(client._t2_failures, 10)
            self.assertIsNotNone(client._t2_handle)
            self.assertEqual(
                client._association_state, RTCSctpTransport.State.SHUTDOWN_ACK_SENT
            )

            # fails 11 times
            client._t2_expired()
            self.assertEqual(client._t2_failures, 11)
            self.assertIsNone(client._t2_handle)
            self.assertEqual(client._association_state, RTCSctpTransport.State.CLOSED)

            # let async code complete
            await asyncio.sleep(0)

    @asynctest
    async def test_t3_expired(self) -> None:
        async def noop_transmit() -> None:
            pass

        async with client_standalone() as client:
            client._local_tsn = 0
            client._send_chunk = noop_send_chunk  # type: ignore

            # 1 chunk
            await client._send(123, 456, b"M" * USERDATA_MAX_LENGTH)
            self.assertIsNotNone(client._t3_handle)
            self.assertEqual(outstanding_tsns(client), [0])
            self.assertEqual(queued_tsns(client), [])

            # t3 expires
            client._transmit = noop_transmit  # type: ignore
            client._t3_expired()
            self.assertIsNone(client._t3_handle)
            self.assertEqual(outstanding_tsns(client), [0])
            self.assertEqual(queued_tsns(client), [])
            for chunk in client._outbound_queue:
                self.assertEqual(chunk._retransmit, True)

            # let async code complete
            await asyncio.sleep(0)
