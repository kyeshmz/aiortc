Use H.265 / HEVC
================

This fork adds HEVC video to aiortc. You can encode raw video frames with
``libx265``, send compressed HEVC packets without re-encoding them, and receive
decoded frames through the existing track API. VP8 and H.264 retain their
existing default preference order. HEVC follows them.

Install the fork
----------------

Install this branch in a virtual environment. The upstream PyPI package does
not contain these changes.

.. code-block:: console

   python -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install 'git+https://github.com/kyeshmz/aiortc.git@feat/h265'

The project requires Python 3.11 or later and PyAV 14 through 19. Your PyAV build
must include the ``libx265`` encoder and the ``hevc`` decoder. The FFmpeg
executable on your system is separate from the FFmpeg libraries that PyAV uses.
Check the codecs available to Python:

.. code-block:: console

   python -c 'import av; print(av.Codec("libx265", "w")); print(av.Codec("hevc", "r"))'

If a codec is unavailable, install a PyAV build with the required codec.
Review the licenses and patent requirements that apply to your distribution
and use of HEVC and x265.

Select HEVC before negotiation
------------------------------

Add a video track, then set the transceiver's codec preferences before you
create an offer or answer:

.. code-block:: python

   from aiortc import RTCPeerConnection, RTCRtpSender, VideoStreamTrack

   pc = RTCPeerConnection()
   sender = pc.addTrack(VideoStreamTrack())
   transceiver = next(t for t in pc.getTransceivers() if t.sender is sender)
   codecs = RTCRtpSender.getCapabilities("video").codecs
   transceiver.setCodecPreferences(
       [c for c in codecs if c.mimeType in ("video/H265", "video/rtx")]
   )

This selection requires HEVC. Negotiation fails if the other endpoint has no
compatible HEVC codec. To allow fallback, append the VP8 and H.264 capabilities
after the HEVC and RTX capabilities instead of excluding them. Keep the existing
SDP signaling and ICE candidate exchange in your application.

For a receiving-only connection, call ``pc.addTransceiver("video",
direction="recvonly")`` and set its codec preferences before you create an offer.

Send compressed video
---------------------

Use ``MediaPlayer`` with ``decode=False`` to send an existing HEVC stream:

.. code-block:: python

   from aiortc.contrib.media import MediaPlayer

   player = MediaPlayer("input.mp4", decode=False)
   if player.video is None:
       raise ValueError("The file has no supported video stream")
   sender = pc.addTrack(player.video)

Apply the HEVC codec preferences to this sender's transceiver as shown above.
The HEVC packetizer converts MP4 length-prefixed packets to Annex B with
FFmpeg's ``hevc_mp4toannexb`` bitstream filter. MPEG-TS Annex-B packets also
work. A bare ``av.Packet`` without a stream must already contain an Annex-B
access unit and have a presentation timestamp and time base.

Compressed input must already match the negotiated Main profile, main tier,
and level. Use progressive 8-bit 4:2:0 video without B-frames. Provide the
parameter sets before each random-access picture, with the VPS, SPS, and PPS
before any prefix SEI. Packet passthrough does not transcode incompatible
profiles, resize video, or lower its bitrate. The external encoder must handle
keyframe requests. Stop ``player.video`` when you finish using the player.

Supported format and limits
---------------------------

The RTP implementation supports RFC 7798 single-NAL packets, aggregation
packets, and fragmentation units. Payloads are at most 1300 bytes. RTX uses the
existing aiortc retransmission path.

The advertised format has these parameters:

.. code-block:: text

   profile-id=1;tier-flag=0;level-id=93;tx-mode=SRST;sprop-max-don-diff=0

An absent profile, tier, level, or transmission mode uses the RFC 7798 default.
The answer selects the lower level advertised by the peers. The encoder enforces
that level's dimensions, sample rate at 30 fps, and bitrate ceiling. Main profile
level 3.1 supports 1280 by 720 at 30 fps. Supply even frame dimensions of at
least 16 by 16. The encoder converts input frames to 8-bit ``yuv420p``.

Raw-frame encoding uses a target bitrate of 1 Mbps, clamped to 500 kbps through
3 Mbps. A lower negotiated level can impose a lower ceiling. A bitrate change
greater than 10 percent or a resolution change restarts the encoder and emits a
new keyframe with parameter sets. Explicit keyframe requests produce IDR
pictures. The encoder disables B-frames and worker pools and uses one frame
thread. The decoder also uses one thread.

This implementation intentionally rejects formats outside its supported subset:

* Main 10, other profiles, nonzero profile spaces, and high tier.
* Nondefault profile compatibility or interoperability constraint parameters.
  The supported explicit values are ``40000000`` and ``b00000000000``.
* MRST, MRMT, positive decoding-order-number differences, and multilayer NALs.
* Out-of-band SDP parameter sets, sub-layer selection, and asymmetric level
  parameters.
* Other unimplemented fmtp extensions and PACI RTP packets.

Parameter sets travel in-band. The implementation does not claim complete
support for every optional RFC 7798 format or the evolving WebRTC HEVC profile.
Browser codec availability depends on the browser, platform, hardware, and
policy. aiortc-to-aiortc tests do not establish browser or hardware interoperability.

Run the HEVC tests
------------------

From a source checkout with the development dependencies installed, run:

.. code-block:: console

   python -m pip install -e '.[dev]'
   python -m unittest tests.test_h265 tests.test_h265_peerconnection -v

These tests cover malformed RTP payloads, payload size limits, Annex-B
reconstruction, codec roundtrips, keyframes, bitrate and resolution changes,
SDP rejection and level negotiation, and decoded video over H.265-only peer
connections. They also send compressed MP4 and MPEG-TS video through
``MediaPlayer(decode=False)`` without invoking the raw-frame encoder.

Protocol references
-------------------

* `RFC 7798 <https://www.rfc-editor.org/rfc/rfc7798.html>`_ defines HEVC RTP
  packets and SDP parameters.
* `H.265 Profile for WebRTC, draft version 10
  <https://www.ietf.org/archive/id/draft-ietf-avtcore-hevc-webrtc-10.txt>`_ is
  work in progress, dated October 20, 2026.
