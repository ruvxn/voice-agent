"""
Twilio <Stream>  <->  Your WebSocket server  <->  Deepgram Agent WebSocket

Goal
----
Bridge live caller audio from Twilio to Deepgram (STT/agent) and pipe the
agent's *audio responses* back to Twilio in real time. We also listen for
Deepgram's *text events* (JSON) to do control actions like barge-in.

Architecture (high level)
-------------------------
Twilio <Stream> opens a WS to *this* server (ws://localhost:5000).
This server, per incoming Twilio connection, opens a *second* WS to
wss://agent.deepgram.com/v1/agent/converse and then shuttles data:

- Caller -> Twilio "media" frames (base64 mu-law) -> decode -> bytes.
- Buffer to ~400 ms chunks -> forward raw mu-law bytes to Deepgram.
- Deepgram sends back:
    * JSON text messages (events, transcripts, etc.) -> we inspect/act.
    * Binary audio (mu-law PCM) -> base64 -> Twilio "media" outbound.

We keep track of Twilio's streamSid so replies are routed to the right call.
"""

import asyncio
import base64
import json
import os
import ssl
import certifi
import websockets
from dotenv import load_dotenv

load_dotenv()

# ---- Deepgram Agent WS ------------------------------------------------------
def sts_connect():
    """
    Create a TLS-verified WebSocket *client* connection to Deepgram Agent.

    Why certifi?
      - Ensures we validate Deepgram's TLS certificate even inside some venvs
        that don't have a proper CA bundle.

    Why disable ping/pong here?
      - Twilio and Deepgram both maintain their own heartbeat behavior.
        We set ping_interval/ping_timeout=None to avoid *our* library
        interfering or closing otherwise-healthy long-lived streams.
    """

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise Exception("DEEPGRAM_API_KEY is not found")

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    return websockets.connect(
        "wss://agent.deepgram.com/v1/agent/converse",
        subprotocols=["token", api_key],
        ssl=ssl_ctx,
        ping_interval=None,   # Twilio/Deepgram handle their own timing; avoid spurious pings
        ping_timeout=None,
    )

def load_config():
    """
    Load the JSON agent configuration that we send as the FIRST message to Deepgram.

    - Deepgram requires a configuration JSON (e.g., audio format, agent system prompt,
      tools/functions, etc.) right after the WS opens.
    - We read config.json from the script directory for portability.
    - We validate it's non-empty and valid JSON so failures are explicit/early.
    """
    path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        raise RuntimeError(f"config.json is empty at {path}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"config.json invalid JSON at {path}: {e}")


# ---- Helpers ----------------------------------------------------------------
"""
    When Deepgram tells us the user started speaking (barge-in),
    we send Twilio a 'clear' command to stop any audio currently playing
    (e.g., interrupting the assistant's TTS playback).
    """

async def handle_barge_in(decoded, twilio_ws, streamsid):
    if decoded.get("type") == "UserStartedSpeaking":
        await twilio_ws.send(json.dumps({"event": "clear", "streamSid": streamsid}))

async def handle_text_message(decoded, twilio_ws, sts_ws, streamsid):
    """
    Handle Deepgram JSON messages (non-audio).

    Extend this with:
      - logging transcripts
      - reacting to tool-calls
      - switching agent state, etc.
    """
    # Add more message types here if you want (assistant responses, logs, etc.)
    await handle_barge_in(decoded, twilio_ws, streamsid)

# ---- Tasks ------------------------------------------------------------------
async def sts_sender(sts_ws, audio_queue):
    """
    Pump caller audio *to* Deepgram.

    Data path:
      Twilio media (inbound track, base64 mu-law) --> bytes
      -> buffered to ~400 ms chunks -> queued -> here we .send(raw bytes)

    Why raw mu-law?
      - Deepgram's Agent endpoint accepts mu-law 8k payload when configured that way.
      - That avoids transcoding on your side and reduces latency.
    """
    print("sts_sender started")
    while True:
        chunk = await audio_queue.get()
        try:
            await sts_ws.send(chunk)  # raw mulaw bytes to Deepgram
        except Exception as e:
            print("sts_sender error:", e)
            break

async def sts_receiver(sts_ws, twilio_ws, streamsid_queue):
    """
    Receive *both* JSON (text control) and binary (agent audio) from Deepgram.

    - JSON str  -> parse -> handle_text_message(...)
    - Binary    -> base64 -> send as Twilio 'media' event (outbound audio)

    We must wait for Twilio's streamSid before emitting media back to Twilio,
    so we await it from streamsid_queue at the start.
    """
    print("sts_receiver started")
    streamsid = await streamsid_queue.get()
    print("sts_receiver got streamsid:", streamsid)

    async for message in sts_ws:
        try:
            # Deepgram sends both text (JSON) and binary (audio)
            if isinstance(message, str):
                decoded = json.loads(message)
                await handle_text_message(decoded, twilio_ws, sts_ws, streamsid)
                continue

            # Binary audio from Deepgram -> Twilio <Stream>
            raw_mulaw = message if isinstance(message, (bytes, bytearray)) else bytes(message)
            media_message = {
                "event": "media",
                "streamSid": streamsid,
                "media": {"payload": base64.b64encode(raw_mulaw).decode("ascii")},
            }
            await twilio_ws.send(json.dumps(media_message))
        except Exception as e:
            print("sts_receiver error:", e)
            break

async def twilio_receiver(twilio_ws, audio_queue, streamsid_queue):
    """
    Receive Twilio <Stream> events and caller audio.

    Twilio WS messages:
      - "start": includes {"start": {"streamSid": "..."}}
      - "media": inbound audio frames (base64 mu-law, 20 ms per frame)
      - "stop": end of stream

    We:
      1) Capture streamSid and publish it so sts_receiver can reply.
      2) Accumulate inbound mu-law in a buffer and ship to Deepgram in ~400 ms
         chunks (20 frames * 20 ms * 20 = 400 ms). This strikes a balance
         between latency and throughput; tune if needed.
    """
    print("twilio_receiver started")
    BUFFER_SIZE = 20 * 160  # 20 x 20ms muLaw frames = 400ms
    inbuffer = bytearray()

    async for message in twilio_ws:
        try:
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                streamsid = data["start"]["streamSid"]
                print("twilio_receiver streamSid:", streamsid)
                streamsid_queue.put_nowait(streamsid)

            elif event == "media":
                media = data["media"]
                if media.get("track") == "inbound":
                    inbuffer.extend(base64.b64decode(media["payload"]))

            elif event == "stop":
                print("twilio_receiver stop received")
                break

            # Ship buffered caller audio to Deepgram in 400ms chunks
            while len(inbuffer) >= BUFFER_SIZE:
                chunk = inbuffer[:BUFFER_SIZE]
                audio_queue.put_nowait(chunk)
                inbuffer = inbuffer[BUFFER_SIZE:]
        except Exception as e:
            print("twilio_receiver error:", e)
            break

async def twilio_handler(twilio_ws):
    """
    Handle a single Twilio <Stream> WebSocket connection.

    For each inbound call stream:
      - Open a *fresh* connection to Deepgram.
      - Immediately send the agent config to Deepgram (required).
      - Run three tasks concurrently:
          1) sts_sender:    caller audio -> Deepgram
          2) sts_receiver:  Deepgram -> agent audio JSON/binary -> Twilio
          3) twilio_receiver: Twilio events/media -> queues
      - If any task fails (FIRST_EXCEPTION), cancel the others and close.
    """
    audio_queue = asyncio.Queue()
    streamsid_queue = asyncio.Queue()

    async with sts_connect() as sts_ws:
        # Send agent config to Deepgram
        await sts_ws.send(json.dumps(load_config()))

        tasks = [
            asyncio.create_task(sts_sender(sts_ws, audio_queue)),
            asyncio.create_task(sts_receiver(sts_ws, twilio_ws, streamsid_queue)),
            asyncio.create_task(twilio_receiver(twilio_ws, audio_queue, streamsid_queue)),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in pending:
            t.cancel()
        await twilio_ws.close()

# IMPORTANT: accept the /twilio path your TwiML points at
async def main():
    """
    Start a local WS server that Twilio <Stream> will connect to.

    Twilio console / TwiML must point <Stream url="wss://..."> to this server.
    In dev, Twilio expects a public URL (use ngrok/cloudflared to expose :5000).
    """
    async def handler_router(ws, path=None):
        # Some websockets versions pass only (ws); others pass (ws, path)
        await twilio_handler(ws)

    server = await websockets.serve(
        handler_router,
        host="localhost",
        port=5000,
        ping_interval=None,
        ping_timeout=None,
        max_size=None,  # Twilio frames are small, but avoid accidental limits
    )
    print("Server started")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())

