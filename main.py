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
async def handle_barge_in(decoded, twilio_ws, streamsid):
    if decoded.get("type") == "UserStartedSpeaking":
        await twilio_ws.send(json.dumps({"event": "clear", "streamSid": streamsid}))

async def handle_text_message(decoded, twilio_ws, sts_ws, streamsid):
    # Add more message types here if you want (assistant responses, logs, etc.)
    await handle_barge_in(decoded, twilio_ws, streamsid)

# ---- Tasks ------------------------------------------------------------------
async def sts_sender(sts_ws, audio_queue):
    print("sts_sender started")
    while True:
        chunk = await audio_queue.get()
        try:
            await sts_ws.send(chunk)  # raw mulaw bytes to Deepgram
        except Exception as e:
            print("sts_sender error:", e)
            break

async def sts_receiver(sts_ws, twilio_ws, streamsid_queue):
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

