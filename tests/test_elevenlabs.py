import asyncio

from app.providers.elevenlabs.provider import ElevenLabsTTS


async def main():
    print("1. Starting ElevenLabs test...", flush=True)

    tts = ElevenLabsTTS()

    print("2. Provider initialized.", flush=True)

    try:
        print("3. Sending request to ElevenLabs...", flush=True)

        result = await tts.synthesize(
            "Hello, I am Weings. This is my ElevenLabs voice test.",
            language="en",
        )

        print("4. Response received.", flush=True)
        print("Audio bytes:", len(result.data), flush=True)

        with open("elevenlabs_test.mp3", "wb") as file:
            file.write(result.data)

        print("5. Audio saved: elevenlabs_test.mp3", flush=True)

    finally:
        print("6. Closing provider...", flush=True)
        await tts.close()
        print("7. Done.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())