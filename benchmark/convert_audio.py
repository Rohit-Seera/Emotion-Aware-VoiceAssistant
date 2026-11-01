from pathlib import Path
import subprocess
import imageio_ffmpeg

src = Path("benchmark/results_generation/audio")
dst = Path("benchmark/results_generation/audio_wav")
dst.mkdir(parents=True, exist_ok=True)

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

files = sorted(src.glob("*.mp3"))

for mp3 in files:
    wav = dst / f"{mp3.stem}.wav"

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(mp3),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav),
    ]

    subprocess.run(cmd, check=True)

    print(f"Converted: {mp3.name} -> {wav.name}")

print(f"\nConverted {len(files)} files.")