"""Render the same terminal cells from real timestamped runs, at original speed."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from bisect import bisect_right
from pathlib import Path

from laya_mlx.snake.replay import TerminalRaster, load_record

from .ui import compose

__all__ = ["TerminalRaster", "load_record", "main"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True, help=".mp4 or .png")
    parser.add_argument("--start", type=float, default=0, help="Source recording time in seconds")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate; playback remains 1×")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--font")
    parser.add_argument("--gif", type=Path, help="Also create a shareable GIF from the MP4")
    parser.add_argument(
        "--gif-seconds",
        type=float,
        default=15,
        help="GIF excerpt length at original speed (default: 15)",
    )
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.start)
        or not math.isfinite(args.seconds)
        or not math.isfinite(args.gif_seconds)
        or args.start < 0
        or args.seconds <= 0
        or args.gif_seconds <= 0
        or args.fps < 1
        or args.width < 640
        or args.height < 480
        or args.width % 2
        or args.height % 2
    ):
        parser.error(
            "Use finite nonnegative start, positive seconds/FPS and even dimensions >= 640×480"
        )
    if args.output.exists():
        parser.error("Output already exists; choose a new filename")
    if args.output.suffix not in (".mp4", ".png"):
        parser.error("--output must end in .mp4 or .png")
    if args.gif and args.gif.exists():
        parser.error("GIF output already exists; choose a new filename")
    metadata, frames = load_record(args.recording)
    times = [f["at"] for f in frames]
    start = max(times[0], args.start)
    end = min(start + args.seconds, times[-1])
    if start > times[-1] or (args.output.suffix == ".mp4" and end <= start):
        parser.error("Requested interval is outside the recording")

    def canvas_at(t):
        entry = frames[max(0, bisect_right(times, t) - 1)]
        return compose(entry["game"], entry["decision"], {**entry["stats"], "replay": True})

    canvas = canvas_at(start)
    raster = TerminalRaster(
        canvas.width,
        canvas.height,
        width=args.width,
        height=args.height,
        font=args.font,
        title="laya-frogger  /  real recorded decisions",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".png":
        raster.render(canvas).save(args.output)
        print(args.output)
        return 0
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("MP4 export needs ffmpeg (on macOS: brew install ffmpeg)")
    count = max(1, int((end - start) * args.fps))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{args.width}x{args.height}",
        "-framerate",
        str(args.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        last_index, pixels = None, None
        for index in range(count):
            t = start + index / args.fps
            source_index = max(0, bisect_right(times, t) - 1)
            if source_index != last_index:
                pixels = raster.render(canvas_at(t)).tobytes()
                last_index = source_index
            process.stdin.write(pixels)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg did not complete the MP4")
    except BaseException:
        process.kill()
        process.wait()
        raise
    if args.gif:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-i",
                str(args.output),
                "-t",
                str(min(args.gif_seconds, count / args.fps)),
                "-filter_complex",
                "fps=10,scale=1040:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=bayer",
                "-loop",
                "0",
                str(args.gif),
            ],
            check=True,
        )
    sidecar = {
        "source_recording": args.recording.name,
        "source_sha256": hashlib.sha256(args.recording.read_bytes()).hexdigest(),
        "model": metadata["model"],
        "playback_speed": 1,
        "source_start_seconds": start,
        "source_end_seconds": start + count / args.fps,
        "video_fps": args.fps,
        "video_frames": count,
        "gif_seconds": min(args.gif_seconds, count / args.fps) if args.gif else None,
        "renderer_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("ui.py").read_bytes()
        ).hexdigest(),
        "note": "Rendered from real inference records using the live terminal layout. Original wall-clock timing; no fabricated probabilities or time compression. Video is sampled at its stated frame rate.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps({"video": str(args.output), "seconds": count / args.fps, "playback_speed": 1}))
    return 0
