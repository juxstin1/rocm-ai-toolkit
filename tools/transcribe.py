#!/usr/bin/env python3
"""
Transcribe audio/video files using Faster-Whisper on GPU.

Usage:
    transcribe audio.mp3
    transcribe video.mp4 --output transcript.txt
    transcribe meeting.wav --model large-v3
"""
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video to text")
    parser.add_argument("input", help="Input audio/video file")
    parser.add_argument("-o", "--output", help="Output text file (default: prints to console)")
    parser.add_argument("-m", "--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--language", default=None, help="Language code (auto-detect if not set)")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    print(f"Loading Whisper {args.model} model...")
    from faster_whisper import WhisperModel

    # Use GPU with float16
    model = WhisperModel(args.model, device="cuda", compute_type="float16")

    print(f"Transcribing: {args.input}")
    segments, info = model.transcribe(args.input, language=args.language)

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    lines = []
    for segment in segments:
        if args.timestamps:
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}"
        else:
            line = segment.text.strip()
        lines.append(line)
        print(line)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main()
