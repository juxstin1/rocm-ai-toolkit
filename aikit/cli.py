"""
Unified CLI entrypoint for AI Toolkit.

Usage:
    python -m aikit <command> [args]

    # Or via installed commands:
    imagine "a cyberpunk city"
    transcribe audio.mp3
    embed index ./docs
    segment photo.jpg
    rmbg photo.jpg
"""

import argparse
import sys


def imagine():
    """Image generation CLI."""
    parser = argparse.ArgumentParser(
        prog="imagine",
        description="Generate images with Stable Diffusion"
    )
    parser.add_argument("prompt", help="Text prompt for generation")
    parser.add_argument("-n", "--negative", default="blurry, bad quality, distorted",
                        help="Negative prompt")
    parser.add_argument("-m", "--model", default="stabilityai/sdxl-turbo",
                        help="Model ID")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=0.0, help="Guidance scale")
    parser.add_argument("-W", "--width", type=int, default=1024, help="Image width")
    parser.add_argument("-H", "--height", type=int, default=1024, help="Image height")
    parser.add_argument("-s", "--seed", type=int, help="Random seed")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--controlnet", choices=["canny", "depth", "pose"],
                        help="ControlNet type")
    parser.add_argument("--input", help="Input image for ControlNet")
    parser.add_argument("--cn-scale", type=float, default=0.8,
                        help="ControlNet conditioning scale")
    parser.add_argument("--lora", help="Path to LoRA weights")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale")

    args = parser.parse_args()

    if args.controlnet and not args.input:
        print("Error: --input required when using --controlnet")
        sys.exit(1)

    from aikit import ImageGen
    gen = ImageGen(model=args.model)
    gen.generate(
        prompt=args.prompt,
        negative_prompt=args.negative,
        steps=args.steps,
        cfg=args.cfg,
        width=args.width,
        height=args.height,
        seed=args.seed,
        controlnet=args.controlnet,
        control_image=args.input,
        controlnet_scale=args.cn_scale,
        lora=args.lora,
        lora_scale=args.lora_scale,
        output=args.output,
    )


def transcribe():
    """Transcription CLI."""
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe audio/video to text"
    )
    parser.add_argument("input", help="Input audio/video file")
    parser.add_argument("-o", "--output", help="Output text file")
    parser.add_argument("-m", "--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model size")
    parser.add_argument("--language", help="Language code")
    parser.add_argument("--timestamps", action="store_true",
                        help="Include timestamps")

    args = parser.parse_args()

    from aikit import Transcriber
    t = Transcriber(model=args.model)
    result = t.transcribe(args.input, language=args.language, timestamps=args.timestamps)

    if args.timestamps:
        import json
        output = json.dumps(result, indent=2)
    else:
        output = result

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved: {args.output}")
    else:
        print(output)


def embed():
    """Embedding/search CLI."""
    parser = argparse.ArgumentParser(
        prog="embed",
        description="Semantic search and embedding tool"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser("index", help="Index files in a folder")
    index_parser.add_argument("folder", help="Folder to index")
    index_parser.add_argument("--extensions", nargs="+",
                              default=[".txt", ".md", ".py", ".js", ".ts"],
                              help="File extensions to include")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed files")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--top", type=int, default=5,
                               help="Number of results")

    args = parser.parse_args()

    from aikit import Embedder
    e = Embedder()

    if args.command == "index":
        count = e.index_folder(args.folder, args.extensions)
        print(f"Indexed {count} files")
    elif args.command == "search":
        results = e.search(args.query, top_k=args.top)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['path']} (score: {r['score']:.3f})")
            print(f"    {r['preview'][:100]}...")


def segment():
    """Segmentation CLI."""
    parser = argparse.ArgumentParser(
        prog="segment",
        description="Image segmentation with SAM"
    )
    parser.add_argument("image", help="Input image")
    parser.add_argument("-o", "--output", help="Output mask image")
    parser.add_argument("--point", nargs=2, type=int, metavar=("X", "Y"),
                        help="Segment at specific point")
    parser.add_argument("--json", action="store_true",
                        help="Output segment info as JSON")

    args = parser.parse_args()

    import json
    import cv2
    import numpy as np
    from pathlib import Path
    from aikit import Segmenter

    s = Segmenter()

    if args.point:
        mask, score = s.segment_point(args.image, args.point[0], args.point[1])

        if args.output:
            cv2.imwrite(args.output, (mask * 255).astype(np.uint8))
            print(f"Saved: {args.output}")

        if args.json:
            print(json.dumps({"score": float(score)}))
        else:
            print(f"Segmented with confidence: {score:.3f}")
    else:
        masks = s.segment(args.image)

        if args.json:
            # Simplify mask data for JSON output
            simplified = [
                {
                    "area": m["area"],
                    "bbox": m["bbox"],
                    "predicted_iou": m["predicted_iou"],
                    "stability_score": m["stability_score"],
                }
                for m in masks
            ]
            print(json.dumps(simplified, indent=2))
        else:
            print(f"Found {len(masks)} segments")
            for i, m in enumerate(masks[:5]):
                print(f"  [{i}] area={m['area']}, iou={m['predicted_iou']:.3f}")
            if len(masks) > 5:
                print(f"  ... and {len(masks) - 5} more")


def rmbg():
    """Background removal CLI."""
    parser = argparse.ArgumentParser(
        prog="rmbg",
        description="Remove backgrounds from images"
    )
    parser.add_argument("input", help="Input image or folder")
    parser.add_argument("-o", "--output", help="Output file or folder")
    parser.add_argument("--batch", action="store_true",
                        help="Process folder of images")

    args = parser.parse_args()

    from aikit import BackgroundRemover
    r = BackgroundRemover()

    if args.batch:
        results = r.batch(args.input, args.output)
        print(f"Processed {len(results)} images")
    else:
        from pathlib import Path
        if not args.output:
            p = Path(args.input)
            output = str(p.parent / f"{p.stem}_nobg.png")
        else:
            output = args.output
        r.remove(args.input, output)


def main():
    """Main CLI dispatcher."""
    commands = {
        "imagine": imagine,
        "transcribe": transcribe,
        "embed": embed,
        "segment": segment,
        "rmbg": rmbg,
    }

    # Check if first arg is a valid command - if so, dispatch directly
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        cmd = sys.argv[1]
        sys.argv = [cmd] + sys.argv[2:]
        commands[cmd]()
        return

    # Otherwise show help
    print("""AI Toolkit - CLI tools for local AI pipelines

Usage: aikit <command> [options]

Commands:
  imagine     Generate images with Stable Diffusion
  transcribe  Transcribe audio/video to text
  embed       Semantic search and embedding
  segment     Image segmentation with SAM
  rmbg        Remove image backgrounds

Examples:
  aikit imagine "a cyberpunk city at night"
  aikit transcribe meeting.mp3 --model large-v3
  aikit embed index ./docs
  aikit embed search "how does auth work"
  aikit segment photo.jpg --point 100 200
  aikit rmbg photo.jpg

Run 'aikit <command> --help' for command-specific help.
""")


if __name__ == "__main__":
    main()
