"""Command-line prediction, conversion, quantization, device profiling, and serving."""

import argparse
import json
from pathlib import Path

from . import __version__

DTYPES = ("bfloat16", "float16", "float32")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya-mlx", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    # predict
    p_predict = commands.add_parser("predict", help="Run decisions on input state")
    p_predict.add_argument("--model", default="convaiinnovations/laya")
    p_predict.add_argument("--subfolder")
    p_predict.add_argument("--revision")
    p_predict.add_argument("--dtype", choices=DTYPES, default="float16")
    source = p_predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", help="Plain text input")
    source.add_argument("--state-file", type=Path, help="JSON state file")
    p_predict.add_argument(
        "--questions", required=True, type=Path, help="JSON question definitions"
    )
    p_predict.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    p_predict.add_argument("--batch-size", type=int, default=16)
    p_predict.add_argument(
        "--quantize", type=int, choices=(4, 8), help="Quantize encoder weights (4 or 8 bits)"
    )
    p_predict.add_argument(
        "--low-memory", action="store_true", help="Enable low memory mode with cache clearing"
    )
    p_predict.add_argument("--compile", action="store_true", help="Compile model execution graph")
    p_predict.add_argument(
        "--cache-prompts", action="store_true", help="Cache repeated prompt token prefixes"
    )

    # convert
    p_convert = commands.add_parser("convert", help="Convert PyTorch checkpoint to MLX")
    p_convert.add_argument("--model", default="convaiinnovations/laya")
    p_convert.add_argument("--subfolder")
    p_convert.add_argument("--revision")
    p_convert.add_argument("--dtype", choices=DTYPES, default="float16")
    p_convert.add_argument("--output", type=Path, required=True, help="Destination directory")

    # quantize
    p_quantize = commands.add_parser(
        "quantize", help="Export offline 4-bit or 8-bit quantized MLX checkpoint"
    )
    p_quantize.add_argument("--model", default="convaiinnovations/laya")
    p_quantize.add_argument("--subfolder")
    p_quantize.add_argument("--revision")
    p_quantize.add_argument("--dtype", choices=DTYPES, default="float16")
    p_quantize.add_argument(
        "--output", type=Path, required=True, help="Destination directory for quantized model"
    )
    p_quantize.add_argument(
        "--bits", type=int, choices=(4, 8), default=8, help="Quantization bits (4 or 8)"
    )
    p_quantize.add_argument(
        "--group-size",
        type=int,
        choices=(32, 64, 128),
        default=64,
        help="Quantization group size",
    )
    p_quantize.add_argument(
        "--quantize-heads",
        action="store_true",
        help="Also quantize decision heads (default: False)",
    )

    # serve
    p_serve = commands.add_parser("serve", help="Start local HTTP/SSE decision server")
    p_serve.add_argument("--model", default="convaiinnovations/laya")
    p_serve.add_argument("--subfolder")
    p_serve.add_argument("--revision")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    p_serve.add_argument("--port", type=int, default=8080, help="Port to listen on")
    p_serve.add_argument("--dtype", choices=DTYPES, default="float16")
    p_serve.add_argument(
        "--quantize", type=int, choices=(4, 8), help="Quantize encoder weights (4 or 8 bits)"
    )
    p_serve.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    p_serve.add_argument("--batch-size", type=int, default=16)
    p_serve.add_argument(
        "--low-memory", action="store_true", help="Enable low memory mode with cache clearing"
    )

    # device
    p_device = commands.add_parser("device", help="Inspect host hardware and show optimal settings")
    p_device.add_argument("--json", action="store_true", help="Output in JSON format")

    args = parser.parse_args(argv)

    if args.command == "convert":
        from .convert import convert

        result = convert(
            args.model,
            args.output,
            dtype=args.dtype,
            revision=args.revision,
            subfolder=args.subfolder,
        )
        print(json.dumps({"output": str(result), "dtype": args.dtype}))

    elif args.command == "quantize":
        from .quantize import export_quantized

        result = export_quantized(
            args.model,
            args.output,
            bits=args.bits,
            group_size=args.group_size,
            quantize_heads=args.quantize_heads,
            dtype=args.dtype,
            revision=args.revision,
            subfolder=args.subfolder,
        )
        print(
            json.dumps(
                {
                    "output": str(result),
                    "bits": args.bits,
                    "group_size": args.group_size,
                    "dtype": args.dtype,
                }
            )
        )

    elif args.command == "serve":
        from .agent import Agent
        from .server import serve

        agent = Agent(
            args.model,
            device=args.device,
            dtype=args.dtype,
            quantize=args.quantize,
            batch_size=args.batch_size,
            low_memory=args.low_memory,
            revision=args.revision,
            subfolder=args.subfolder,
        )
        serve(agent, host=args.host, port=args.port)

    elif args.command == "device":
        from .device import profile_hardware, suggest_device_config

        profile = profile_hardware()
        if args.json:
            out = {
                "platform": profile.platform,
                "is_apple_silicon": profile.is_apple_silicon,
                "chip_name": profile.chip_name,
                "total_memory_gb": profile.total_memory_gb,
                "cpu_cores": profile.cpu_cores,
                "recommended_quantize": profile.recommended_quantize,
                "recommended_batch_size": profile.recommended_batch_size,
                "recommended_dtype": profile.recommended_dtype,
                "low_memory_mode": profile.low_memory_mode,
                "suggested_config": suggest_device_config(),
            }
            print(json.dumps(out, indent=2))
        else:
            print("Laya-MLX Hardware Profile & Recommendations")
            print("===========================================")
            print(f"Platform:              {profile.platform}")
            print(f"Apple Silicon:         {profile.is_apple_silicon}")
            print(f"Processor/Chip:        {profile.chip_name}")
            print(f"Total System Memory:   {profile.total_memory_gb} GB")
            print(f"CPU Cores:             {profile.cpu_cores}")
            print("-------------------------------------------")
            print(
                f"Recommended Quantize:  {f'Q{profile.recommended_quantize} ({profile.recommended_quantize}-bit)' if profile.recommended_quantize else 'None (FP16/FP32)'}"
            )
            print(f"Recommended Batch Size:{profile.recommended_batch_size}")
            print(f"Recommended Precision: {profile.recommended_dtype}")
            print(f"Low Memory Mode:       {profile.low_memory_mode}")

    else:
        from .agent import Agent

        state = args.state if args.state is not None else json.loads(args.state_file.read_text())
        questions = json.loads(args.questions.read_text())
        agent = Agent(
            args.model,
            device=args.device,
            dtype=args.dtype,
            revision=args.revision,
            subfolder=args.subfolder,
            batch_size=args.batch_size,
            quantize=args.quantize,
            low_memory=args.low_memory,
            compile=args.compile,
            cache_prompts=args.cache_prompts,
        )
        print(json.dumps(agent.predict(state, questions), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
