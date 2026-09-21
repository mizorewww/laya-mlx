"""Command-line prediction and checkpoint conversion."""

import argparse
import json
from pathlib import Path

from . import __version__
from .agent import DTYPES, Agent


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya-mlx", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("predict", "convert"):
        sub = commands.add_parser(command)
        sub.add_argument("--model", default="convaiinnovations/laya")
        sub.add_argument("--subfolder")
        sub.add_argument("--revision")
        sub.add_argument("--dtype", choices=DTYPES, default="float16")
        if command == "predict":
            source = sub.add_mutually_exclusive_group()
            source.add_argument("--state", help="Plain text input")
            source.add_argument("--state-file", type=Path, help="JSON state file")
            sub.add_argument("--image", type=Path, help="Image file scored with CLIP")
            sub.add_argument(
                "--questions", required=True, type=Path, help="JSON question definitions"
            )
            sub.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
            sub.add_argument("--batch-size", type=int, default=16)
        else:
            sub.add_argument("--output", type=Path, required=True)
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
    else:
        if args.state is None and args.state_file is None and args.image is None:
            parser.error("predict needs --state, --state-file or --image")
        state = args.state
        if args.state_file is not None:
            state = json.loads(args.state_file.read_text())
        questions = json.loads(args.questions.read_text())
        agent = Agent(
            args.model,
            device=args.device,
            dtype=args.dtype,
            revision=args.revision,
            subfolder=args.subfolder,
            batch_size=args.batch_size,
        )
        result = agent.predict(state, questions, image=args.image)
        print(json.dumps(result, ensure_ascii=False, indent=2))
