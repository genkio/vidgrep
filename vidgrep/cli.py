import sys

from vidgrep import __version__

USAGE = """\
vidgrep - natural-language search over local video files

usage: vidgrep <command> [args]

commands:
  oneshot        index + cut matching clips in one go, one video at a time
  index          index video file(s) into the search database
  search         search indexed videos, print timestamps
  cut            search, then cut results into clips
  export-encoder export a portable text encoder for torch-free cutting
  version        print the installed version

run vidgrep <command> --help for details
"""


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return
    cmd = args[0]
    if cmd in ("version", "--version", "-V"):
        print(f"vidgrep {__version__}")
        return
    # lazy imports keep --help instant, torch loads only when a command runs
    if cmd == "index":
        from vidgrep.index import main as cmd_main
    elif cmd == "search":
        from vidgrep.search import main as cmd_main
    elif cmd == "cut":
        from vidgrep.cut import main as cmd_main
    elif cmd == "oneshot":
        from vidgrep.oneshot import main as cmd_main
    elif cmd == "export-encoder":
        from vidgrep.export import main as cmd_main
    else:
        sys.exit(f"unknown command: {cmd}\n\n{USAGE}")
    sys.argv = [f"vidgrep {cmd}", *args[1:]]
    cmd_main()
