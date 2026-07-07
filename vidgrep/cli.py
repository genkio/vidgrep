import sys

USAGE = """\
vidgrep - natural-language search over local video files

usage: vidgrep <command> [args]

commands:
  oneshot  index + cut matching clips in one go, one video at a time
  index    index video file(s) into the search database
  search   search indexed videos, print timestamps
  cut      search, then cut results into clips

run vidgrep <command> --help for details
"""


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return
    cmd = args[0]
    # import inside the branch: keeps --help instant, torch loads only when needed
    if cmd == "index":
        from vidgrep.index import main as cmd_main
    elif cmd == "search":
        from vidgrep.search import main as cmd_main
    elif cmd == "cut":
        from vidgrep.cut import main as cmd_main
    elif cmd == "oneshot":
        from vidgrep.oneshot import main as cmd_main
    else:
        sys.exit(f"unknown command: {cmd}\n\n{USAGE}")
    sys.argv = [f"vidgrep {cmd}", *args[1:]]
    cmd_main()
