from __future__ import annotations

import json

from graph import run_workflow


def main() -> None:
    result = run_workflow()
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
