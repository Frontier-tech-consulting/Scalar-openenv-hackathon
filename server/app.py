from __future__ import annotations

from egocentric_dataset_test.competition.server import app as competition_app
from egocentric_dataset_test.competition.server import main as competition_main

app = competition_app


def main() -> None:
    competition_main(host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
