from __future__ import annotations

from egocentric_dataset_test.competition.hf_space_demo import create_hf_space_demo


app = create_hf_space_demo()


if __name__ == "__main__":
    app.launch()