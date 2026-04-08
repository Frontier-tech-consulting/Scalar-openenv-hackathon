"""Submission-focused package for the Scaler OpenEnv competition.

Only the deterministic electronics-assembly environment in
`egocentric_dataset_test.competition` is part of the supported runtime surface.
Legacy research code from the earlier Egocentric-100K / MyoSuite / VLA stack has
been removed from this package to keep the Hugging Face Space lean and aligned
with the competition specification.
"""

from egocentric_dataset_test.competition import (
	EgocentricFactoryAction,
	EgocentricFactoryCompetitionEnv,
	EgocentricFactoryObservation,
	EgocentricFactoryState,
	list_task_specs,
)

__all__ = [
	"EgocentricFactoryAction",
	"EgocentricFactoryCompetitionEnv",
	"EgocentricFactoryObservation",
	"EgocentricFactoryState",
	"list_task_specs",
]

__version__ = "0.2.0"
