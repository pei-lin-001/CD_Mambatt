from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cd_mambatt.data import CMAPSSSplit


@dataclass(frozen=True)
class CrossDomainTask:
    name: str
    source_subset: str
    target_subset: str
    difficulty: str
    description: str


@dataclass(frozen=True)
class FewShotTargetPartition:
    labeled_units: np.ndarray
    validation_units: np.ndarray
    unlabeled_units: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "labeled_units": self.labeled_units.tolist(),
            "validation_units": self.validation_units.tolist(),
            "unlabeled_units": self.unlabeled_units.tolist(),
        }


DEFAULT_CROSS_DOMAIN_TASKS: dict[str, CrossDomainTask] = {
    "FD001_TO_FD003": CrossDomainTask(
        name="FD001_TO_FD003",
        source_subset="FD001",
        target_subset="FD003",
        difficulty="stage1",
        description="single-condition to single-condition with different fault modes",
    ),
    "FD003_TO_FD001": CrossDomainTask(
        name="FD003_TO_FD001",
        source_subset="FD003",
        target_subset="FD001",
        difficulty="stage1",
        description="reverse single-condition transfer with different fault modes",
    ),
    "FD002_TO_FD004": CrossDomainTask(
        name="FD002_TO_FD004",
        source_subset="FD002",
        target_subset="FD004",
        difficulty="stage1",
        description="multi-condition to multi-condition with different fault modes",
    ),
    "FD004_TO_FD002": CrossDomainTask(
        name="FD004_TO_FD002",
        source_subset="FD004",
        target_subset="FD002",
        difficulty="stage1",
        description="reverse multi-condition transfer with different fault modes",
    ),
    "FD001_TO_FD002": CrossDomainTask(
        name="FD001_TO_FD002",
        source_subset="FD001",
        target_subset="FD002",
        difficulty="stage2",
        description="single-condition to multi-condition single-fault transfer",
    ),
    "FD001_TO_FD004": CrossDomainTask(
        name="FD001_TO_FD004",
        source_subset="FD001",
        target_subset="FD004",
        difficulty="stage2",
        description="simplest to most complex transfer",
    ),
}


def resolve_cross_domain_task(
    task_name: str | None,
    *,
    source_subset: str | None = None,
    target_subset: str | None = None,
) -> CrossDomainTask:
    if task_name:
        key = task_name.upper()
        if key not in DEFAULT_CROSS_DOMAIN_TASKS:
            raise ValueError(f"Unknown cross-domain task '{task_name}'. Expected one of: {sorted(DEFAULT_CROSS_DOMAIN_TASKS)}")
        return DEFAULT_CROSS_DOMAIN_TASKS[key]

    if not source_subset or not target_subset:
        raise ValueError("Provide either --task or both --source-subset and --target-subset")

    source_subset = source_subset.upper()
    target_subset = target_subset.upper()
    if source_subset == target_subset:
        raise ValueError("source_subset and target_subset must be different for cross-domain experiments")
    return CrossDomainTask(
        name=f"{source_subset}_TO_{target_subset}",
        source_subset=source_subset,
        target_subset=target_subset,
        difficulty="custom",
        description="custom cross-domain task",
    )


def build_few_shot_target_partition(
    split: CMAPSSSplit,
    *,
    num_shots: int,
    num_val_units: int,
    seed: int,
) -> FewShotTargetPartition:
    if split.split != "train":
        raise ValueError("Few-shot partitioning expects a training split")
    if num_shots <= 0:
        raise ValueError("num_shots must be > 0")
    if num_val_units <= 0:
        raise ValueError("num_val_units must be > 0")

    unique_units = np.unique(split.unit_ids)
    required_units = num_shots + num_val_units
    if unique_units.size <= required_units:
        raise ValueError(
            f"Target split has only {unique_units.size} units, but {required_units + 1} or more are needed "
            "to create labeled, validation, and unlabeled pools"
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_units)
    labeled_units = np.sort(shuffled[:num_shots]).astype(np.int32)
    validation_units = np.sort(shuffled[num_shots : num_shots + num_val_units]).astype(np.int32)
    unlabeled_units = np.sort(shuffled[num_shots + num_val_units :]).astype(np.int32)
    if unlabeled_units.size == 0:
        raise ValueError("Few-shot partition left no unlabeled target units")

    return FewShotTargetPartition(
        labeled_units=labeled_units,
        validation_units=validation_units,
        unlabeled_units=unlabeled_units,
    )
