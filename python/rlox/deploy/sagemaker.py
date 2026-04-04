"""AWS SageMaker deployment utilities for rlox training."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field


_ENTRY_SCRIPT_NAME = "rlox_sagemaker_entry.py"


def generate_entry_script(config_path: str) -> str:
    """Generate SageMaker entry point script content.

    The generated script imports rlox and runs training from the given
    config path, falling back to the ``SM_CHANNEL_CONFIG`` environment
    variable set by SageMaker.

    Parameters
    ----------
    config_path : str
        Default config path (S3 URI or local path).

    Returns
    -------
    str
        Valid Python script content.
    """
    return textwrap.dedent(f"""\
        import os
        from rlox import train_from_config

        config = os.environ.get("SM_CHANNEL_CONFIG", "{config_path}")
        train_from_config(config)
    """)


@dataclass
class SageMakerEstimator:
    """Launch rlox training on AWS SageMaker.

    This is a lightweight descriptor; calling :meth:`fit` requires
    ``boto3`` and ``sagemaker`` to be installed.

    Parameters
    ----------
    role : str
        IAM role ARN for SageMaker execution.
    instance_type : str
        SageMaker instance type (default ``"ml.g5.xlarge"``).
    instance_count : int
        Number of training instances (default 1).
    framework_version : str
        Python/PyTorch framework version (default ``"2.2"``).
    """

    role: str
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    framework_version: str = "2.2"
    _job_name: str | None = field(default=None, repr=False)

    @property
    def entry_point(self) -> str:
        """Return the entry point script filename."""
        return _ENTRY_SCRIPT_NAME

    def fit(self, config_path: str) -> str:
        """Submit a training job to SageMaker.

        Generates a small entry-point script, then submits via the
        SageMaker PyTorch estimator.

        Parameters
        ----------
        config_path : str
            S3 URI or local path to the training config.

        Returns
        -------
        str
            The SageMaker training job name.

        Raises
        ------
        ImportError
            If ``sagemaker`` SDK is not installed.
        """
        try:
            import sagemaker  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "AWS SageMaker SDK required: pip install sagemaker"
            ) from exc

        import tempfile
        from pathlib import Path

        from sagemaker.pytorch import PyTorch

        # Write entry script to a temp directory
        tmpdir = tempfile.mkdtemp(prefix="rlox_sm_")
        entry_path = Path(tmpdir) / _ENTRY_SCRIPT_NAME
        entry_path.write_text(generate_entry_script(config_path))

        estimator = PyTorch(
            entry_point=_ENTRY_SCRIPT_NAME,
            source_dir=tmpdir,
            role=self.role,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            framework_version=self.framework_version,
            py_version="py312",
        )
        estimator.fit(config_path)
        self._job_name = estimator.latest_training_job.name
        return self._job_name
