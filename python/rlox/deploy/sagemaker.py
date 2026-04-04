"""AWS SageMaker deployment utilities for rlox training."""

from __future__ import annotations

from dataclasses import dataclass, field


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

    def fit(self, config_path: str) -> str:
        """Submit a training job to SageMaker.

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

        from sagemaker.pytorch import PyTorch

        estimator = PyTorch(
            entry_point="python -m rlox train --config config.yaml",
            role=self.role,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            framework_version=self.framework_version,
            py_version="py312",
        )
        estimator.fit(config_path)
        self._job_name = estimator.latest_training_job.name
        return self._job_name
