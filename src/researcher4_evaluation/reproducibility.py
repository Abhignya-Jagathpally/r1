"""
Reproducibility infrastructure: Dockerfile, DVC, seeds, environment snapshots.

Provides:
  - Seed management across frameworks
  - Environment snapshot (package versions, git state)
  - Dockerfile generation for containerization
  - DVC pipeline definitions
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import subprocess
import json
import logging
import sys
import platform

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentSnapshot:
    """Snapshot of environment and dependencies."""

    timestamp: str
    python_version: str
    platform_info: str
    git_hash: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    packages: Dict[str, str] = None

    def __post_init__(self):
        if self.packages is None:
            self.packages = {}

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save snapshot as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def capture() -> "EnvironmentSnapshot":
        """Capture current environment state."""
        snapshot = EnvironmentSnapshot(
            timestamp=datetime.now().isoformat(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform_info=platform.platform(),
        )

        # Get git information
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            snapshot.git_hash = git_hash
        except Exception as e:
            logger.warning(f"Could not get git hash: {e}")

        try:
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            snapshot.git_branch = git_branch
        except Exception as e:
            logger.warning(f"Could not get git branch: {e}")

        try:
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            snapshot.git_dirty = len(git_status) > 0
        except Exception as e:
            logger.warning(f"Could not check git status: {e}")

        # Get installed packages
        try:
            import pkg_resources
            packages = {}
            for dist in pkg_resources.working_set:
                packages[dist.project_name] = dist.version
            snapshot.packages = packages
        except Exception as e:
            logger.warning(f"Could not get package list: {e}")

        logger.info("Environment snapshot captured")
        return snapshot


class SeedManager:
    """Manage random seeds across frameworks."""

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set seeds for numpy, python random, and any framework-specific RNG.

        Args:
            seed: Random seed value
        """
        import random

        # Python random
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # Try to set seeds for common ML frameworks
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            tf.keras.utils.set_random_seed(seed)
        except ImportError:
            pass

        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        except ImportError:
            pass

        logger.info(f"Random seeds set to {seed}")

    @staticmethod
    def get_seed_state() -> Dict:
        """Get current state of random number generators."""
        state = {
            "numpy": {"random_state": str(np.random.get_state())[:100]},
        }

        try:
            import tensorflow as tf
            state["tensorflow"] = {"seed_set": True}
        except ImportError:
            pass

        try:
            import torch
            state["pytorch"] = {"seed_set": True}
        except ImportError:
            pass

        return state


class DockerfileGenerator:
    """Generate Dockerfile for reproducible environments."""

    def __init__(
        self,
        base_image: str = "python:3.10-slim",
        requirements_file: Optional[str] = None,
    ):
        """
        Initialize Dockerfile generator.

        Args:
            base_image: Base Docker image
            requirements_file: Path to requirements.txt
        """
        self.base_image = base_image
        self.requirements_file = requirements_file

    def generate(self, output_path: str = "Dockerfile") -> str:
        """
        Generate Dockerfile content.

        Args:
            output_path: Path to write Dockerfile

        Returns:
            Dockerfile content
        """
        requirements_section = ""
        if self.requirements_file:
            requirements_section = f"""
COPY {self.requirements_file} /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
"""

        dockerfile_content = f"""FROM {self.base_image}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
{requirements_section}

# Copy application code
COPY . /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "src.researcher4_evaluation"]
"""

        with open(output_path, "w") as f:
            f.write(dockerfile_content)

        logger.info(f"Dockerfile written to {output_path}")

        return dockerfile_content


class DVCPipelineBuilder:
    """Build DVC pipeline definitions for reproducibility."""

    def __init__(self):
        """Initialize DVC pipeline builder."""
        self.stages = {}

    def add_stage(
        self,
        name: str,
        cmd: str,
        deps: Optional[List[str]] = None,
        outs: Optional[List[str]] = None,
        params: Optional[List[str]] = None,
    ) -> None:
        """
        Add a stage to the DVC pipeline.

        Args:
            name: Stage name
            cmd: Command to run
            deps: Input dependencies
            outs: Output artifacts
            params: Parameter dependencies
        """
        stage = {
            "cmd": cmd,
        }

        if deps:
            stage["deps"] = deps

        if outs:
            stage["outs"] = outs

        if params:
            stage["params"] = params

        self.stages[name] = stage

        logger.debug(f"Added DVC stage: {name}")

    def generate_dvc_yaml(self, output_path: str = "dvc.yaml") -> str:
        """
        Generate dvc.yaml file.

        Args:
            output_path: Path to write dvc.yaml

        Returns:
            YAML content
        """
        import yaml

        dvc_config = {"stages": self.stages}

        with open(output_path, "w") as f:
            yaml.dump(dvc_config, f, default_flow_style=False)

        logger.info(f"DVC pipeline written to {output_path}")

        return yaml.dump(dvc_config)

    def generate_pipeline_dag(self, output_path: str = "pipeline_dag.txt") -> None:
        """
        Generate simple text visualization of pipeline DAG.

        Args:
            output_path: Path to write DAG visualization
        """
        dag_text = "DVC Pipeline DAG\n"
        dag_text += "================\n\n"

        for stage_name, stage_config in self.stages.items():
            dag_text += f"{stage_name}:\n"

            if "deps" in stage_config:
                for dep in stage_config["deps"]:
                    dag_text += f"  <- {dep}\n"

            dag_text += f"  CMD: {stage_config['cmd']}\n"

            if "outs" in stage_config:
                for out in stage_config["outs"]:
                    dag_text += f"  -> {out}\n"

            dag_text += "\n"

        with open(output_path, "w") as f:
            f.write(dag_text)

        logger.info(f"Pipeline DAG written to {output_path}")


class ReproducibilityManager:
    """
    Central manager for reproducibility.
    Coordinates seeds, environment snapshots, and infrastructure.
    """

    def __init__(self, output_dir: str = "./reproducibility"):
        """
        Initialize reproducibility manager.

        Args:
            output_dir: Directory for reproducibility artifacts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = None
        self.snapshot = None

    def setup(self, seed: int = 42) -> None:
        """
        Setup reproducibility infrastructure.

        Args:
            seed: Random seed
        """
        # Set seeds
        SeedManager.set_seed(seed)
        self.seed = seed

        # Capture environment
        self.snapshot = EnvironmentSnapshot.capture()
        self.snapshot.to_json(str(self.output_dir / "environment_snapshot.json"))

        logger.info(f"Reproducibility setup complete (seed={seed})")

    def generate_dockerfile(
        self,
        base_image: str = "python:3.10-slim",
        requirements_file: Optional[str] = None,
    ) -> Path:
        """
        Generate Dockerfile.

        Args:
            base_image: Base Docker image
            requirements_file: Path to requirements.txt

        Returns:
            Path to generated Dockerfile
        """
        generator = DockerfileGenerator(
            base_image=base_image,
            requirements_file=requirements_file,
        )

        dockerfile_path = self.output_dir / "Dockerfile"
        generator.generate(str(dockerfile_path))

        return dockerfile_path

    def generate_dvc_pipeline(self) -> Path:
        """
        Generate DVC pipeline configuration.

        Returns:
            Path to generated dvc.yaml
        """
        builder = DVCPipelineBuilder()

        # Example stages
        builder.add_stage(
            name="preprocess",
            cmd="python src/preprocessing.py",
            deps=["data/raw/"],
            outs=["data/processed/"],
        )

        builder.add_stage(
            name="train",
            cmd="python src/train.py",
            deps=["data/processed/", "src/model.py"],
            outs=["models/model.pkl"],
            params=["config.yaml:model"],
        )

        builder.add_stage(
            name="evaluate",
            cmd="python src/evaluate.py",
            deps=["models/model.pkl", "data/processed/"],
            outs=["results/metrics.json"],
        )

        dvc_yaml_path = self.output_dir / "dvc.yaml"
        builder.generate_dvc_yaml(str(dvc_yaml_path))

        dag_path = self.output_dir / "pipeline_dag.txt"
        builder.generate_pipeline_dag(str(dag_path))

        return dvc_yaml_path

    def generate_summary(self) -> Path:
        """
        Generate reproducibility summary.

        Returns:
            Path to summary file
        """
        summary = f"""
Reproducibility Summary
=======================

Generated: {datetime.now().isoformat()}

Seed: {self.seed}

Environment:
  Python Version: {self.snapshot.python_version if self.snapshot else 'N/A'}
  Platform: {self.snapshot.platform_info if self.snapshot else 'N/A'}
  Git Hash: {self.snapshot.git_hash if self.snapshot else 'N/A'}
  Git Branch: {self.snapshot.git_branch if self.snapshot else 'N/A'}
  Git Dirty: {self.snapshot.git_dirty if self.snapshot else 'N/A'}

Artifacts:
  - environment_snapshot.json: Full environment and package versions
  - Dockerfile: Container image definition
  - dvc.yaml: Data pipeline definition
  - pipeline_dag.txt: Pipeline visualization
"""

        summary_path = self.output_dir / "reproducibility_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)

        logger.info(f"Reproducibility summary written to {summary_path}")

        return summary_path
