# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
eval.py
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
import torchvision.utils as vutils
from time import time

img_names = [
    "frame_00003.JPG",
    "frame_00005.JPG"
]

def get_average_image_metrics(
    pipeline,
    data_loader,
    image_prefix: str,
    step: Optional[int] = None,
    output_path: Optional[Path] = None,
    get_std: bool = False,
):
    pipeline.eval()
    metrics_dict_list = []
    num_images = len(data_loader)
    if output_path is not None:
        output_path.mkdir(exist_ok=True, parents=True)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Evaluating all images...", total=num_images)
        idx = 0
        for camera, batch in data_loader:
            assert idx < 3, "broken dataset with unknown issue."
            # time this the following line
            inner_start = time()
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            height, width = camera.height, camera.width
            num_rays = height * width
            metrics_dict, image_dict = pipeline.model.get_image_metrics_and_images(outputs, batch)
            CONSOLE.log(f"Processing...")
            if output_path is not None:
                for key in image_dict.keys():
                    if key != "img":
                        continue
                    image = image_dict[key]  # [H, W, C] order
                    # Preserve only the right half of the image
                    image_width = image.shape[1]
                    right_half = image[:, image_width // 2:, :]
                    image = right_half.clone()  # Create a new tensor with only the right half
                    vutils.save_image(image.permute(2, 0, 1).cpu(), output_path / img_names[idx])

            progress.advance(task)
            idx = idx + 1

    pipeline.train()
    return metrics_dict

@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = Path("submit/")

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        self.render_output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = get_average_image_metrics(pipeline, pipeline.datamanager.fixed_indices_eval_dataloader, "", output_path=self.render_output_path, get_std=True)
        CONSOLE.log("DONE.")
        # Clean up unnecessary codes and create a zip file
        zip_path = self.render_output_path.parent / f"{self.render_output_path.name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for image_file in img_names:
                file_path = self.render_output_path / image_file
                if file_path.exists():
                    zipf.write(file_path, image_file)
                else:
                    CONSOLE.log(f"Warning: {image_file} not found in {self.render_output_path}")
        
        CONSOLE.log(f"Created zip file: {zip_path}")
        


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
