"""Main evaluation script.

Please consider making a copy and customizing it yourself if you aim to use a custom class that is not already 
defined on the library.

### Usage

```bash 
a2t.evaluation [-h] [--config CONFIG]

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  Config with task (schema) and data information.
```
"""
import argparse
import json
import os
from pprint import pprint
from types import SimpleNamespace
import numpy as np
import torch

# Changed imports to local path to avoid dependency on venv-located library files
from a2t.tasks.tuple_classification import TACREDFeatures
from a2t.base import EntailmentClassifier
from defintions_dnli import DLNIRelationClassificationTask, DNLIRelationClassificationDataset


def main(args):

    with open(args.config, "rt") as f:
        config = SimpleNamespace(**json.load(f))

    os.makedirs(f"experiments/{config.name}", exist_ok=True)

    # all task definitions are "hardcoded" as DNLI can't be added to library easily
    task_class = DLNIRelationClassificationTask
    task = task_class.from_config(args.config)  # (**vars(config))

    dataset_class = DNLIRelationClassificationDataset

    assert hasattr(config, "dev_path") or hasattr(config, "test_path"), "At least a test or dev path must be provided."

    # Run dev evaluation
    if hasattr(config, "dev_path"):
        dev_dataset = dataset_class(config.dev_path, task.labels)
    else:
        dev_dataset = None

    if hasattr(config, "test_path"):
        test_dataset = dataset_class(config.test_path, task.labels)
    else:
        test_dataset = None

    results = {}
    for pretrained_model in config.nli_models:

        nlp = EntailmentClassifier(pretrained_model, **vars(config))

        results[pretrained_model] = {}

        if dev_dataset:
            _, output = nlp(task=task, features=dev_dataset, negative_threshold=0.0, return_raw_output=True, **vars(config))

            dev_labels = dev_dataset.labels

            # Save the output
            os.makedirs(
                f"experiments/{config.name}/{pretrained_model}/dev",
                exist_ok=True,
            )
            np.save(
                f"experiments/{config.name}/{pretrained_model}/dev/output.npy",
                output,
            )
            np.save(
                f"experiments/{config.name}/{pretrained_model}/dev/labels.npy",
                dev_labels,
            )

            # If dev data then optimize the threshold on it
            dev_results = task.compute_metrics(dev_labels, output, threshold="optimize")
            results[pretrained_model]["dev"] = dev_results

            with open(f"experiments/{config.name}/results.json", "wt") as f:
                json.dump(results, f, indent=4)

        if test_dataset:
            _, output = nlp(task=task, features=test_dataset, negative_threshold=0.0, return_raw_output=True, **vars(config))

            test_labels = test_dataset.labels

            # Save the output
            os.makedirs(
                f"experiments/{config.name}/{pretrained_model}/test",
                exist_ok=True,
            )
            np.save(
                f"experiments/{config.name}/{pretrained_model}/test/output.npy",
                output,
            )
            np.save(
                f"experiments/{config.name}/{pretrained_model}/test/labels.npy",
                test_labels,
            )

            optimal_threshold = 0.5 if not dev_dataset else dev_results["optimal_threshold"]
            test_results = task.compute_metrics(test_labels, output, threshold=optimal_threshold)
            results[pretrained_model]["test"] = test_results

            with open(f"experiments/{config.name}/results.json", "wt") as f:
                json.dump(results, f, indent=4)

        nlp.clear_gpu_memory()
        del nlp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("a2t.evaluation")
    parser.add_argument("--config", type=str, help="Config with task (schema) and data information.")

    args = parser.parse_args(['--config', 'config_dnli.json'])
    main(args)
