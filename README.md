<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: End-To-End Named-Entity-Recognition for Cyber Threat Intelligence.

This has to be added.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `info` | Creates the README.md of this project. |
| `convert` | Convert the CyNER data to .spacy format. |
| `initialize` | Initialize the configuration for training. Make sure to add `[paths]` in `base_config.cfg` like `train = "./corpus/CyNER/train.spacy"`. This is important for debugging the data before training. |
| `debug-data` | This method debugs the training and validation data specified in `[path]` against the model as declared in `training_config.cfg`. |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/CyNER/train.txt` | URL |  |
| `assets/CyNER/valid.txt` | URL |  |
| `assets/CyNER/test.txt` | URL |  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->