# Relation Extraction in conversational data
This repository contains the code used for my Bachelor Thesis research on Relation Extraction in conversational data using Few-Shot Entailment. It is a clone of Sainz et al.'s [A2T Transformer library](https://github.com/osainz59/Ask2Transformers/), as my work aims at reproducing their results on a different dataset. Additionally, it contains a conversation Relation Extraction dataset under `a2t/data/G2KY`, which is based on Wu et al.'s [Conversation Attribute Extraction Dataset](https://github.com/jasonwu0731/GettingToKnowYou).
In the following, only the major additions/changes I made to the library will be mentioned, as more information regarding the original can be found in the aforementioned repository.

# Relevant changes compared to A2T
Below, the files are listed in order of appearance in the repository which contain code or changes not explained in the original [A2T Transformer library](https://github.com/osainz59/Ask2Transformers/). The dataset is explained separately in the next section.

- `requirements.txt` contains the necessary libraries (and specific versions, if applicable) to run the code.
- `a2t/config_g2ky.json` is an example configuration for the test runs. It contains the labels and used verbalization templates for the G2KY dataset.
- `a2t/definitions_dnli.py` contains the task and dataset definitions necessary to adapt a custom dataset for use with the A2T library. It allows for both the G2KY dataset as well as the [DialogueNLI dataset](https://wellecks.com/dialogue_nli/) used for model training to be used for evaluation.
- `a2t/evaluation_dnli` is a copy of the original evaluation script with some slight adaptations to allow for a custom dataset.
- `scripts/dnli2mnli.py` converts the [DialogueNLI dataset](https://wellecks.com/dialogue_nli/) into the MNLI format so that it can be used for model fine-tuning (as described in the A2T documentation).


# The Relation Extraction dataset
As mentioned, Wu et al.'s [Conversation Attribute Extraction Dataset](https://github.com/jasonwu0731/GettingToKnowYou) was used to create conversation Relation Extraction dataset (named G2KY after the "Getting To Know You: User Attribute Extraction from Dialogues" paper by Wu et al. from which the dataset stems). The `G2KY.ipynb` Jupyter notebook contains the code used to convert the original dataset and perform the sampling for the Few-Shot scenarios.
The actual dataset is at `a2t/data/G2KY`, which contains the full train, development, and test files, a `test_small.json` with 20 samples for debugging purposes, and the test and development Few-Shot scenario files.

