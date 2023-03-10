from typing import List
import json

from a2t.tasks.tuple_classification import TACREDFeatures
from .base import Dataset


class DNLIRelationClassificationDataset(Dataset):
    """A class to handle DNLI datasets.

    This class converts DNLI data files into a list of `a2t.tasks.TACREDFeatures`.
    While DNLI has a different format to TACRED, it shares enough similarities to exploit them for easier conversion;
    The dataset is effectively a merge of PersonaChat for content (conversations) and TACRED for relation labels.
    For more information, see original paper by Welleck et al. arXiv:1811.00671v2

    DNLI asks if sentence2 is entailed by sentence 1: therefore, subject and object are taken from second sentence,
    but context is the first sentence. Then, the label is logical.
    (otherwise it would always be entailed as the triple belongs to the first sentence)
    """

    def __init__(self, input_path: str, labels: List[str], *args, **kwargs) -> None:
        """
        Args:
            input_path (str): The path to the input file.
            labels (List[str]): The possible label set of the dataset.
        """
        super().__init__(labels=labels, *args, **kwargs)

        with open(input_path, "rt") as f:
            for i, line in enumerate(json.load(f)):
                self.append(
                    TACREDFeatures(
                        subj=line['triple2'][0],
                        obj=line['triple2'][2],
                        # doesn't exist in DNLI dataset; can potentially be inferred using an external library
                        # for tagging
                        # inst_type=f"{line['subj_type']}:{line['obj_type']}",
                        context=line['sentence1'],
                        label=line['label'],
                    )
                )
