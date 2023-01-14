from typing import Dict, List
import json


from a2t.data.base import Dataset
from a2t.tasks.tuple_classification import RelationClassificationTask, TACREDFeatures


class DLNIRelationClassificationTask(RelationClassificationTask):
    """A class handler for DNLI Relation Classification task. It inherits from `RelationClassificationTask` class."""

    def __init__(
        self, labels: List[str], templates: Dict[str, List[str]], valid_conditions: Dict[str, List[str]], **kwargs
    ) -> None:
        """Initialization of the DNLI RelationClassification task

        Args:
            labels (List[str]): The labels for the task.
            templates (Dict[str, List[str]]): The templates/verbalizations for the task.
            valid_conditions (Dict[str, List[str]]): The valid conditions or constraints for the task.
        """
        for key in ["name", "required_variables", "additional_variables", "features_class", "multi_label", "negative_label_id"]:
            kwargs.pop(key, None)
        super().__init__(
            "DNLI Relation Classification task",
            labels=labels,
            required_variables=["subj", "obj"],
            # not used because it doesn't exist in DNLI; might be added later
            # additional_variables=["inst_type"],
            templates=templates,
            valid_conditions=valid_conditions,
            features_class=TACREDFeatures,
            multi_label=True,
            negative_label_id=0,
            **kwargs
        )


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
                        # needs to be mapped depending on whether sentence2 is entailed or not
                        label=relation_mapper(line),
                    )
                )


def relation_mapper(line: Dict):
    """
    Determines the appropriate relation for a given dataset entry,
    depending on the label (positive, neutral, contradiction).

    Currently only supports distinction between positive and non-positive relations;
    TACRED does not have any contradictory statements, so unclear how to handle these.
    For now, contradiction and neutral are both mapped to no_relation.
    """
    if line['label'] == 'positive':
        relation = line['triple2'][1]
    else:
        relation = "no_relation"
    return relation
