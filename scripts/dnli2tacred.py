import json

path_dnli = r"D:\Documents\Uni\3\Thesis\Data\dnli\dialogue_nli\dialogue_nli_test.jsonl"
path_output = r"D:\Documents\Uni\3\Thesis\Data\dnli\dialogue_nli\dnli2tacred_test.json"
"""
Relevant for NLI are:
subj and obj 
context = token/sentence
label = relation
pair_type for restrictions (somewhat optional)
"""


def dnli2tacred(dnli_element):
    tacred_element = {
        'subj': dnli_element['triple2'][0],
        'obj': dnli_element['triple2'][2],
        'relation': dnli_element['label'],
        'token': dnli_element['sentence1'].split(),
        'pair_type': ""
    }
    return tacred_element


with open(path_dnli, "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())
    tacred_data = []

    for element in dataset:
        tacred_instance = dnli2tacred(element)
        tacred_data.extend(tacred_instance)

with open(path_output, "wt") as f:
    for data in tacred_data:
        f.write(f"{json.dumps(data.__dict__)}\n")