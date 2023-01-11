import os
from argparse import ArgumentParser
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict
import json
from pprint import pprint
import random
import sys


# set up classes and parser
@dataclass
class REInputFeatures:
    subj: str
    obj: str
    context: str
    pair_type: str = None
    label: str = None


@dataclass
class MNLIInputFeatures:
    premise: str
    hypothesis: str
    label: int


# sys.path.append("./")
parser = ArgumentParser()
input_path = os.path.join("dialogue_nli", "dialogue_nli_test.jsonl")
output_path = os.path.join("dialogue_nli", "dialogue_nli_test.mlni.jsonl")
parser.add_argument("--input_file", type=str, default=input_path)
# parser.add_argument('--model', type=str, default='microsoft/deberta-v2-xlarge-mnli')
parser.add_argument("--output_file", type=str, default=output_path)
parser.add_argument("--negative_pattern", action="store_true", default=False)
parser.add_argument("--negn", type=int, default=1)

args = parser.parse_args()

# available relations
relation_labels = [
    "has_profession",
    "employed_by_general",
    "like_food",
    "physical_attribute",
    "place_origin",
    "like_read",
    "have",
    "have_chidren",
    "like_activity",
    "have_sibling",
    "like_drink",
    "marital_status",
    "live_in_citystatecountry",
    "employed_by_company",
    "teach", "dislike",
    "attend_school",
    "like_music",
    "job_status",
    "favorite_season",
    "like_animal",
    "want_do",
    "have_pet",
    "own",
    "favorite_activity",
    "has_hobby",
    "has_ability",
    "school_status",
    "favorite_color",
    "favorite_music",
    "has_age",
    "misc_attribute",
    "live_in_general",
    "gender",
    "favorite_food",
    "like_general",
    "previous_profession",
    "have_vehicle",
    "like_sports",
    "favorite_drink",
    "favorite_animal",
    "has_degree",
    "like_goto",
    "favorite_music_artist",
    "want",
    "want_job",
    "like_watching",
    "favorite_sport",
    "member_of",
    "have_family",
    "not_have",
    "favorite_show",
    "like_movie",
    "favorite_hobby",
    "favorite",
    "nationality",
    "favorite_movie"
]

relation_templates = {
    "has_profession": [
        "{subj} work as {obj}.",
        "{subj} works as {obj}."
    ],
    "employed_by_general": [
        "{subj} am an employee of {obj}.",
        "{subj} is an employee of {obj}."
    ],
    "employed_by_company": [
        "{subj} am member of {obj}.",
        "{subj} is member of {obj}.",
        "{subj} am an employee of {obj}.",
        "{subj} is an employee of {obj}."
    ],
    "previous_profession": [
        "{subj} used to work as {obj}.",
        "{subj} used to have {obj} as a job."
    ],
    "member_of": [
        "{subj} am member of {obj}.",
        "{subj} is member of {obj}.",
        "{obj} joined {subj}."
    ],
    "teach": [
        "{subj} teach {obj}.",
        "{subj} teaches {obj}."
    ],
    "attend_school": [
        "{subj} studied in {obj}.",
        "{subj} graduated from {obj}."
    ],
    "has_degree": [
        "{subj} have a degree in {obj}.",
        "{subj} has a degree in {obj}.",
        "{subj} obtained a degree in {obj}."
    ],
    "school_status": [
        "{subj} am currently {obj}.",
        "{subj} is currently {obj}."
    ],
    "job_status": [
        "{subj} am currently {obj}.",
        "{subj} is currently {obj}."
    ],
    "place_origin": [
        "{subj} come from {obj}.",
        "{subj} comes from {obj}.",
        "{subj} was born in {obj}."
    ],
    "nationality": [
        "{subj} have a {obj} nationality.",
        "{subj} has a {obj} nationality.",
        "{obj} is the nationality of {subj}."
    ],
    "live_in_citystatecountry": [
        "{subj} live in {obj}.",
        "{subj} lives in {obj}.",
        "{subj} have a legal order to stay in {obj}.",
        "{subj} has a legal order to stay in {obj}."
    ],
    "live_in_general": [
        "{subj} live in {obj}.",
        "{subj} lives in {obj}.",
        "{subj} have a legal order to stay in {obj}.",
        "{subj} has a legal order to stay in {obj}."
    ],
    "physical_attribute": [
        "{subj} am {obj}.",
        "{subj} is {obj}."
    ],
    "gender": [
        "{subj} am a {obj}.",
        "{subj} is a {obj}.",
        "{subj}\u00e2\u0080\u0099s gender is {obj}."
    ],
    "have_chidren": [
        "{subj} am the parent of {obj}.",
        "{subj} is the parent of {obj}.",
        "{subj} am the mother of {obj}.",
        "{subj} is the mother of {obj}.",
        "{subj} am the father of {obj}.",
        "{subj} is the father of {obj}.",
        "{obj} is the son of {subj}.",
        "{obj} is the daughter of {subj}."
    ],
    "have_sibling": [
        "{subj} and {obj} are siblings.",
        "{subj} am brother of {obj}.",
        "{subj} is brother of {obj}.",
        "{subj} am sister of {obj}.",
        "{subj} is sister of {obj}."
    ],
    "have_family": [
        "{subj} and {obj} are family.",
        "{subj} am a brother in law of {obj}.",
        "{subj} is a brother in law of {obj}.",
        "{subj} am a sister in law of {obj}.",
        "{subj} is a sister in law of {obj}.",
        "{subj} am the cousin of {obj}.",
        "{subj} is the cousin of {obj}.",
        "{subj} am the uncle of {obj}.",
        "{subj} is the uncle of {obj}.",
        "{subj} am the aunt of {obj}.",
        "{subj} is the aunt of {obj}.",
        "{subj} am the grandparent of {obj}.",
        "{subj} is the grandparent of {obj}.",
        "{subj} am the grandmother of {obj}.",
        "{subj} is the grandmother of {obj}.",
        "{subj} am the grandfather of {obj}.",
        "{subj} is the grandfather of {obj}.",
        "{subj} am the grandson of {obj}.",
        "{subj} is the grandson of {obj}.",
        "{subj} am the granddaughter of {obj}.",
        "{subj} is the granddaughter of {obj}."
    ],
    "marital_status": [
        "{subj} am the spouse of {obj}.",
        "{subj} is the spouse of {obj}.",
        "{subj} am the wife of {obj}.",
        "{subj} is the wife of {obj}.",
        "{subj} am the husband of {obj}.",
        "{subj} is the husband of {obj}."
    ],
    "have": [
        "{subj} have {obj}.",
        "{subj} has {obj}.",
        "{subj} have a {obj}.",
        "{subj} has a {obj}."
    ],
    "have_pet": [
        "{subj} have a {obj}.",
        "{subj} has a {obj}."
    ],
    "has_hobby": [
        "{subj} have {obj} as a hobby.",
        "{subj} has {obj} as a hobby.",
        "{subj}\u00e2\u0080\u0099s hobby is {obj}."
    ],
    "has_ability": [
        "{subj} have the ability to {obj}.",
        "{subj} has the ability to{obj}.",
        "{subj} can {obj}.",
        "{subj} can do {obj}."
    ],
    "has_age": [
        "{subj} am {obj} years old.",
        "{subj} is {obj} years old."
    ],
    "have_vehicle": [
        "{subj} have a {obj}.",
        "{subj} has a {obj}."
    ],
    "not_have": [
        "{subj} don't have {obj}.",
        "{subj} doesn't have {obj}."
    ],
    "own": [
        "{subj} own {obj}.",
        "{subj} owns {obj}."
    ],
    "like_general": [
        "{subj} like {obj}.",
        "{subj} likes {obj}."
    ],
    "like_food": [
        "{subj} like to eat {obj}.",
        "{subj} likes to eat {obj}."
    ],
    "like_read": [
        "{subj} like to read {obj}.",
        "{subj} likes to read {obj}."
    ],
    "like_activity": [
        "{subj} like to do {obj}.",
        "{subj} like to {obj}.",
        "{subj} likes to do {obj}.",
        "{subj} likes to {obj}."
    ],
    "like_drink": [
        "{subj} like to drink {obj}.",
        "{subj} likes to drink {obj}."
    ],
    "like_music": [
        "{subj} like to listen to {obj}.",
        "{subj} likes to listen to {obj}."
    ],
    "like_animal": [
        "{subj} like {obj}.",
        "{subj} likes {obj}."
    ],
    "like_sports": [
        "{subj} like {obj}.",
        "{subj} likes {obj}."
    ],
    "like_goto": [
        "{subj} like to go to {obj}.",
        "{subj} likes to go to {obj}.",
        "{subj} like going to {obj}.",
        "{subj} likes going to {obj}."
    ],
    "like_movie": [
        "{subj} like the movie {obj}.",
        "{subj} likes the movie {obj}."
    ],
    "like_watching": [
        "{subj} like watching {obj}.",
        "{subj} likes watching {obj}."
    ],
    "dislike": [
        "{subj} don't like {obj}.",
        "{subj} doesn't like {obj}.",
        "{subj} dislike {obj}",
        "{subj} dislikes {obj}"
    ],

    "favorite": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_season": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_activity": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_color": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_music": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_food": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_drink": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_animal": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_music_artist": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_sport": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_show": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_hobby": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],
    "favorite_movie": [
        "{subj} have {obj} as a favorite.",
        "{subj} has {obj} as a favorite.",
        "{subj}\u00e2\u0080\u0099s favorite is {obj}."
    ],

    "want_do": [
        "{subj} want to do {obj}.",
        "{subj} wants to do {obj}.",
        "{subj} want to {obj}.",
        "{subj} wants to {obj}."
    ],
    "want": [
        "{subj} want {obj}.",
        "{subj} wants {obj}.",
    ],
    "want_job": [
        "{subj} want {obj} as a job.",
        "{subj} wants {obj} as a job.",
        "{subj} want to work as {obj}.",
        "{subj} wants to  work as {obj}."
    ],

    "misc_attribute": [
        "{subj} have {obj} as an attribute.",
        "{subj} has {obj} as an attribute."
    ]
}

# problem for templates: sentences can be in first or second person
# not quite sure what these templates do - they are just sentence versions of the available relations in tacred2mlni
templates = [
    "{subj} and {obj} are not related",
    "{subj} work as {obj}.",
    "{subj} works as {obj}.",
    "{subj} am an employee of {obj}.",
    "{subj} is an employee of {obj}.",
    "{subj} used to work as {obj}.",
    "{subj} am member of {obj}.",
    "{subj} is member of {obj}.",
    "{subj} teach {obj}.",
    "{subj} teaches {obj}.",
    "{subj} studied in {obj}.",
    "{subj} have a degree in {obj}.",
    "{subj} has a degree in {obj}.",
    "{subj} am currently {obj}.",
    "{subj} is currently {obj}.",
    "{subj} was born in {obj}.",
    "{obj} is the nationality of {subj}.",
    "{subj} live in {obj}.",
    "{subj} lives in {obj}.",
    "{subj} am {obj}.",
    "{subj} is {obj}.",
    "{subj} am a {obj}.",
    "{subj} is a {obj}.",
    "{subj} am the parent of {obj}.",
    "{subj} is the parent of {obj}.",
    "{subj} and {obj} are siblings.",
    "{subj} and {obj} are family.",
    "{subj} am the spouse of {obj}.",
    "{subj} is the spouse of {obj}.",
    "{subj} have {obj}.",
    "{subj} has {obj}.",
    "{subj} have a {obj}.",
    "{subj} has a {obj}.",
    "{subj} have {obj} as a hobby.",
    "{subj} has {obj} as a hobby.",
    "{subj} have the ability to {obj}.",
    "{subj} has the ability to{obj}.",
    "{subj} am {obj} years old.",
    "{subj} is {obj} years old.",
    "{subj} don't have {obj}.",
    "{subj} doesn't have {obj}.",
    "{subj} own {obj}.",
    "{subj} owns {obj}.",
    "{subj} like {obj}.",
    "{subj} likes {obj}.",
    "{subj} like to eat {obj}.",
    "{subj} likes to eat {obj}.",
    "{subj} like to read {obj}.",
    "{subj} likes to read {obj}.",
    "{subj} like to {obj}.",
    "{subj} likes to {obj}.",
    "{subj} like to drink {obj}.",
    "{subj} likes to drink {obj}.",
    "{subj} like to listen to {obj}.",
    "{subj} likes to listen to {obj}.",
    "{subj} like to go to {obj}.",
    "{subj} likes to go to {obj}.",
    "{subj} like the movie {obj}.",
    "{subj} likes the movie {obj}.",
    "{subj} like watching {obj}.",
    "{subj} likes watching {obj}.",
    "{subj} dislike {obj}",
    "{subj} dislikes {obj}",
    "{subj} have {obj} as a favorite.",
    "{subj} has {obj} as a favorite.",
    "{subj} want to do {obj}.",
    "{subj} wants to do {obj}.",
    "{subj} want {obj}.",
    "{subj} wants {obj}.",
    "{subj} want to work as {obj}.",
    "{subj} wants to  work as {obj}.",
    "{subj} have {obj} as an attribute.",
    "{subj} has {obj} as an attribute."
]

# converts type to MLNI id, but always hardcoded so not super necessary
labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}

# positive: holds the regular templates
# negative: holds all other templates (for a given label) to generate contradictory verbalizations
positive_templates: Dict[str, list] = defaultdict(list)
negative_templates: Dict[str, list] = defaultdict(list)

# removes no-relation template if not used
if not args.negative_pattern:
    templates = templates[1:]

# adds verbalizations to corresponding dictionary
for label in relation_labels:
    for template in templates:
        if template in relation_templates[label]:
            positive_templates[label].append(template)
        else:
            negative_templates[label].append(template)


def dnli2mlni(
        instance: REInputFeatures,
        positive_templates,
        negative_templates,
        templates,
        negn=1,
        posn=1,
):
    if instance.label == "neutral":
        template = random.choices(templates, k=negn)
        return [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["contradiction"],
            )
            for t in template
        ]

    # Generate the positive examples
    mnli_instances = []
    print(len(positive_templates[instance.label]))
    positive_template = random.choices(positive_templates[instance.label], k=posn)
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["entailment"],
            )
            for t in positive_template
        ]
    )

    # Generate the negative templates
    negative_template = random.choices(negative_templates[instance.label], k=negn)
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["neutral"],
            )
            for t in negative_template
        ]
    )

    return mnli_instances


with open(args.input_file, "rt") as f:
    mnli_data = []
    stats = []
    for line in json.load(f):
        # filter out relations with "other", "<blank>" for now as no template specified;
        # other could potentially be manually converted but not really worth it: ~2.8% of test set
        if line['triple2'][1] == 'other' or line['triple2'][1] == '<blank>':
            continue
        """ 
        DNLI asks if sentence2 is entailed by sentence 1:
        therefore, subject and object are taken from second sentence,
        but context is the first sentence. Then, the label is logical
        (otherwise it would always be entailed as the triple belongs to the first sentence)
        """
        mnli_instance = dnli2mlni(
            REInputFeatures(
                subj=line['triple2'][0],
                obj=line['triple2'][2],
                pair_type="",  # possibly inferrable from verified test set
                context=line['sentence1'],
                label=line['triple2'][1]  # label = relation, not entailment!
            ),
            positive_templates,
            negative_templates,
            templates,
            negn=args.negn,
        )
        mnli_data.extend(mnli_instance)
        # stats.append(line["relation"] != "no_relation")

with open(args.output_file, "wt") as f:
    for data in mnli_data:
        f.write(f"{json.dumps(data.__dict__)}\n")
    # json.dump([data.__dict__ for data in mnli_data], f, indent=2)
