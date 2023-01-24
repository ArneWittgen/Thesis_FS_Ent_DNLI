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
    relation: str = None
    label: str = None


@dataclass
class MNLIInputFeatures:
    premise: str
    hypothesis: str
    label: str


# sys.path.append("./")
parser = ArgumentParser()
input_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'a2t', 'data', 'dialogue_nli', 'dnli_train_full.jsonl'))
output_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'a2t', 'data', 'dialogue_nli', 'dnli_train_full.mnli.json'))
parser.add_argument("--input_file", type=str, default=input_path)
parser.add_argument("--output_file", type=str, default=output_path)
parser.add_argument("--negative_pattern", action="store_true", default=False)
parser.add_argument("--negn", type=int, default=1)

args = parser.parse_args()

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
    "favorite_place": [
        "{subj} have {obj} as favorite place.",
        "{subj} has {obj} as favorite place.",
        "{subj}\u00e2\u0080\u0099s favorite place is {obj}."
    ],
    "favorite_book": [
        "{subj} have {obj} as favorite book.",
        "{subj} has {obj} as favorite book.",
        "{subj}\u00e2\u0080\u0099s favorite book is {obj}."
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

# converts type to MLNI id, but always hardcoded so not super necessary
# 2-1-0 was wrong according to MNLI labels
labels2id = {"positive": "entailment", "neutral": "neutral", "negative": "contradiction"}


def dnli2mlni(
        instance: REInputFeatures,
        templates,
):
    mnli_instances = []
    relation = instance.relation
    # use first person template if appropriate - always first, otherwise use second template for third person
    if instance.subj.lower() == "i":
        template = templates[relation][0]
    else:
        template = templates[relation][1]
    mnli_instances.append(
        MNLIInputFeatures(
            premise=instance.context,
            hypothesis=f"{template.format(subj=instance.subj, obj=instance.obj)}",
            label=labels2id[instance.label],
        )
    )

    return mnli_instances


with open(args.input_file, "rt") as f:
    mnli_data = []
    for line in json.load(f):
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
                relation=line['triple2'][1],
                context=line['sentence1'],
                label=line['label']
            ),
            relation_templates
        )
        mnli_data.extend(mnli_instance)

with open(args.output_file, "wt") as f:
    # f.write(json.dumps(mnli_data))
    json.dump([data.__dict__ for data in mnli_data], f, indent=2)
