"""
LLM-Physics-style atomic-repair SFT data generator.

Builds a symbolic world (six relation families) over a synthetic entity
inventory, deterministically realizes each item into one of three surface
forms (naturalized text / fact-table / compact-symbolic), and emits one of
five cell types: H-Aug, H-Abl, H-Cor, K-Cor, Clean.

No API calls. No GPU. Pure-Python, seedable, oracle-verifiable.

Outputs:
  <out_dir>/repair_raw_train.jsonl
  <out_dir>/repair_raw_eval.jsonl
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 1. Symbolic world: synthetic entity inventory + per-family graphs
# ---------------------------------------------------------------------------
# We use fake-but-plausible-sounding entity strings so that prior knowledge of
# real-world facts cannot leak in. Every link is fully recorded in symbolic
# form so we can verify gold answers programmatically.

# Country / nationality pool (used by several families).
COUNTRY_INFO = [
    # (country, demonym/nationality, currency)
    ("Lydoria",    "Lydorian",    "lydar"),
    ("Norlandia",  "Norlandian",  "noric"),
    ("Veltria",    "Veltrian",    "veltrek"),
    ("Korbenia",   "Korbenian",   "korbi"),
    ("Mavinia",    "Mavinian",    "marin"),
    ("Tarsisia",   "Tarsisian",   "tarsis"),
    ("Pendralia",  "Pendralian",  "pendra"),
    ("Solenia",    "Solenian",    "solen"),
    ("Brendara",   "Brendaran",   "brend"),
    ("Quintaria",  "Quintarian",  "quinta"),
    ("Olmaris",    "Olmarisian",  "olmar"),
    ("Forenza",    "Forenzan",    "forenc"),
    ("Drasivia",   "Drasivian",   "drasi"),
    ("Hesperin",   "Hesperinian", "hespera"),
    ("Caltoria",   "Caltorian",   "calto"),
]

# City -> deterministically assigned country index, by position.
CITIES = [
    "Velmar", "Trindale", "Korba", "Marinpoint", "Olstaad",
    "Pendral", "Solen Bay", "Brindar", "Quintas", "Olfast",
    "Forencia", "Drasov", "Hesper", "Caltor", "Lydor City",
    "Norhaven", "Veltra Bay", "Korbenburg", "Mavin Hill", "Tarsia",
    "Bremmer", "Quintholm", "Olmari", "Forenheim", "Drasivol",
    "Hespermill", "Calton Ridge", "Pendholt", "Solendel", "Lyterra",
]

BOOK_TITLES = [
    "Silver River", "The Iron Falcon", "Beneath the Salt Sea",
    "Whisper of Brass", "The Crooked Mile", "Halls of Vellum",
    "The Last Cartographer", "Eleven Mornings", "The Bone Garden",
    "Tide and Tower", "The Pale Calendar", "Quill and Coin",
    "The Glass Apiary", "Atlas of Storms", "Verses for Strangers",
    "The Hollow Concord", "Lantern Tongue", "Threshold Birds",
    "The Long Margin", "Bell Without a City", "Salt and Compass",
    "Years of the Ferry", "The Mended Sextant", "Smoke for Breakfast",
    "An Inventory of Doors", "The Borrowed Astronomer",
    "Carpenters of Silence", "Riverbook", "The Quiet Concordance",
    "Twelve Departures",
]

AUTHOR_NAMES = [
    "Maria Voss", "Anton Hale", "Petra Olenko", "Jules Bremmer",
    "Idris Calder", "Helena Marsh", "Roman Felk", "Ines Tarvik",
    "Oskar Bremm", "Nadia Solven", "Lev Korbin", "Eira Pendral",
    "Otto Vellin", "Soraya Drask", "Mattias Olmari", "Theia Quill",
    "Brand Forenz", "Iva Drasivol", "Casimir Hesper", "Nia Caltor",
    "Alva Brindar", "Karl Lytera", "Mireille Norhaven", "Olin Veltra",
    "Sasha Korbenburg", "Veda Mavinhill", "Petros Tarsia",
    "Linnea Bremmer", "Yuval Quintholm", "Iskra Olmari",
]

COMPANY_NAMES = [
    "Hesperin Dynamics", "Marsh & Calder", "Olstaad Robotics",
    "Vellum Press", "Pendral Foundry", "Solen Avionics",
    "Brindar Optics", "Quintas Hydraulics", "Olfast Foods",
    "Forencia Steelworks", "Drasov Networks", "Hesper Glassworks",
    "Caltor Cement", "Lydor Logistics", "Norhaven Pharmaceuticals",
    "Veltra Composites", "Korbenburg Sensors", "Mavinhill Apparel",
    "Tarsia Components", "Bremmer Locomotive", "Quintholm Heat",
    "Olmari Acoustics", "Forenheim Beverage", "Drasivol Aerospace",
    "Hespermill Energy", "Calton Cooperative", "Pendholt Polymers",
    "Solendel Marine", "Lyterra Automation", "Trindale Foundry",
]

FOUNDER_NAMES = [
    "Helga Marsh", "Cyrus Vellum", "Mira Olstaad", "Bertil Pendral",
    "Wren Solen", "Jorah Brindar", "Avrim Quintas", "Lena Olfast",
    "Tobias Forencia", "Niels Drasov", "Pia Hesper", "Roan Caltor",
    "Inga Lydor", "Mikael Norhaven", "Hadi Veltra", "Greta Korbenburg",
    "Stenz Mavinhill", "Velma Tarsia", "Henrik Bremmer", "Onal Quintholm",
    "Saskia Olmari", "Reuben Forenheim", "Marsh Drasivol",
    "Kira Hespermill", "Brand Calton", "Lou Pendholt", "Vesper Solendel",
    "Anik Lyterra", "Calla Trindale", "Hugo Olstaad",
]

PRODUCT_NAMES = [
    "Sentra X1", "PolyGrid 7", "BlueFin 9", "AstroFork",
    "ClearStitch", "LumenPath", "OakSpan",  "RiverBeam",
    "SilverLoft", "PalePort", "WrenLine", "QuiltDrive",
    "TideFold", "PaperKiln", "DrumLine", "FlintWeave",
    "VerseEngine", "QuillSheath", "PolarHull", "HelioFan",
    "BrassRig", "CloudVein", "SaltMark", "GlasswingPro",
    "RestComb", "WhetGrade", "LeafKit", "AtlasArc",
    "ConstantTwo", "MoonStripe",
]

ARTWORK_NAMES = [
    "Three Sisters at Dusk", "The Salt Wedding", "Roof Over the Marsh",
    "Eleven Crows", "Soft Border", "The Long Ferry", "Cartographer's Hand",
    "Two Bridges", "The Pale Window", "Verses for a Cousin",
    "The Borrowed Garden", "Quiet Hammer", "The Last Veranda",
    "Sunday Inventory", "Coastline by Lamplight", "Brass and Apricot",
    "Anchor Choir", "Half-Built Lighthouse", "Iron Pew",
    "The Postal Sea", "Cherry Salt", "The Glass Kitchen",
    "Memorandum in Blue", "Five Reservoirs", "Bay of Open Doors",
    "Calendar of Wires", "Two Carts at Dawn", "The Tide Reader",
    "Threshold and Frost", "The Mended Net",
]

ARTIST_NAMES = [
    "Onal Vesna", "Pelle Brask", "Mara Olst", "Selim Drink",
    "Hilde Olmari", "Pers Volla", "Reva Korben", "Stas Quill",
    "Marit Hesper", "Olen Forenza", "Mira Caltor", "Reuben Pendholt",
    "Iva Solendel", "Lior Norhaven", "Cyra Brindar", "Otto Lyterra",
    "Frej Drasov", "Maeve Quintas", "Niko Olfast", "Anja Forencia",
    "Tess Hespermill", "Wim Bremmer", "Idra Olmari", "Karst Veltra",
    "Mago Korbenburg", "Pell Mavinhill", "Yara Tarsia",
    "Eero Pendral", "Vela Quintholm", "Helka Trindale",
]

SCIENTIST_NAMES = [
    "Dr. Voss Pendral", "Dr. Korbenburg Hale", "Dr. Iva Marsh",
    "Dr. Sten Vellum", "Dr. Mira Olstaad", "Dr. Bert Norhaven",
    "Dr. Anya Drasivol", "Dr. Pelle Olmari", "Dr. Ines Solendel",
    "Dr. Casimir Brindar", "Dr. Lena Quintas", "Dr. Otto Lyterra",
    "Dr. Mago Caltor", "Dr. Sela Trindale", "Dr. Mattis Hespermill",
    "Dr. Niels Forencia", "Dr. Helga Drasov", "Dr. Roan Veltra",
    "Dr. Tobias Mavinhill", "Dr. Wren Pendholt", "Dr. Eira Quintholm",
    "Dr. Reuben Tarsia", "Dr. Pia Olmari", "Dr. Marit Olstaad",
    "Dr. Frej Hesper", "Dr. Cyra Bremmer", "Dr. Onal Caltor",
    "Dr. Mira Forenheim", "Dr. Stas Lyterra", "Dr. Idra Norhaven",
]

DISCOVERIES = [
    "the kelvin-stripe effect", "the Olmari instability",
    "the dual-saline reaction", "the cold-bend invariant",
    "the marsh-light coefficient", "the bridge-current law",
    "the slow-glass anomaly", "the open-shell convergence",
    "the lattice-snow ratio", "the frostline conjecture",
    "the rotational drift law", "the soft-flux scaling",
    "the third-band rule", "the dim-photon coupling",
    "the orchard transform", "the linear-edge limit",
    "the pale-shift bound", "the unison-decay constant",
    "the candlewick gradient", "the salt-fold theorem",
    "the wide-plait condition", "the tertiary-quiet law",
    "the long-room scaling", "the half-mirror identity",
    "the open-letter coefficient", "the woven-spectrum rule",
    "the riverbank invariant", "the postal-channel result",
    "the upper-vault bound", "the cold-quartz limit",
]

# Discovery -> deterministic field-of-science assignment.
FIELDS = [
    "thermodynamics", "optics", "fluid dynamics", "materials science",
    "acoustics", "electromagnetism", "crystallography", "quantum optics",
    "biochemistry", "geophysics", "atmospheric physics",
    "computational physics", "astrophysics", "solid-state physics",
    "polymer science",
]


# ---------------------------------------------------------------------------
# 2. Surface templates per relation family.
#
# Conventions:
#   - Templates are written so that ENTITIES_USED is the variable substitutions.
#   - Each family lists train-only and eval-only template ids: eval may NOT
#     use train templates, so eval also tests template generalization.
# ---------------------------------------------------------------------------

# Each template entry:
#   "id":          unique template id (string)
#   "natural":     2-hop natural-language question template
#   "fact_table":  fact-table-style template (Facts / Question)
#   "compact":     compact-symbolic template (lightweight formal notation)
#   "hint_aug":    sentence injecting the bridge fact (for H-Aug repair-needed)
#   "ablate":      version of natural template with bridge entity masked
#   "wrong_inj":   sentence injecting a wrong bridge (for H-Cor)
#   "k_question":  1-hop K-Cor question template (uses only the inner edge,
#                  e.g. "What is the nationality of {author}?")
#   "k_wrong_inj": K-Cor wrong-claim injection sentence
#
# Some families share a structure but use different surface phrasings so train
# and eval templates *cannot* overlap word-for-word.

FAMILIES: dict[str, dict[str, Any]] = {
    "book_author_nationality": {
        "graph_pattern": "two_hop_bridge",
        "relations": ("written_by", "nationality"),
        "describe": ("the book {head}", "its author", "the author's nationality"),
        "templates_train": [
            {
                "id": "book_T1",
                "natural": "What is the nationality of the author who wrote {head}?",
                "fact_table": "Facts:\n- {head} is a novel.\nQuestion:\n- What is the nationality of the author?",
                "compact": "Q: nationality(author_of({head})) = ?",
                "hint_aug": "Note: {head} was written by {bridge}.",
                "ablate": "What is the nationality of the author of a novel that some call simply 'the book'?",
                "wrong_inj": "It is widely said that {head} was written by {wrong_bridge}.",
                "k_question": "What is the nationality of {bridge}?",
                "k_wrong_inj": "Some claim {bridge} is {wrong_tail}.",
            },
            {
                "id": "book_T2",
                "natural": "The novel {head} -- where is its author from?",
                "fact_table": "Facts:\n- {head} is published as a novel.\nQuestion:\n- The author's country of citizenship?",
                "compact": "Eval: author_country_of({head})",
                "hint_aug": "Background: {bridge} is listed as the author of {head}.",
                "ablate": "The novel in question -- where is its author from?",
                "wrong_inj": "Many readers assume {head} was authored by {wrong_bridge}.",
                "k_question": "Where is {bridge} a citizen of?",
                "k_wrong_inj": "Some readers think {bridge} is {wrong_tail}.",
            },
        ],
        "templates_eval": [
            {
                "id": "book_E1",
                "natural": "Of which country is the author of {head} a national?",
                "fact_table": "Facts:\n- The novel {head} has a single recorded author.\nQuestion:\n- Author's nationality?",
                "compact": "ask: nat(writer({head}))",
                "hint_aug": "Aside: the author of {head} is {bridge}.",
                "ablate": "Of which country is the author of this novel a national?",
                "wrong_inj": "It is sometimes stated that {head} was penned by {wrong_bridge}.",
                "k_question": "Of which country is {bridge} a national?",
                "k_wrong_inj": "Some sources state {bridge} is {wrong_tail}.",
            },
        ],
    },

    "city_country_currency": {
        "graph_pattern": "two_hop_bridge",
        "relations": ("located_in", "uses_currency"),
        "describe": ("the city {head}", "its country", "that country's currency"),
        "templates_train": [
            {
                "id": "city_T1",
                "natural": "What currency is used in the country where {head} is located?",
                "fact_table": "Facts:\n- {head} is a city.\nQuestion:\n- The currency of its country?",
                "compact": "Q: currency(country_of({head})) = ?",
                "hint_aug": "Note: {head} is located in {bridge}.",
                "ablate": "What currency is used in the country where this city is located?",
                "wrong_inj": "It is sometimes claimed that {head} is located in {wrong_bridge}.",
                "k_question": "What currency does {bridge} use?",
                "k_wrong_inj": "Some claim the currency of {bridge} is the {wrong_tail}.",
            },
            {
                "id": "city_T2",
                "natural": "The currency of the country in which {head} sits -- what is it?",
                "fact_table": "Facts:\n- {head} is a settled urban area.\nQuestion:\n- The currency of the country it lies in?",
                "compact": "Eval: currency_of(country({head}))",
                "hint_aug": "Background: {bridge} is recorded as the country of {head}.",
                "ablate": "The currency of the country in which this city sits -- what is it?",
                "wrong_inj": "Many travel guides say {head} sits within {wrong_bridge}.",
                "k_question": "Which currency does {bridge} use?",
                "k_wrong_inj": "Some travel guides state that {bridge} uses the {wrong_tail}.",
            },
        ],
        "templates_eval": [
            {
                "id": "city_E1",
                "natural": "Which currency would you use in the country containing {head}?",
                "fact_table": "Facts:\n- {head} is a populated place.\nQuestion:\n- Currency of its country?",
                "compact": "ask: cur(country({head}))",
                "hint_aug": "Aside: {head} belongs to {bridge}.",
                "ablate": "Which currency would you use in the country containing this city?",
                "wrong_inj": "Some sources place {head} inside {wrong_bridge}.",
                "k_question": "Which currency would you use in {bridge}?",
                "k_wrong_inj": "Some sources say {bridge}'s currency is the {wrong_tail}.",
            },
        ],
    },

    "company_founder_nationality": {
        "graph_pattern": "two_hop_bridge",
        "relations": ("founded_by", "nationality"),
        "describe": ("the company {head}", "its founder", "the founder's nationality"),
        "templates_train": [
            {
                "id": "company_T1",
                "natural": "What is the nationality of the founder of {head}?",
                "fact_table": "Facts:\n- {head} is a company.\nQuestion:\n- Nationality of its founder?",
                "compact": "Q: nationality(founder({head})) = ?",
                "hint_aug": "Note: {head} was founded by {bridge}.",
                "ablate": "What is the nationality of the founder of this company?",
                "wrong_inj": "It is widely reported that {head} was founded by {wrong_bridge}.",
                "k_question": "What is the nationality of {bridge}?",
                "k_wrong_inj": "Some reports state {bridge} is {wrong_tail}.",
            },
            {
                "id": "company_T2",
                "natural": "The person who founded {head} -- what country are they a citizen of?",
                "fact_table": "Facts:\n- {head} operates as a company.\nQuestion:\n- Citizenship of the founder?",
                "compact": "Eval: nat(founder_of({head}))",
                "hint_aug": "Background: records list {bridge} as the founder of {head}.",
                "ablate": "The person who founded this company -- what country are they a citizen of?",
                "wrong_inj": "It is sometimes reported that {head} was founded by {wrong_bridge}.",
                "k_question": "What country is {bridge} a citizen of?",
                "k_wrong_inj": "Some reports say {bridge} is {wrong_tail}.",
            },
        ],
        "templates_eval": [
            {
                "id": "company_E1",
                "natural": "What citizenship does the founder of {head} hold?",
                "fact_table": "Facts:\n- {head} is a registered company.\nQuestion:\n- The founder's citizenship?",
                "compact": "ask: cit(founder({head}))",
                "hint_aug": "Aside: the founder of {head} is {bridge}.",
                "ablate": "What citizenship does the founder of this company hold?",
                "wrong_inj": "It is occasionally claimed that {head} was founded by {wrong_bridge}.",
                "k_question": "What citizenship does {bridge} hold?",
                "k_wrong_inj": "Some claim {bridge} is {wrong_tail}.",
            },
        ],
    },

    "product_company_country": {
        "graph_pattern": "two_hop_bridge",
        "relations": ("made_by", "headquartered_in"),
        "describe": ("the product {head}", "its manufacturer", "the manufacturer's headquarters country"),
        "templates_train": [
            {
                "id": "product_T1",
                "natural": "In which country is the company that makes {head} headquartered?",
                "fact_table": "Facts:\n- {head} is a product.\nQuestion:\n- HQ country of its maker?",
                "compact": "Q: hq_country(maker({head})) = ?",
                "hint_aug": "Note: {head} is manufactured by {bridge}.",
                "ablate": "In which country is the company that makes this product headquartered?",
                "wrong_inj": "It is sometimes said that {head} is manufactured by {wrong_bridge}.",
                "k_question": "In which country is {bridge} headquartered?",
                "k_wrong_inj": "Some say {bridge} is headquartered in {wrong_tail}.",
            },
            {
                "id": "product_T2",
                "natural": "Where is the maker of {head} based?",
                "fact_table": "Facts:\n- {head} is sold as a product.\nQuestion:\n- HQ country of the maker?",
                "compact": "Eval: country(hq(maker({head})))",
                "hint_aug": "Background: the manufacturer of {head} is {bridge}.",
                "ablate": "Where is the maker of this product based?",
                "wrong_inj": "Many say the maker of {head} is {wrong_bridge}.",
                "k_question": "Where is {bridge} based?",
                "k_wrong_inj": "Many say {bridge} is based in {wrong_tail}.",
            },
        ],
        "templates_eval": [
            {
                "id": "product_E1",
                "natural": "The company that manufactures {head} -- in which country is its head office?",
                "fact_table": "Facts:\n- {head} is a manufactured product.\nQuestion:\n- Head-office country of its manufacturer?",
                "compact": "ask: hq(maker({head}))",
                "hint_aug": "Aside: {bridge} is the company that makes {head}.",
                "ablate": "The company that manufactures this product -- in which country is its head office?",
                "wrong_inj": "It is reported in some places that {head} is made by {wrong_bridge}.",
                "k_question": "In which country is {bridge}'s head office?",
                "k_wrong_inj": "It is sometimes said {bridge}'s head office is in {wrong_tail}.",
            },
        ],
    },

    "artwork_artist_country": {
        "graph_pattern": "two_hop_bridge",
        "relations": ("created_by", "birth_country"),
        "describe": ("the artwork {head}", "its creator", "the creator's birth country"),
        "templates_train": [
            {
                "id": "artwork_T1",
                "natural": "In which country was the artist who created {head} born?",
                "fact_table": "Facts:\n- {head} is an artwork.\nQuestion:\n- Birth country of its creator?",
                "compact": "Q: birth_country(creator({head})) = ?",
                "hint_aug": "Note: {head} was created by {bridge}.",
                "ablate": "In which country was the artist who created this artwork born?",
                "wrong_inj": "It is sometimes said {head} was created by {wrong_bridge}.",
                "k_question": "In which country was {bridge} born?",
                "k_wrong_inj": "Some say {bridge} was born in {wrong_tail}.",
            },
            {
                "id": "artwork_T2",
                "natural": "The artist who painted {head} -- where were they born?",
                "fact_table": "Facts:\n- {head} is a painting.\nQuestion:\n- Where was its painter born?",
                "compact": "Eval: birth_country(painter({head}))",
                "hint_aug": "Background: the recorded painter of {head} is {bridge}.",
                "ablate": "The artist who painted this work -- where were they born?",
                "wrong_inj": "It is often suggested that {head} was painted by {wrong_bridge}.",
                "k_question": "Where was {bridge} born?",
                "k_wrong_inj": "It is often suggested {bridge} was born in {wrong_tail}.",
            },
        ],
        "templates_eval": [
            {
                "id": "artwork_E1",
                "natural": "Of which country is the artist who produced {head} a native?",
                "fact_table": "Facts:\n- {head} is a recognized artwork.\nQuestion:\n- Native country of its artist?",
                "compact": "ask: born_in(artist({head}))",
                "hint_aug": "Aside: the artist of {head} is {bridge}.",
                "ablate": "Of which country is the artist who produced this artwork a native?",
                "wrong_inj": "Several reviewers state that {head} was made by {wrong_bridge}.",
                "k_question": "Of which country is {bridge} a native?",
                "k_wrong_inj": "Several reviewers state {bridge} is a native of {wrong_tail}.",
            },
        ],
    },

    "scientist_discovery_field": {
        "graph_pattern": "two_hop_bridge",
        "relations": ("known_for", "field"),
        "describe": ("the scientist {head}", "their key discovery", "the field of that discovery"),
        "templates_train": [
            {
                "id": "scientist_T1",
                "natural": "To which scientific field does the discovery {head} is known for belong?",
                "fact_table": "Facts:\n- {head} is a working scientist.\nQuestion:\n- Field of the discovery they are known for?",
                "compact": "Q: field(known_for({head})) = ?",
                "hint_aug": "Note: {head} is known for {bridge}.",
                "ablate": "To which scientific field does this scientist's key discovery belong?",
                "wrong_inj": "It is often said that {head} is known for {wrong_bridge}.",
                "k_question": "Which scientific field does {bridge} belong to?",
                "k_wrong_inj": "Some say {bridge} belongs to {wrong_tail}.",
            },
            {
                "id": "scientist_T2",
                "natural": "The discovery that made {head} famous -- which scientific field is it part of?",
                "fact_table": "Facts:\n- {head} is a researcher.\nQuestion:\n- The field of their famous discovery?",
                "compact": "Eval: field(famous_for({head}))",
                "hint_aug": "Background: {head} is credited with {bridge}.",
                "ablate": "The discovery that made this researcher famous -- which scientific field is it part of?",
                "wrong_inj": "Many texts state {head} is credited with {wrong_bridge}.",
                "k_question": "Which scientific field does {bridge} fall under?",
                "k_wrong_inj": "Many texts say {bridge} falls under {wrong_tail}.",
            },
        ],
        "templates_eval": [
            {
                "id": "scientist_E1",
                "natural": "What is the scientific field of the discovery for which {head} is most cited?",
                "fact_table": "Facts:\n- {head} is a cited scientist.\nQuestion:\n- Field of their most-cited discovery?",
                "compact": "ask: field(top_work({head}))",
                "hint_aug": "Aside: {head} is most cited for {bridge}.",
                "ablate": "What is the scientific field of this scientist's most-cited discovery?",
                "wrong_inj": "Several biographies claim {head} is most cited for {wrong_bridge}.",
                "k_question": "What is the scientific field of {bridge}?",
                "k_wrong_inj": "Several biographies claim {bridge} is part of {wrong_tail}.",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# 3. Per-family graph builder.
#
# We build one symbolic graph per family by:
#   - picking N head entities for the family,
#   - assigning each head to a bridge entity (round-robin from the head pool),
#   - assigning each bridge entity to a tail entity (round-robin),
# so every edge is recorded as (head, rel, bridge) and (bridge, rel, tail).
# Held-out split: ~80% heads go to train, the rest to eval.
# ---------------------------------------------------------------------------

def _zip_round_robin(heads: list[str], bridges: list[str], tails: list[str]):
    """Deterministic mapping of head -> bridge -> tail.

    Each head gets a bridge by index (mod), each bridge gets a tail by index
    (mod). We later pick "wrong bridges" / "wrong tails" from the *same*
    family's bridge / tail pools but never equal to the gold value.
    """
    graph: list[dict[str, str]] = []
    for i, h in enumerate(heads):
        b = bridges[i % len(bridges)]
        t = tails[i % len(tails)]
        graph.append({"head": h, "bridge": b, "tail": t})
    return graph


def build_family_graph(name: str) -> dict[str, Any]:
    """Return symbolic graph + tail-string mapper for a family."""
    if name == "book_author_nationality":
        heads, bridges = BOOK_TITLES, AUTHOR_NAMES
        tails = [c[1] for c in COUNTRY_INFO]  # nationalities
        relations = ("written_by", "nationality")
        tail_format = "{bridge} is {tail}."
    elif name == "city_country_currency":
        heads = CITIES
        bridges = [c[0] for c in COUNTRY_INFO]  # countries
        tails   = [c[2] for c in COUNTRY_INFO]  # currencies
        relations = ("located_in", "uses_currency")
        tail_format = "{bridge} uses the {tail}."
    elif name == "company_founder_nationality":
        heads, bridges = COMPANY_NAMES, FOUNDER_NAMES
        tails = [c[1] for c in COUNTRY_INFO]
        relations = ("founded_by", "nationality")
        tail_format = "{bridge} is {tail}."
    elif name == "product_company_country":
        heads, bridges = PRODUCT_NAMES, COMPANY_NAMES
        tails = [c[0] for c in COUNTRY_INFO]
        relations = ("made_by", "headquartered_in")
        tail_format = "{bridge} is headquartered in {tail}."
    elif name == "artwork_artist_country":
        heads, bridges = ARTWORK_NAMES, ARTIST_NAMES
        tails = [c[0] for c in COUNTRY_INFO]
        relations = ("created_by", "birth_country")
        tail_format = "{bridge} was born in {tail}."
    elif name == "scientist_discovery_field":
        heads, bridges, tails = SCIENTIST_NAMES, DISCOVERIES, FIELDS
        relations = ("known_for", "field")
        tail_format = "{bridge} belongs to {tail}."
    else:
        raise ValueError(f"unknown family {name}")

    edges = _zip_round_robin(heads, bridges, tails)
    return {
        "name": name,
        "heads": heads,
        "bridges": bridges,
        "tails": tails,
        "relations": relations,
        "tail_sentence_format": tail_format,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# 4. Oracle sentence rendering.
#
# We always realize the same three oracle facts:
#   1. head <rel1> bridge        (e.g. "Silver River was written by Maria Voss.")
#   2. bridge <rel2> tail        (e.g. "Maria Voss is Lydorian.")
#   3. wrong_bridge <rel2> wrong_tail   (for Corrupt cells, used to show why
#                                       the planted wrong answer would be
#                                       consistent if the wrong bridge were true)
# ---------------------------------------------------------------------------

def first_fact_sentence(family: str, head: str, bridge: str) -> str:
    if family == "book_author_nationality":
        return f"{head} was written by {bridge}."
    if family == "city_country_currency":
        return f"{head} is located in {bridge}."
    if family == "company_founder_nationality":
        return f"{head} was founded by {bridge}."
    if family == "product_company_country":
        return f"{head} is made by {bridge}."
    if family == "artwork_artist_country":
        return f"{head} was created by {bridge}."
    if family == "scientist_discovery_field":
        return f"{head} is known for {bridge}."
    raise ValueError(family)


def second_fact_sentence(family: str, bridge: str, tail: str) -> str:
    if family == "book_author_nationality":
        return f"{bridge} is {tail}."
    if family == "city_country_currency":
        return f"{bridge} uses the {tail}."
    if family == "company_founder_nationality":
        return f"{bridge} is {tail}."
    if family == "product_company_country":
        return f"{bridge} is headquartered in {tail}."
    if family == "artwork_artist_country":
        return f"{bridge} was born in {tail}."
    if family == "scientist_discovery_field":
        return f"{bridge} belongs to {tail}."
    raise ValueError(family)


# ---------------------------------------------------------------------------
# 5. Item builders, one per cell.
#
# Each builder picks an edge, picks a template, picks a surface_type, picks a
# wrong_bridge / wrong_tail when needed, and emits one full record.
# ---------------------------------------------------------------------------

CELL_SPEC = {
    "H-Aug": dict(
        task_type="hybrid_bridge_retrieval",
        diagnosis="missing_bridge_fact",
        repair_skill="retrieve_bridge_fact",
        should_repair=True,
    ),
    "H-Abl": dict(
        task_type="hybrid_bridge_recovery",
        diagnosis="bridge_entity_missing",
        repair_skill="recover_bridge_entity",
        should_repair=True,
    ),
    "H-Cor": dict(
        task_type="hybrid_bridge_verification",
        diagnosis="wrong_bridge_contamination",
        repair_skill="bridge_source_verification",
        should_repair=True,
    ),
    "K-Cor": dict(
        task_type="knowledge_contradiction_check",
        diagnosis="wrong_factual_claim",
        repair_skill="contradiction_check",
        should_repair=True,
    ),
    "Clean": dict(
        task_type="no_repair_control",
        diagnosis="no_failure_detected",
        repair_skill="keep_answer",
        should_repair=False,
    ),
}

SURFACE_WEIGHTS = [("naturalized", 0.80), ("fact_table", 0.15), ("compact", 0.05)]


def pick_surface(rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for s, w in SURFACE_WEIGHTS:
        cum += w
        if r <= cum:
            return s
    return "naturalized"


def render_problem(template: dict[str, str], surface: str, head: str,
                   wrong_bridge: str | None = None) -> str:
    if surface == "naturalized":
        return template["natural"].format(head=head)
    if surface == "fact_table":
        return template["fact_table"].format(head=head)
    if surface == "compact":
        return template["compact"].format(head=head)
    raise ValueError(surface)


def render_problem_with_injection(template: dict[str, str], surface: str,
                                  head: str, inject_sentence: str) -> str:
    """Concatenate an injected sentence + the surface-formatted question.

    For fact-table we add the injected sentence as another fact bullet so the
    surface stays internally consistent.
    """
    if surface == "fact_table":
        # Split fact-table: put injected sentence as a fact bullet.
        ft = template["fact_table"].format(head=head)
        # ft starts with "Facts:\n- ...\nQuestion:\n- ..."
        try:
            facts_block, q_block = ft.split("Question:")
        except ValueError:
            # Fallback if format ever shifts
            return f"{inject_sentence}\n{ft}"
        facts_block = facts_block.rstrip() + f"\n- {inject_sentence}\n"
        return facts_block + "Question:" + q_block
    if surface == "naturalized":
        return f"{inject_sentence} " + template["natural"].format(head=head)
    if surface == "compact":
        # Compact form: keep the formal Q, but emit injected sentence as a
        # prelude line, so structure is preserved.
        return f"hint: {inject_sentence}\n" + template["compact"].format(head=head)
    raise ValueError(surface)


def render_ablate_problem(template: dict[str, str], surface: str, head: str) -> str:
    """For H-Abl, the bridge entity is removed from the question.

    We rely on the per-template `ablate` field, which already has the bridge
    entity stripped (e.g. 'this novel' instead of '{head}'). We additionally
    mask the head in compact form to make the bridge-recovery requirement
    explicit.
    """
    if surface == "naturalized":
        return template["ablate"]
    if surface == "fact_table":
        # Replace head with a [MASK]ed marker so the head cannot be looked up.
        ft = template["fact_table"].format(head="[MASK]")
        return ft
    if surface == "compact":
        return template["compact"].format(head="[MASK]")
    raise ValueError(surface)


def render_kcor_problem(template: dict[str, str], surface: str,
                        bridge: str, wrong_tail: str) -> str:
    """K-Cor: 1-hop question + planted wrong claim about that exact 1-hop edge."""
    wrong_inj = template["k_wrong_inj"].format(bridge=bridge, wrong_tail=wrong_tail)
    if surface == "naturalized":
        q = template["k_question"].format(bridge=bridge)
        return f"{wrong_inj} {q}"
    if surface == "fact_table":
        q = template["k_question"].format(bridge=bridge)
        return (
            "Facts:\n"
            f"- {wrong_inj}\n"
            "Question:\n"
            f"- {q}"
        )
    if surface == "compact":
        q = template["k_question"].format(bridge=bridge)
        return f"claim: {wrong_inj}\nQ: {q}"
    raise ValueError(surface)


def _pick_distinct(rng: random.Random, pool: list[str], banned: str) -> str:
    pick = rng.choice(pool)
    tries = 0
    while pick == banned and tries < 20:
        pick = rng.choice(pool)
        tries += 1
    return pick


def build_oracle_facts(family: str, edge: dict[str, str],
                       extra: dict[str, str] | None = None) -> tuple[list[str], list[list[str]]]:
    """Always include the two gold facts; optionally extra facts (e.g. the
    wrong bridge's actual tail, for H-Cor / K-Cor verifiability)."""
    head, bridge, tail = edge["head"], edge["bridge"], edge["tail"]
    rel1, rel2 = FAMILIES[family]["relations"]
    oracle = [
        first_fact_sentence(family, head, bridge),
        second_fact_sentence(family, bridge, tail),
    ]
    sym = [
        [head, rel1, bridge],
        [bridge, rel2, tail],
    ]
    if extra:
        if "wrong_bridge" in extra:
            wb, wt = extra["wrong_bridge"], extra["wrong_tail"]
            oracle.append(second_fact_sentence(family, wb, wt))
            sym.append([wb, rel2, wt])
    return oracle, sym


def build_record(*, cell: str, rng: random.Random, family_name: str,
                 family_graph: dict[str, Any], template: dict[str, str],
                 edge: dict[str, str], split: str, entity_split: str,
                 idx_in_cell_split: int) -> dict[str, Any]:
    """Return one fully-formed JSONL record."""
    spec = CELL_SPEC[cell]
    head, bridge, tail = edge["head"], edge["bridge"], edge["tail"]
    surface = pick_surface(rng)

    common = dict(
        id=f"{cell}_{split}_{idx_in_cell_split:06d}",
        cell=cell,
        surface_type=surface,
        relation_family=family_name,
        graph_pattern=family_graph.get("graph_pattern", "two_hop_bridge"),
        task_type=spec["task_type"],
        diagnosis=spec["diagnosis"],
        repair_skill=spec["repair_skill"],
        should_repair=spec["should_repair"],
        split=split,
        template_id=template["id"],
        entity_split=entity_split,
    )

    if cell == "Clean":
        problem = render_problem(template, surface, head)
        gold = tail
        oracle, sym = build_oracle_facts(family_name, edge)
        record = dict(
            **common,
            problem=problem,
            tentative_answer=gold,
            gold_answer=gold,
            planted_wrong_answer=None,
            repair_trace=(
                "The tentative answer is already consistent with the known facts. "
                f"Specifically, {oracle[0]} {oracle[1]} "
                f"So the tentative answer '{gold}' matches the chain. No repair is needed."
            ),
            final_answer=gold,
            oracle_facts=oracle,
            symbolic_facts=sym,
        )
        return record

    if cell == "H-Aug":
        # Bridge fact missing -> model retrieves it.
        problem = render_problem(template, surface, head)
        gold = tail
        # Tentative answer: some wrong but on-type guess (a different tail
        # from the same pool). Marks "unsupported" by design.
        wrong_tail = _pick_distinct(rng, family_graph["tails"], gold)
        oracle, sym = build_oracle_facts(family_name, edge)
        trace = (
            "The tentative answer is unsupported because the bridge fact is "
            "missing from the problem. "
            f"Retrieve the bridge fact: {oracle[0]} "
            f"Then use the second fact: {oracle[1]} "
            f"Therefore the final answer is {gold}."
        )
        record = dict(
            **common,
            problem=problem,
            tentative_answer=wrong_tail,
            gold_answer=gold,
            planted_wrong_answer=None,
            repair_trace=trace,
            final_answer=gold,
            oracle_facts=oracle,
            symbolic_facts=sym,
        )
        return record

    if cell == "H-Abl":
        # Bridge entity masked out -> model recovers from oracle. We prefix
        # an opaque case id derived from the head + split index so that
        # different ablated items produce distinct problem strings without
        # leaking the head identity. The id has no semantic content; the
        # model still has to consult oracle_facts to answer.
        case_id = f"Case #{abs(hash(head)) % 10000:04d}-{idx_in_cell_split % 100:02d}"
        body = render_ablate_problem(template, surface, head)
        problem = f"{case_id}: {body}"
        gold = tail
        wrong_tail = _pick_distinct(rng, family_graph["tails"], gold)
        oracle, sym = build_oracle_facts(family_name, edge)
        trace = (
            "The problem hides or removes the bridge entity. "
            f"Recover the bridge entity from oracle facts: {oracle[0]} "
            f"Then use the recovered entity to answer: {oracle[1]} "
            f"Therefore the final answer is {gold}."
        )
        record = dict(
            **common,
            problem=problem,
            tentative_answer=wrong_tail,
            gold_answer=gold,
            planted_wrong_answer=None,
            repair_trace=trace,
            final_answer=gold,
            oracle_facts=oracle,
            symbolic_facts=sym,
        )
        return record

    if cell == "H-Cor":
        # Wrong-bridge planted in the problem. Tentative answer accepts it.
        wrong_bridge = _pick_distinct(rng, family_graph["bridges"], bridge)
        # Find the tail that the wrong bridge actually maps to (gold table
        # lookup, not random) so the planted wrong answer is type-consistent
        # and oracle-verifiable.
        wb_edge = next((e for e in family_graph["edges"] if e["bridge"] == wrong_bridge), None)
        wrong_tail = wb_edge["tail"] if wb_edge else _pick_distinct(rng, family_graph["tails"], tail)
        if wrong_tail == tail:
            wrong_tail = _pick_distinct(rng, family_graph["tails"], tail)
        wrong_inj = template["wrong_inj"].format(head=head, wrong_bridge=wrong_bridge)
        problem = render_problem_with_injection(template, surface, head, wrong_inj)
        gold = tail
        oracle, sym = build_oracle_facts(
            family_name, edge,
            extra=dict(wrong_bridge=wrong_bridge, wrong_tail=wrong_tail),
        )
        trace = (
            f"The tentative answer follows the planted bridge '{wrong_bridge}'. "
            f"Verify the bridge claim against the known facts. "
            f"The planted bridge is false: {oracle[0]} (not {wrong_bridge}). "
            f"Use the correct bridge fact: {oracle[1]} "
            f"Therefore the final answer is {gold}, not {wrong_tail}."
        )
        record = dict(
            **common,
            problem=problem,
            tentative_answer=wrong_tail,
            gold_answer=gold,
            planted_wrong_answer=wrong_tail,
            repair_trace=trace,
            final_answer=gold,
            oracle_facts=oracle,
            symbolic_facts=sym,
        )
        return record

    if cell == "K-Cor":
        # 1-hop question; wrong claim is about the bridge -> tail edge.
        wrong_tail = _pick_distinct(rng, family_graph["tails"], tail)
        problem = render_kcor_problem(template, surface, bridge, wrong_tail)
        gold = tail
        # Oracle: include the gold inner fact AND the (true) edge so the
        # validator can check contradiction.
        oracle = [second_fact_sentence(family_name, bridge, tail)]
        rel2 = FAMILIES[family_name]["relations"][1]
        sym = [[bridge, rel2, tail]]
        trace = (
            f"Check the stated claim against the known fact. "
            f"The stated claim says '{bridge}' is '{wrong_tail}', but the known fact is: "
            f"{oracle[0]} "
            f"The stated claim is false. Use the correct fact. "
            f"Therefore the final answer is {gold}."
        )
        record = dict(
            **common,
            problem=problem,
            tentative_answer=wrong_tail,
            gold_answer=gold,
            planted_wrong_answer=wrong_tail,
            repair_trace=trace,
            final_answer=gold,
            oracle_facts=oracle,
            symbolic_facts=sym,
        )
        return record

    raise ValueError(cell)


# ---------------------------------------------------------------------------
# 6. Top-level generator.
# ---------------------------------------------------------------------------

# How many items per (cell, split).
TARGET_COUNTS = {
    "train": {"H-Aug": 400, "H-Abl": 400, "H-Cor": 500, "K-Cor": 400, "Clean": 400},
    "eval":  {"H-Aug": 100, "H-Abl": 100, "H-Cor": 150, "K-Cor": 100, "Clean": 100},
}

FAMILY_NAMES = list(FAMILIES.keys())  # 6 families


def split_edges_for_family(graph: dict[str, Any], split_seed: int):
    """Return (train_edges, eval_edges) such that BOTH head pools AND bridge
    pools are disjoint across splits.

    H-* cells expose the head ('Silver River'); K-Cor exposes the bridge
    ('Maria Voss'). To guarantee both `symbolic_facts[0][0]` checks pass, we
    independently hold out ~25% of bridges, then keep only edges whose bridge
    lies in the appropriate side, and additionally hold out ~25% of *heads*
    on top so head pools also don't leak.
    """
    rng = random.Random(split_seed)
    bridges = list(set(e["bridge"] for e in graph["edges"]))
    bridges.sort()  # determinism
    rng.shuffle(bridges)
    n_eval_b = max(2, int(round(0.25 * len(bridges))))
    eval_bridges = set(bridges[:n_eval_b])
    train_bridges = set(bridges[n_eval_b:])

    heads = list(set(e["head"] for e in graph["edges"]))
    heads.sort()
    rng.shuffle(heads)
    n_eval_h = max(2, int(round(0.25 * len(heads))))
    eval_heads = set(heads[:n_eval_h])
    train_heads = set(heads[n_eval_h:])

    train_edges = [e for e in graph["edges"]
                   if e["bridge"] in train_bridges and e["head"] in train_heads]
    eval_edges  = [e for e in graph["edges"]
                   if e["bridge"] in eval_bridges  and e["head"] in eval_heads]

    if not train_edges or not eval_edges:
        # Degenerate split — fall back to a head-only split (still gives a
        # reasonable head split for H-*, but K-Cor bridges may overlap; this
        # branch is a safety net, not the expected path).
        n_eval = max(4, int(round(0.25 * len(graph["edges"]))))
        eval_edges = list(graph["edges"])[:n_eval]
        train_edges = list(graph["edges"])[n_eval:]
    return train_edges, eval_edges


def generate(out_dir: Path, seed: int):
    rng = random.Random(seed)
    family_graphs = {fname: build_family_graph(fname) for fname in FAMILY_NAMES}

    # Per-family held-out entity split (deterministic in seed).
    family_splits = {
        fname: split_edges_for_family(family_graphs[fname], split_seed=seed + i)
        for i, fname in enumerate(FAMILY_NAMES)
    }

    train_records, eval_records = [], []

    # Per-cell idx so ids are dense and unique.
    cell_idx = {"train": {c: 0 for c in TARGET_COUNTS["train"]},
                "eval":  {c: 0 for c in TARGET_COUNTS["eval"]}}

    # Enforce uniqueness on (problem, tentative_answer) pairs. Different
    # cells can legitimately share a `problem` (H-Aug and Clean differ only
    # in `tentative_answer`), but the *pair* must be unique so the validator
    # has a clean diagnostic signal.
    used_pairs: set[tuple[str, str]] = set()

    def emit(cell: str, split: str):
        # Pick a family (uniform).
        fname = FAMILY_NAMES[cell_idx[split][cell] % len(FAMILY_NAMES)]
        fgraph = family_graphs[fname]
        train_edges, eval_edges = family_splits[fname]
        edges = train_edges if split == "train" else eval_edges
        if not edges:
            raise RuntimeError(f"family {fname} has no {split} edges")
        edge = edges[cell_idx[split][cell] % len(edges)]

        # Template pool depends on split: eval uses held-out templates only.
        tpool = FAMILIES[fname]["templates_train"] if split == "train" else FAMILIES[fname]["templates_eval"]
        template = tpool[cell_idx[split][cell] % len(tpool)]

        # Retry loop to dodge duplicates: if we hit one, perturb the template
        # / edge index until we get a fresh problem.
        for attempt in range(80):
            local_template = tpool[(cell_idx[split][cell] + attempt) % len(tpool)]
            local_edge = edges[(cell_idx[split][cell] + attempt) % len(edges)]
            rec = build_record(
                cell=cell, rng=rng, family_name=fname, family_graph=fgraph,
                template=local_template, edge=local_edge, split=split,
                entity_split=split, idx_in_cell_split=cell_idx[split][cell],
            )
            key = (rec["problem"], rec["tentative_answer"])
            if key not in used_pairs:
                used_pairs.add(key)
                cell_idx[split][cell] += 1
                return rec
        # If we somehow could not get a fresh pair, force-burn the slot:
        # add a small surface tweak (suffix) so it doesn't crash.
        rec["problem"] = rec["problem"] + f"  // variant#{cell_idx[split][cell]}"
        used_pairs.add((rec["problem"], rec["tentative_answer"]))
        cell_idx[split][cell] += 1
        return rec

    for split in ("train", "eval"):
        targets = TARGET_COUNTS[split]
        # Interleave cells to avoid template streaks. Track planned slots
        # explicitly so cells with smaller targets drop out of the cycle
        # once their quota is reached.
        cells_cycle: list[str] = []
        planned = {c: 0 for c in targets}
        max_per_cell = max(targets.values())
        for i in range(max_per_cell):
            for c in ("H-Aug", "H-Abl", "H-Cor", "K-Cor", "Clean"):
                if planned[c] < targets[c]:
                    cells_cycle.append(c)
                    planned[c] += 1
        out_list = train_records if split == "train" else eval_records
        for c in cells_cycle:
            out_list.append(emit(c, split))

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "repair_raw_train.jsonl"
    eval_path  = out_dir / "repair_raw_eval.jsonl"
    with train_path.open("w") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with eval_path.open("w") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(train_records):5d} train records -> {train_path}")
    print(f"wrote {len(eval_records):5d} eval  records -> {eval_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.out_dir, args.seed)


if __name__ == "__main__":
    main()
