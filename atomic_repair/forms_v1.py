"""Form banks for atomic-repair v1 (known-component repair setting).

v1's generalization axis is NOT entity-OOD (that was v0). In v1 every entity /
atomic fact is seen during training; what is held out is the *form*: the wording
of the question, of the corruption sentence (wrong-bridge / wrong-claim), of the
fact-recall question, and of the repair trace. This module is the single source
of truth for those forms, so the generator and the validator agree on exactly
which form ids belong to train vs eval.

Every form entry carries a stable `form_id` and a `form_split in {train, eval}`.
The validator asserts:
  - question / corruption / trace form-id sets are disjoint between train & eval,
  - fact-question form-id sets are disjoint between train & eval,
while the underlying entities / facts are deliberately shared.

Conventions
-----------
- Question templates use `{head}` (the 2-hop head, e.g. a book title).
- Corruption phrasings use `{head}` and `{wrong_bridge}` (H-Cor) or `{bridge}`
  and `{wrong_tail}` (K-Cor).
- Fact-question phrasings use `{head}` (first hop) or `{bridge}` (second hop).
- Trace templates are `str.format`-ed by the generator with named fields
  (fact1, fact2, gold, wrong_tail, wrong_bridge) depending on the cell.

Family / relation names mirror generate_repair_data.build_family_graph exactly.
"""
from __future__ import annotations

from typing import Any

FAMILY_NAMES = [
    "book_author_nationality",
    "city_country_currency",
    "company_founder_nationality",
    "product_company_country",
    "artwork_artist_country",
    "scientist_discovery_field",
]

# First-hop / second-hop relation per family (matches build_family_graph).
FAMILY_RELATIONS: dict[str, tuple[str, str]] = {
    "book_author_nationality": ("written_by", "nationality"),
    "city_country_currency": ("located_in", "uses_currency"),
    "company_founder_nationality": ("founded_by", "nationality"),
    "product_company_country": ("made_by", "headquartered_in"),
    "artwork_artist_country": ("created_by", "birth_country"),
    "scientist_discovery_field": ("known_for", "field"),
}


# --------------------------------------------------------------------------- #
# 1. Repair question templates (2-hop). Per family: >=3 train + >=3 eval.
#
# Each entry is a dict with:
#   id        : unique form id (carries the split implicitly via _train/_eval)
#   natural   : naturalized 2-hop question, uses {head}
#   fact_table: "Facts: ... / Question: ..." surface, uses {head}
#   compact   : compact-symbolic surface, uses {head}
#   ablate    : H-Abl surface with the head/bridge identity removed
#
# TRAIN templates and EVAL templates share NO wording.
# --------------------------------------------------------------------------- #

QUESTION_TEMPLATES: dict[str, dict[str, list[dict[str, str]]]] = {
    "book_author_nationality": {
        "train": [
            {
                "id": "book_qT1",
                "natural": "What is the nationality of the author who wrote {head}?",
                "fact_table": "Facts:\n- {head} is a novel.\nQuestion:\n- What is the nationality of the author?",
                "compact": "Q: nationality(author_of({head})) = ?",
                "ablate": "What is the nationality of the author of a novel that some call simply 'the book'?",
            },
            {
                "id": "book_qT2",
                "natural": "The novel {head} -- where is its author from?",
                "fact_table": "Facts:\n- {head} is published as a novel.\nQuestion:\n- The author's country of citizenship?",
                "compact": "Eval: author_country_of({head})",
                "ablate": "The novel in question -- where is its author from?",
            },
            {
                "id": "book_qT3",
                "natural": "Tell me which country the writer of {head} comes from.",
                "fact_table": "Facts:\n- {head} is a book.\nQuestion:\n- Country the writer comes from?",
                "compact": "compute nationality <- writer({head})",
                "ablate": "Tell me which country the writer of this book comes from.",
            },
        ],
        "eval": [
            {
                "id": "book_qE1",
                "natural": "Of which country is the author of {head} a national?",
                "fact_table": "Facts:\n- The novel {head} has a single recorded author.\nQuestion:\n- Author's nationality?",
                "compact": "ask: nat(writer({head}))",
                "ablate": "Of which country is the author of this novel a national?",
            },
            {
                "id": "book_qE2",
                "natural": "{head} was penned by someone -- what nationality do they hold?",
                "fact_table": "Facts:\n- {head} has one author of record.\nQuestion:\n- That author's nationality?",
                "compact": "resolve: citizenship( author({head}) )",
                "ablate": "This work was penned by someone -- what nationality do they hold?",
            },
            {
                "id": "book_qE3",
                "natural": "Identify the home country of the person who authored {head}.",
                "fact_table": "Facts:\n- {head} counts as a literary work.\nQuestion:\n- Home country of its author?",
                "compact": "lookup home_country := author({head})",
                "ablate": "Identify the home country of the person who authored this literary work.",
            },
        ],
    },
    "city_country_currency": {
        "train": [
            {
                "id": "city_qT1",
                "natural": "What currency is used in the country where {head} is located?",
                "fact_table": "Facts:\n- {head} is a city.\nQuestion:\n- The currency of its country?",
                "compact": "Q: currency(country_of({head})) = ?",
                "ablate": "What currency is used in the country where this city is located?",
            },
            {
                "id": "city_qT2",
                "natural": "The currency of the country in which {head} sits -- what is it?",
                "fact_table": "Facts:\n- {head} is a settled urban area.\nQuestion:\n- The currency of the country it lies in?",
                "compact": "Eval: currency_of(country({head}))",
                "ablate": "The currency of the country in which this city sits -- what is it?",
            },
            {
                "id": "city_qT3",
                "natural": "If you travelled to {head}, what money would you spend there?",
                "fact_table": "Facts:\n- {head} is a town.\nQuestion:\n- Money spent in its country?",
                "compact": "compute currency <- country({head})",
                "ablate": "If you travelled to this town, what money would you spend there?",
            },
        ],
        "eval": [
            {
                "id": "city_qE1",
                "natural": "Which currency would you use in the country containing {head}?",
                "fact_table": "Facts:\n- {head} is a populated place.\nQuestion:\n- Currency of its country?",
                "compact": "ask: cur(country({head}))",
                "ablate": "Which currency would you use in the country containing this city?",
            },
            {
                "id": "city_qE2",
                "natural": "{head} belongs to some country -- name that country's currency.",
                "fact_table": "Facts:\n- {head} lies within one country.\nQuestion:\n- That country's currency?",
                "compact": "resolve: money( nation({head}) )",
                "ablate": "This place belongs to some country -- name that country's currency.",
            },
            {
                "id": "city_qE3",
                "natural": "State the legal tender of the nation that holds {head}.",
                "fact_table": "Facts:\n- {head} is a municipality.\nQuestion:\n- Legal tender of its nation?",
                "compact": "lookup tender := nation({head})",
                "ablate": "State the legal tender of the nation that holds this municipality.",
            },
        ],
    },
    "company_founder_nationality": {
        "train": [
            {
                "id": "company_qT1",
                "natural": "What is the nationality of the founder of {head}?",
                "fact_table": "Facts:\n- {head} is a company.\nQuestion:\n- Nationality of its founder?",
                "compact": "Q: nationality(founder({head})) = ?",
                "ablate": "What is the nationality of the founder of this company?",
            },
            {
                "id": "company_qT2",
                "natural": "The person who founded {head} -- what country are they a citizen of?",
                "fact_table": "Facts:\n- {head} operates as a company.\nQuestion:\n- Citizenship of the founder?",
                "compact": "Eval: nat(founder_of({head}))",
                "ablate": "The person who founded this company -- what country are they a citizen of?",
            },
            {
                "id": "company_qT3",
                "natural": "Which country does the founder of {head} hail from?",
                "fact_table": "Facts:\n- {head} is a firm.\nQuestion:\n- Country the founder hails from?",
                "compact": "compute nationality <- founder({head})",
                "ablate": "Which country does the founder of this firm hail from?",
            },
        ],
        "eval": [
            {
                "id": "company_qE1",
                "natural": "What citizenship does the founder of {head} hold?",
                "fact_table": "Facts:\n- {head} is a registered company.\nQuestion:\n- The founder's citizenship?",
                "compact": "ask: cit(founder({head}))",
                "ablate": "What citizenship does the founder of this company hold?",
            },
            {
                "id": "company_qE2",
                "natural": "{head} was established by someone -- what is their nationality?",
                "fact_table": "Facts:\n- {head} has a single founder of record.\nQuestion:\n- That founder's nationality?",
                "compact": "resolve: citizenship( establisher({head}) )",
                "ablate": "This company was established by someone -- what is their nationality?",
            },
            {
                "id": "company_qE3",
                "natural": "Name the home country of whoever started {head}.",
                "fact_table": "Facts:\n- {head} is an enterprise.\nQuestion:\n- Home country of its starter?",
                "compact": "lookup home_country := founder({head})",
                "ablate": "Name the home country of whoever started this enterprise.",
            },
        ],
    },
    "product_company_country": {
        "train": [
            {
                "id": "product_qT1",
                "natural": "In which country is the company that makes {head} headquartered?",
                "fact_table": "Facts:\n- {head} is a product.\nQuestion:\n- HQ country of its maker?",
                "compact": "Q: hq_country(maker({head})) = ?",
                "ablate": "In which country is the company that makes this product headquartered?",
            },
            {
                "id": "product_qT2",
                "natural": "Where is the maker of {head} based?",
                "fact_table": "Facts:\n- {head} is sold as a product.\nQuestion:\n- HQ country of the maker?",
                "compact": "Eval: country(hq(maker({head})))",
                "ablate": "Where is the maker of this product based?",
            },
            {
                "id": "product_qT3",
                "natural": "Which country hosts the headquarters of the firm producing {head}?",
                "fact_table": "Facts:\n- {head} is an item for sale.\nQuestion:\n- Country hosting its producer's HQ?",
                "compact": "compute hq_country <- maker({head})",
                "ablate": "Which country hosts the headquarters of the firm producing this item?",
            },
        ],
        "eval": [
            {
                "id": "product_qE1",
                "natural": "The company that manufactures {head} -- in which country is its head office?",
                "fact_table": "Facts:\n- {head} is a manufactured product.\nQuestion:\n- Head-office country of its manufacturer?",
                "compact": "ask: hq(maker({head}))",
                "ablate": "The company that manufactures this product -- in which country is its head office?",
            },
            {
                "id": "product_qE2",
                "natural": "{head} is built by some company -- where is that company seated?",
                "fact_table": "Facts:\n- {head} has one manufacturer of record.\nQuestion:\n- Where that manufacturer is seated?",
                "compact": "resolve: seat( builder({head}) )",
                "ablate": "This product is built by some company -- where is that company seated?",
            },
            {
                "id": "product_qE3",
                "natural": "Identify the nation in which the producer of {head} keeps its base.",
                "fact_table": "Facts:\n- {head} is a commercial product.\nQuestion:\n- Nation of its producer's base?",
                "compact": "lookup base_nation := producer({head})",
                "ablate": "Identify the nation in which the producer of this product keeps its base.",
            },
        ],
    },
    "artwork_artist_country": {
        "train": [
            {
                "id": "artwork_qT1",
                "natural": "In which country was the artist who created {head} born?",
                "fact_table": "Facts:\n- {head} is an artwork.\nQuestion:\n- Birth country of its creator?",
                "compact": "Q: birth_country(creator({head})) = ?",
                "ablate": "In which country was the artist who created this artwork born?",
            },
            {
                "id": "artwork_qT2",
                "natural": "The artist who painted {head} -- where were they born?",
                "fact_table": "Facts:\n- {head} is a painting.\nQuestion:\n- Where was its painter born?",
                "compact": "Eval: birth_country(painter({head}))",
                "ablate": "The artist who painted this work -- where were they born?",
            },
            {
                "id": "artwork_qT3",
                "natural": "What is the birthplace country of the creator of {head}?",
                "fact_table": "Facts:\n- {head} is a work of art.\nQuestion:\n- Birthplace country of its creator?",
                "compact": "compute birth_country <- creator({head})",
                "ablate": "What is the birthplace country of the creator of this work of art?",
            },
        ],
        "eval": [
            {
                "id": "artwork_qE1",
                "natural": "Of which country is the artist who produced {head} a native?",
                "fact_table": "Facts:\n- {head} is a recognized artwork.\nQuestion:\n- Native country of its artist?",
                "compact": "ask: born_in(artist({head}))",
                "ablate": "Of which country is the artist who produced this artwork a native?",
            },
            {
                "id": "artwork_qE2",
                "natural": "{head} was made by an artist -- in what country did that artist come into the world?",
                "fact_table": "Facts:\n- {head} has one artist of record.\nQuestion:\n- Country that artist was born in?",
                "compact": "resolve: birthplace( maker_art({head}) )",
                "ablate": "This artwork was made by an artist -- in what country did that artist come into the world?",
            },
            {
                "id": "artwork_qE3",
                "natural": "Name the country of birth of whoever produced {head}.",
                "fact_table": "Facts:\n- {head} is a piece of art.\nQuestion:\n- Country of birth of its producer?",
                "compact": "lookup birth := producer_art({head})",
                "ablate": "Name the country of birth of whoever produced this piece of art.",
            },
        ],
    },
    "scientist_discovery_field": {
        "train": [
            {
                "id": "scientist_qT1",
                "natural": "To which scientific field does the discovery {head} is known for belong?",
                "fact_table": "Facts:\n- {head} is a working scientist.\nQuestion:\n- Field of the discovery they are known for?",
                "compact": "Q: field(known_for({head})) = ?",
                "ablate": "To which scientific field does this scientist's key discovery belong?",
            },
            {
                "id": "scientist_qT2",
                "natural": "The discovery that made {head} famous -- which scientific field is it part of?",
                "fact_table": "Facts:\n- {head} is a researcher.\nQuestion:\n- The field of their famous discovery?",
                "compact": "Eval: field(famous_for({head}))",
                "ablate": "The discovery that made this researcher famous -- which scientific field is it part of?",
            },
            {
                "id": "scientist_qT3",
                "natural": "Under which branch of science falls the discovery credited to {head}?",
                "fact_table": "Facts:\n- {head} is a scientist.\nQuestion:\n- Branch of science of their discovery?",
                "compact": "compute field <- known_for({head})",
                "ablate": "Under which branch of science falls the discovery credited to this scientist?",
            },
        ],
        "eval": [
            {
                "id": "scientist_qE1",
                "natural": "What is the scientific field of the discovery for which {head} is most cited?",
                "fact_table": "Facts:\n- {head} is a cited scientist.\nQuestion:\n- Field of their most-cited discovery?",
                "compact": "ask: field(top_work({head}))",
                "ablate": "What is the scientific field of this scientist's most-cited discovery?",
            },
            {
                "id": "scientist_qE2",
                "natural": "{head} is remembered for a discovery -- in what scientific area does it sit?",
                "fact_table": "Facts:\n- {head} has one notable discovery.\nQuestion:\n- Scientific area of that discovery?",
                "compact": "resolve: area( notable_work({head}) )",
                "ablate": "This scientist is remembered for a discovery -- in what scientific area does it sit?",
            },
            {
                "id": "scientist_qE3",
                "natural": "Classify the discovery associated with {head} into its scientific discipline.",
                "fact_table": "Facts:\n- {head} is an academic.\nQuestion:\n- Scientific discipline of their discovery?",
                "compact": "lookup discipline := work_of({head})",
                "ablate": "Classify the discovery associated with this academic into its scientific discipline.",
            },
        ],
    },
}


# --------------------------------------------------------------------------- #
# 2. Corruption-wording banks (decoupled from question templates).
#
# H-Cor injects a wrong BRIDGE: uses {head} + {wrong_bridge}.
# K-Cor injects a wrong CLAIM about the bridge->tail edge: uses {bridge} +
# {wrong_tail}. Both banks are per-family so the relation verb fits, and split
# train/eval so eval corruption wording is never seen at train.
# --------------------------------------------------------------------------- #

WRONG_BRIDGE_PHRASINGS: dict[str, dict[str, list[dict[str, str]]]] = {
    "book_author_nationality": {
        "train": [
            {"id": "book_wbT1", "text": "It is widely said that {head} was written by {wrong_bridge}."},
            {"id": "book_wbT2", "text": "Many readers assume {head} was authored by {wrong_bridge}."},
            {"id": "book_wbT3", "text": "A common claim is that {wrong_bridge} wrote {head}."},
        ],
        "eval": [
            {"id": "book_wbE1", "text": "Some catalogues misattribute {head} to {wrong_bridge}."},
            {"id": "book_wbE2", "text": "You may have heard that {head} came from the pen of {wrong_bridge}."},
            {"id": "book_wbE3", "text": "A persistent rumour credits {head} to {wrong_bridge}."},
        ],
    },
    "city_country_currency": {
        "train": [
            {"id": "city_wbT1", "text": "It is sometimes claimed that {head} is located in {wrong_bridge}."},
            {"id": "city_wbT2", "text": "Many travel guides say {head} sits within {wrong_bridge}."},
            {"id": "city_wbT3", "text": "A frequent assertion is that {head} lies in {wrong_bridge}."},
        ],
        "eval": [
            {"id": "city_wbE1", "text": "Some maps mistakenly place {head} inside {wrong_bridge}."},
            {"id": "city_wbE2", "text": "You will occasionally read that {head} belongs to {wrong_bridge}."},
            {"id": "city_wbE3", "text": "A widespread misconception puts {head} within {wrong_bridge}."},
        ],
    },
    "company_founder_nationality": {
        "train": [
            {"id": "company_wbT1", "text": "It is widely reported that {head} was founded by {wrong_bridge}."},
            {"id": "company_wbT2", "text": "It is sometimes reported that {head} was founded by {wrong_bridge}."},
            {"id": "company_wbT3", "text": "A common story is that {wrong_bridge} founded {head}."},
        ],
        "eval": [
            {"id": "company_wbE1", "text": "Some profiles wrongly credit the founding of {head} to {wrong_bridge}."},
            {"id": "company_wbE2", "text": "You may have read that {head} was set up by {wrong_bridge}."},
            {"id": "company_wbE3", "text": "A lingering myth names {wrong_bridge} as the founder of {head}."},
        ],
    },
    "product_company_country": {
        "train": [
            {"id": "product_wbT1", "text": "It is sometimes said that {head} is manufactured by {wrong_bridge}."},
            {"id": "product_wbT2", "text": "Many say the maker of {head} is {wrong_bridge}."},
            {"id": "product_wbT3", "text": "A common belief is that {wrong_bridge} makes {head}."},
        ],
        "eval": [
            {"id": "product_wbE1", "text": "Some listings wrongly say {head} is produced by {wrong_bridge}."},
            {"id": "product_wbE2", "text": "You might assume {head} comes from {wrong_bridge}."},
            {"id": "product_wbE3", "text": "A recurring error attributes {head} to {wrong_bridge}."},
        ],
    },
    "artwork_artist_country": {
        "train": [
            {"id": "artwork_wbT1", "text": "It is sometimes said {head} was created by {wrong_bridge}."},
            {"id": "artwork_wbT2", "text": "It is often suggested that {head} was painted by {wrong_bridge}."},
            {"id": "artwork_wbT3", "text": "A common attribution gives {head} to {wrong_bridge}."},
        ],
        "eval": [
            {"id": "artwork_wbE1", "text": "Some galleries misattribute {head} to {wrong_bridge}."},
            {"id": "artwork_wbE2", "text": "You may have seen {head} credited to {wrong_bridge}."},
            {"id": "artwork_wbE3", "text": "A stubborn rumour assigns {head} to {wrong_bridge}."},
        ],
    },
    "scientist_discovery_field": {
        "train": [
            {"id": "scientist_wbT1", "text": "It is often said that {head} is known for {wrong_bridge}."},
            {"id": "scientist_wbT2", "text": "Many texts state {head} is credited with {wrong_bridge}."},
            {"id": "scientist_wbT3", "text": "A common claim is that {head} discovered {wrong_bridge}."},
        ],
        "eval": [
            {"id": "scientist_wbE1", "text": "Some summaries wrongly tie {head} to {wrong_bridge}."},
            {"id": "scientist_wbE2", "text": "You may have read that {head}'s big result was {wrong_bridge}."},
            {"id": "scientist_wbE3", "text": "A persistent error links {head} with {wrong_bridge}."},
        ],
    },
}

# K-Cor: wrong claim about the bridge's tail. Uses {bridge} + {wrong_tail}.
WRONG_CLAIM_PHRASINGS: dict[str, dict[str, list[dict[str, str]]]] = {
    "book_author_nationality": {
        "train": [
            {"id": "book_wcT1", "text": "Some claim {bridge} is {wrong_tail}."},
            {"id": "book_wcT2", "text": "Some readers think {bridge} is {wrong_tail}."},
        ],
        "eval": [
            {"id": "book_wcE1", "text": "A common error labels {bridge} as {wrong_tail}."},
            {"id": "book_wcE2", "text": "You might be told that {bridge} is {wrong_tail}."},
        ],
    },
    "city_country_currency": {
        "train": [
            {"id": "city_wcT1", "text": "Some claim the currency of {bridge} is the {wrong_tail}."},
            {"id": "city_wcT2", "text": "Some travel guides state that {bridge} uses the {wrong_tail}."},
        ],
        "eval": [
            {"id": "city_wcE1", "text": "A frequent mistake says {bridge} runs on the {wrong_tail}."},
            {"id": "city_wcE2", "text": "You might read that {bridge}'s money is the {wrong_tail}."},
        ],
    },
    "company_founder_nationality": {
        "train": [
            {"id": "company_wcT1", "text": "Some reports state {bridge} is {wrong_tail}."},
            {"id": "company_wcT2", "text": "Some reports say {bridge} is {wrong_tail}."},
        ],
        "eval": [
            {"id": "company_wcE1", "text": "A common error calls {bridge} {wrong_tail}."},
            {"id": "company_wcE2", "text": "You may be told {bridge} is {wrong_tail}."},
        ],
    },
    "product_company_country": {
        "train": [
            {"id": "product_wcT1", "text": "Some say {bridge} is headquartered in {wrong_tail}."},
            {"id": "product_wcT2", "text": "Many say {bridge} is based in {wrong_tail}."},
        ],
        "eval": [
            {"id": "product_wcE1", "text": "A frequent mistake places {bridge}'s HQ in {wrong_tail}."},
            {"id": "product_wcE2", "text": "You might read that {bridge} sits in {wrong_tail}."},
        ],
    },
    "artwork_artist_country": {
        "train": [
            {"id": "artwork_wcT1", "text": "Some say {bridge} was born in {wrong_tail}."},
            {"id": "artwork_wcT2", "text": "It is often suggested {bridge} was born in {wrong_tail}."},
        ],
        "eval": [
            {"id": "artwork_wcE1", "text": "A common error says {bridge} hails from {wrong_tail}."},
            {"id": "artwork_wcE2", "text": "You may read that {bridge} was born in {wrong_tail}."},
        ],
    },
    "scientist_discovery_field": {
        "train": [
            {"id": "scientist_wcT1", "text": "Some say {bridge} belongs to {wrong_tail}."},
            {"id": "scientist_wcT2", "text": "Many texts say {bridge} falls under {wrong_tail}."},
        ],
        "eval": [
            {"id": "scientist_wcE1", "text": "A common error files {bridge} under {wrong_tail}."},
            {"id": "scientist_wcE2", "text": "You might read that {bridge} is part of {wrong_tail}."},
        ],
    },
}


# --------------------------------------------------------------------------- #
# 3. Fact-question paraphrases (one-hop), per relation. >=3 train + >=3 eval.
#
# first-hop relations resolve head -> bridge ("Who wrote {head}?" -> bridge)
# second-hop relations resolve bridge -> tail ("What nationality is {bridge}?")
#
# Placeholder is always {subj} (the generator passes head for first-hop,
# bridge for second-hop). Train/eval phrasings share no wording.
# --------------------------------------------------------------------------- #

FACT_QUESTION_PHRASINGS: dict[str, dict[str, list[dict[str, str]]]] = {
    # ---- first hop ----
    "written_by": {
        "train": [
            {"id": "q_written_by_T1", "text": "Who wrote {subj}?"},
            {"id": "q_written_by_T2", "text": "Who is the author of {subj}?"},
            {"id": "q_written_by_T3", "text": "Name the author of {subj}."},
        ],
        "eval": [
            {"id": "q_written_by_E1", "text": "{subj} was written by whom?"},
            {"id": "q_written_by_E2", "text": "The author of {subj} is who?"},
            {"id": "q_written_by_E3", "text": "Which person authored {subj}?"},
        ],
    },
    "located_in": {
        "train": [
            {"id": "q_located_in_T1", "text": "Which country is {subj} located in?"},
            {"id": "q_located_in_T2", "text": "In which country is {subj}?"},
            {"id": "q_located_in_T3", "text": "Name the country that contains {subj}."},
        ],
        "eval": [
            {"id": "q_located_in_E1", "text": "{subj} is situated in which country?"},
            {"id": "q_located_in_E2", "text": "The country of {subj} is what?"},
            {"id": "q_located_in_E3", "text": "What nation does {subj} sit in?"},
        ],
    },
    "founded_by": {
        "train": [
            {"id": "q_founded_by_T1", "text": "Who founded {subj}?"},
            {"id": "q_founded_by_T2", "text": "Who is the founder of {subj}?"},
            {"id": "q_founded_by_T3", "text": "Name the founder of {subj}."},
        ],
        "eval": [
            {"id": "q_founded_by_E1", "text": "{subj} was founded by whom?"},
            {"id": "q_founded_by_E2", "text": "The founder of {subj} is who?"},
            {"id": "q_founded_by_E3", "text": "Which person established {subj}?"},
        ],
    },
    "made_by": {
        "train": [
            {"id": "q_made_by_T1", "text": "Which company makes {subj}?"},
            {"id": "q_made_by_T2", "text": "Who manufactures {subj}?"},
            {"id": "q_made_by_T3", "text": "Name the maker of {subj}."},
        ],
        "eval": [
            {"id": "q_made_by_E1", "text": "{subj} is made by which company?"},
            {"id": "q_made_by_E2", "text": "The manufacturer of {subj} is what?"},
            {"id": "q_made_by_E3", "text": "What firm produces {subj}?"},
        ],
    },
    "created_by": {
        "train": [
            {"id": "q_created_by_T1", "text": "Who created {subj}?"},
            {"id": "q_created_by_T2", "text": "Who is the artist behind {subj}?"},
            {"id": "q_created_by_T3", "text": "Name the creator of {subj}."},
        ],
        "eval": [
            {"id": "q_created_by_E1", "text": "{subj} was created by whom?"},
            {"id": "q_created_by_E2", "text": "The artist of {subj} is who?"},
            {"id": "q_created_by_E3", "text": "Which person produced {subj}?"},
        ],
    },
    "known_for": {
        "train": [
            {"id": "q_known_for_T1", "text": "What is {subj} known for?"},
            {"id": "q_known_for_T2", "text": "What discovery is {subj} known for?"},
            {"id": "q_known_for_T3", "text": "Name the discovery associated with {subj}."},
        ],
        "eval": [
            {"id": "q_known_for_E1", "text": "{subj} is famous for which discovery?"},
            {"id": "q_known_for_E2", "text": "The discovery credited to {subj} is what?"},
            {"id": "q_known_for_E3", "text": "Which result is {subj} most associated with?"},
        ],
    },
    # ---- second hop ----
    "nationality": {
        "train": [
            {"id": "q_nationality_T1", "text": "What is the nationality of {subj}?"},
            {"id": "q_nationality_T2", "text": "What nationality is {subj}?"},
            {"id": "q_nationality_T3", "text": "State the nationality of {subj}."},
        ],
        "eval": [
            {"id": "q_nationality_E1", "text": "{subj} holds the citizenship of where?"},
            {"id": "q_nationality_E2", "text": "Of what nationality is {subj}?"},
            {"id": "q_nationality_E3", "text": "Which nationality does {subj} carry?"},
        ],
    },
    "uses_currency": {
        "train": [
            {"id": "q_uses_currency_T1", "text": "What currency does {subj} use?"},
            {"id": "q_uses_currency_T2", "text": "Which currency is used in {subj}?"},
            {"id": "q_uses_currency_T3", "text": "Name the currency of {subj}."},
        ],
        "eval": [
            {"id": "q_uses_currency_E1", "text": "{subj} runs on which currency?"},
            {"id": "q_uses_currency_E2", "text": "The money used in {subj} is what?"},
            {"id": "q_uses_currency_E3", "text": "What legal tender does {subj} have?"},
        ],
    },
    "headquartered_in": {
        "train": [
            {"id": "q_headquartered_in_T1", "text": "In which country is {subj} headquartered?"},
            {"id": "q_headquartered_in_T2", "text": "Where is {subj} headquartered?"},
            {"id": "q_headquartered_in_T3", "text": "Name the HQ country of {subj}."},
        ],
        "eval": [
            {"id": "q_headquartered_in_E1", "text": "{subj} keeps its head office in which country?"},
            {"id": "q_headquartered_in_E2", "text": "The headquarters of {subj} is in what country?"},
            {"id": "q_headquartered_in_E3", "text": "Which nation hosts {subj}'s base?"},
        ],
    },
    "birth_country": {
        "train": [
            {"id": "q_birth_country_T1", "text": "In which country was {subj} born?"},
            {"id": "q_birth_country_T2", "text": "Where was {subj} born?"},
            {"id": "q_birth_country_T3", "text": "Name the birth country of {subj}."},
        ],
        "eval": [
            {"id": "q_birth_country_E1", "text": "{subj} came into the world in which country?"},
            {"id": "q_birth_country_E2", "text": "The birthplace country of {subj} is what?"},
            {"id": "q_birth_country_E3", "text": "Which nation is {subj} a native of?"},
        ],
    },
    "field": {
        "train": [
            {"id": "q_field_T1", "text": "Which scientific field does {subj} belong to?"},
            {"id": "q_field_T2", "text": "What field is {subj} part of?"},
            {"id": "q_field_T3", "text": "Name the field of {subj}."},
        ],
        "eval": [
            {"id": "q_field_E1", "text": "{subj} falls under which scientific field?"},
            {"id": "q_field_E2", "text": "The discipline of {subj} is what?"},
            {"id": "q_field_E3", "text": "Which branch of science covers {subj}?"},
        ],
    },
}


# --------------------------------------------------------------------------- #
# 4. Repair-trace templates, per cell. >=2 train + >=2 eval.
#
# Format fields available per cell (generator fills what it has):
#   all cells   : {fact1} {fact2}        (oracle sentences; K-Cor uses only fact1)
#   H-Aug/H-Abl : {gold}
#   H-Cor       : {gold} {wrong_tail} {wrong_bridge}
#   K-Cor       : {gold} {wrong_tail} {bridge}
#   Clean       : {gold}
#
# Trace train/eval phrasings share no scaffold wording.
# --------------------------------------------------------------------------- #

TRACE_TEMPLATES: dict[str, dict[str, list[dict[str, str]]]] = {
    "H-Aug": {
        "train": [
            {"id": "tr_haug_T1", "text": "The tentative answer is unsupported because the bridge fact is missing from the problem. Retrieve the bridge fact: {fact1} Then use the second fact: {fact2} Therefore the final answer is {gold}."},
            {"id": "tr_haug_T2", "text": "No bridge fact is given, so the tentative answer cannot be trusted. Recall the missing link: {fact1} Combine it with: {fact2} Hence the answer is {gold}."},
        ],
        "eval": [
            {"id": "tr_haug_E1", "text": "The problem omits the bridging fact, leaving the tentative answer ungrounded. Surface it from memory: {fact1} Chain it onward: {fact2} So the corrected answer is {gold}."},
            {"id": "tr_haug_E2", "text": "Because the connecting fact is absent, the guess stands unverified. Bring back: {fact1} Apply next: {fact2} The answer therefore resolves to {gold}."},
        ],
    },
    "H-Abl": {
        "train": [
            {"id": "tr_habl_T1", "text": "The problem hides or removes the bridge entity. Recover the bridge entity from oracle facts: {fact1} Then use the recovered entity to answer: {fact2} Therefore the final answer is {gold}."},
            {"id": "tr_habl_T2", "text": "The bridging entity is masked out of the question. Reconstruct it: {fact1} With it in hand, continue: {fact2} Hence the final answer is {gold}."},
        ],
        "eval": [
            {"id": "tr_habl_E1", "text": "Since the middle entity is concealed, recover it first: {fact1} Then carry it through: {fact2} The corrected answer is {gold}."},
            {"id": "tr_habl_E2", "text": "The intermediate entity has been stripped from the prompt. Restore it: {fact1} Use the restored entity to finish: {fact2} So the answer is {gold}."},
        ],
    },
    "H-Cor": {
        "train": [
            {"id": "tr_hcor_T1", "text": "The tentative answer follows the planted bridge '{wrong_bridge}'. Verify the bridge claim against the known facts. The planted bridge is false: {fact1} (not {wrong_bridge}). Use the correct bridge fact: {fact2} Therefore the final answer is {gold}, not {wrong_tail}."},
            {"id": "tr_hcor_T2", "text": "The guess trusts the injected bridge '{wrong_bridge}'. Check the source: the real link is {fact1}, so '{wrong_bridge}' is wrong. Apply the true fact: {fact2} The answer is {gold}, not {wrong_tail}."},
        ],
        "eval": [
            {"id": "tr_hcor_E1", "text": "The tentative answer rests on the misattributed bridge '{wrong_bridge}'. Cross-check it: the true relation is {fact1}, which rules out '{wrong_bridge}'. Then: {fact2} Hence the corrected answer is {gold} rather than {wrong_tail}."},
            {"id": "tr_hcor_E2", "text": "A false bridge '{wrong_bridge}' was inserted, and the guess accepted it. The records actually say {fact1}, so reject '{wrong_bridge}'. Continue with {fact2} The answer resolves to {gold}, not {wrong_tail}."},
        ],
    },
    "K-Cor": {
        "train": [
            {"id": "tr_kcor_T1", "text": "Check the stated claim against the known fact. The stated claim says '{bridge}' is '{wrong_tail}', but the known fact is: {fact1} The stated claim is false. Use the correct fact. Therefore the final answer is {gold}."},
            {"id": "tr_kcor_T2", "text": "Compare the planted claim with what is known. It asserts '{bridge}' is '{wrong_tail}'; however {fact1} contradicts it. Discard the claim. The answer is {gold}."},
        ],
        "eval": [
            {"id": "tr_kcor_E1", "text": "Test the injected claim about '{bridge}'. It states '{wrong_tail}', yet the established fact is {fact1}, which conflicts. Reject the claim, and the corrected answer is {gold}."},
            {"id": "tr_kcor_E2", "text": "The claim that '{bridge}' is '{wrong_tail}' must be checked. The known record {fact1} overrides it. Therefore the right answer is {gold}."},
        ],
    },
    "Clean": {
        "train": [
            {"id": "tr_clean_T1", "text": "The tentative answer is already consistent with the known facts. Specifically, {fact1} {fact2} So the tentative answer '{gold}' matches the chain. No repair is needed."},
            {"id": "tr_clean_T2", "text": "Checking the facts: {fact1} {fact2} The tentative answer '{gold}' agrees with them, so it should be kept unchanged."},
        ],
        "eval": [
            {"id": "tr_clean_E1", "text": "The facts {fact1} {fact2} support the tentative answer '{gold}' exactly. Nothing is wrong, so keep it."},
            {"id": "tr_clean_E2", "text": "Verifying against {fact1} {fact2}, the answer '{gold}' holds up. No correction is warranted."},
        ],
    },
}


# --------------------------------------------------------------------------- #
# Helpers for the validator: flat sets of form ids per split, per kind.
# --------------------------------------------------------------------------- #

def _collect_ids(bank: dict[str, dict[str, list[dict[str, str]]]], split: str) -> set[str]:
    """Collect every form id under `split` across all families/relations."""
    out: set[str] = set()
    for grp in bank.values():
        for entry in grp.get(split, []):
            out.add(entry["id"])
    return out


def all_form_ids(split: str) -> dict[str, set[str]]:
    """Return {kind -> set of form ids} for one split, for disjointness checks."""
    return {
        "question": _collect_ids(QUESTION_TEMPLATES, split),
        "wrong_bridge": _collect_ids(WRONG_BRIDGE_PHRASINGS, split),
        "wrong_claim": _collect_ids(WRONG_CLAIM_PHRASINGS, split),
        "fact_question": _collect_ids(FACT_QUESTION_PHRASINGS, split),
        "trace": _collect_ids(TRACE_TEMPLATES, split),
    }


# Self-check on import: train/eval ids must be disjoint for every bank.
def _assert_disjoint() -> None:
    banks = {
        "QUESTION_TEMPLATES": QUESTION_TEMPLATES,
        "WRONG_BRIDGE_PHRASINGS": WRONG_BRIDGE_PHRASINGS,
        "WRONG_CLAIM_PHRASINGS": WRONG_CLAIM_PHRASINGS,
        "FACT_QUESTION_PHRASINGS": FACT_QUESTION_PHRASINGS,
        "TRACE_TEMPLATES": TRACE_TEMPLATES,
    }
    for name, bank in banks.items():
        tr = _collect_ids(bank, "train")
        ev = _collect_ids(bank, "eval")
        overlap = tr & ev
        assert not overlap, f"{name}: train/eval form ids overlap: {sorted(overlap)}"


_assert_disjoint()
