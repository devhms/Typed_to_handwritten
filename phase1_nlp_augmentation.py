"""
Phase 1: NLP Data Augmentation — Pakistani English (PakE) Dialect Modeling
Objective: Augment raw input text with documented sociolinguistic features of
Pakistani English to build a low-resource World Englishes OCR training corpus.

References:
  [1] Baumgardner (1990) - The indigenization of English in Pakistan
  [2] Mahboob & Ahmar (2004) - Pakistani English: Phonology
  [3] Talaat (2003) - Certain grammatical features of Pakistani English
  [4] Rahman (1990) - Pakistani English
  [5] Platt, Weber & Ho (1984) - The New Englishes
  [6] Kachru (1986) - The Alchemy of English
"""

import re
import random
import json
import sys
from pathlib import Path
from typing import Optional

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LEXICAL RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

# PakE transitional / discourse markers (Ref [3], [4], [2023 update])
DISCOURSE_MARKERS = [
    "Because,", "Hence,", "Likewise,", "Moreover,", "Kindly note that",
    "It is worth mentioning that", "As such,", "Thereby,", "Henceforth,",
    "In this regard,", "For the same,", "Accordingly,", 
    "In the light of the above,", "Furthermore,", "As a matter of fact,"
]

# Uncountable → countable pluralization (Ref [5], [6], [2023 update])
UNCOUNTABLE_PLURALS = {
    r"\binformation\b": "informations",
    r"\bsoftware\b":    "softwares",
    r"\bfurniture\b":   "furnitures",
    r"\badvice\b":      "advices",
    r"\bfeedback\b":    "feedbacks",
    r"\bwork\b":        "works",
    r"\bresearch\b":    "researches",
    r"\bknowledge\b":   "knowledges",
    r"\bequipment\b":   "equipments",
    r"\bstaff\b":       "staffs",
    r"\btraffic\b":     "traffics",
    r"\bluggage\b":     "luggages",
    r"\bmachinery\b":   "machineries",
    r"\btraining\b":    "trainings",
    r"\bmanagement\b":  "managements",
}

# Urdu lexical borrowings / code-switching (Ref: Jadoon 2023, Asgher 2023)
URDU_BORROWINGS = {
    r"\bmeeting\b":        "ijlas",
    r"\bproblem\b":        "masla",
    r"\bgovernment\b":     "sarkar",
    r"\bthank you\b":      "JazakAllah",
    r"\bgoodbye\b":        "Allah Hafiz",
    r"\bpeople\b":         "awam",
    r"\bworker\b":         "mulazim",
    r"\binstitution\b":    "idara",
    r"\bdecisions\b":      "faislas",
}

# Prepositional verb variations (Ref [5])
PREP_VERBS = {
    r"\bcomprises?\b":      "comprises of",
    r"\bstresses?\b":       "stresses on",
    r"\bdiscusses?\b":      "discusses about",
    r"\bemphasizes?\b":     "emphasizes on",
    r"\bconsists?\b":       "consists of",          # already standard, but over-applied
    r"\bexplains?\b":       "explains about",
    r"\bmentions?\b":       "mentions about",
    r"\bcontact\b":         "contact with",
    r"\benter\b":           "enter into",
    r"\boppose\b":          "oppose against",
}

# Verb-agreement substitutions (Ref [2], [3])
# Applied at 14% rate as specified
VERB_AGREEMENT_SUBS = [
    (r"\bthey is\b",       "they is"),
    (r"\bhe are\b",        "he are"),
    (r"\bshe were\b",      "she were"),
    (r"\bI are\b",         "I are"),
    (r"\bwe was\b",        "we was"),
    (r"\bit were\b",       "it were"),
    # Standard → PakE direction (inject variation)
    (r"\bthey are\b",      "they is"),
    (r"\bwe are\b",        "we was"),
    (r"\bhe is\b",         "he are"),
    (r"\bshe is\b",        "she were"),
    (r"\bit is\b",         "it were"),
]

VERB_AGREEMENT_RATE = 0.14   # 14% of eligible verb occurrences

# Filler / hedge phrases common in PakE formal writing (Ref [4])
PAKE_HEDGES = [
    "do the needful",
    "revert back",
    "prepone the meeting",
    "out of station",
    "passed out from university",
    "updation of records",
    "kindly do the needful and oblige",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. AUGMENTATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def inject_discourse_markers(sentences: list[str]) -> list[str]:
    """
    Prepend PakE discourse markers to ~25% of sentences (Ref [3]).
    Creates high burstiness by occasionally chaining two short markers.
    """
    augmented = []
    for sent in sentences:
        stripped = sent.strip()
        if not stripped:
            augmented.append(sent)
            continue
        r = random.random()
        if r < 0.10:
            # Chain two markers (long clause opener)
            m1 = random.choice(DISCOURSE_MARKERS)
            m2 = random.choice(DISCOURSE_MARKERS)
            stripped = f"{m1} {m2} {stripped[0].lower()}{stripped[1:]}"
        elif r < 0.25:
            marker = random.choice(DISCOURSE_MARKERS)
            stripped = f"{marker} {stripped[0].lower()}{stripped[1:]}"
        augmented.append(stripped)
    return augmented


def pluralize_uncountables(text: str) -> str:
    """Replace uncountable nouns with their PakE pluralized forms (Ref [5], [6])."""
    for pattern, replacement in UNCOUNTABLE_PLURALS.items():
        def _replace(m, rep=replacement):
            return rep if random.random() < 0.65 else m.group(0)
        text = re.sub(pattern, _replace, text, flags=re.IGNORECASE)
    return text

def inject_urdu_lexis(text: str) -> str:
    """Inject Urdu borrowings / Urduization (Ref: Jadoon 2023)."""
    for pattern, replacement in URDU_BORROWINGS.items():
        if random.random() < 0.15: # 15% code-switch rate
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE, count=1)
    return text

def apply_stative_progressive(text: str) -> str:
    """Transform stative verbs into PakE progressive aspect (Ref: Rahman 2022)."""
    stative_map = {
        r"\bi know\b":   "I am knowing",
        r"\bi see\b":    "I am seeing",
        r"\bi hear\b":   "I am hearing",
        r"\bhe knows\b": "he is knowing",
        r"\bit costs\b": "it is costing",
        r"\bwe want\b":  "we are wanting",
        r"\byou want\b": "you are wanting",
        r"\bi understand\b": "I am understanding"
    }
    for pattern, replacement in stative_map.items():
        if random.random() < 0.30:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def handle_articles(text: str) -> str:
    """Article omission / addition (Ref: Ahmar & Mahboob 2004)."""
    # Omission in prepositional phrases
    text = re.sub(r"\bin the office\b", "in office", text, flags=re.IGNORECASE)
    text = re.sub(r"\bto the university\b", "to university", text, flags=re.IGNORECASE)
    # Addition
    if random.random() < 0.20:
        text = re.sub(r"\bPakistan\b", "the Pakistan", text)
    return text

def add_prepositional_verbs(text: str) -> str:
    """Inject extra prepositions to transitive verbs (Ref [5])."""
    for pattern, replacement in PREP_VERBS.items():
        def _replace(m, rep=replacement):
            return rep if random.random() < 0.55 else m.group(0)
        text = re.sub(pattern, _replace, text, flags=re.IGNORECASE)
    return text


def inject_verb_agreement_variations(text: str, rate: float = VERB_AGREEMENT_RATE) -> str:
    """
    Apply verb-agreement substitutions at the specified rate (14%).
    Operates on standard verb phrases and substitutes PakE variant.
    """
    for std_pattern, pake_form in VERB_AGREEMENT_SUBS[6:]:   # only injection patterns
        matches = list(re.finditer(std_pattern, text, flags=re.IGNORECASE))
        for m in matches:
            if random.random() < rate:
                text = text[:m.start()] + pake_form + text[m.end():]
    return text


def inject_burstiness(sentences: list[str]) -> list[str]:
    """
    Ensure high burstiness: randomly split some sentences into 2–3 very
    short fragments, and concatenate others into very long compound sentences.
    Tests OCR line-segmentation on both extremes.
    """
    output = []
    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        r = random.random()

        if r < 0.12 and len(sent) > 60:
            # Fragment: split at a comma or mid-point
            mid = sent.find(',', len(sent) // 3)
            if mid == -1:
                mid = len(sent) // 2
            output.append(sent[:mid].strip() + '.')
            output.append(sent[mid:].strip().lstrip(',').strip())

        elif r < 0.20 and i + 1 < len(sentences):
            # Fuse two sentences with a PakE connective
            connective = random.choice([
                "and also", "moreover", "likewise", "as well as", "hence"
            ])
            fused = sent.rstrip('.') + f", {connective} " + \
                    sentences[i + 1].strip().lstrip().lower()
            output.append(fused)
            i += 2
            continue

        else:
            output.append(sent)
        i += 1
    return output


def inject_pake_hedges(text: str, rate: float = 0.05) -> str:
    """
    Replace small phrases with PakE formal hedges at low rate (Ref [4]).
    E.g., 'please reply' → 'kindly do the needful and oblige'.
    """
    replacements = {
        r"\bplease reply\b":   "kindly revert back",
        r"\breschedule\b":     "prepone or postpone",
        r"\bgraduate[sd]?\b":  "passed out from university",
        r"\bupdate[sd]?\b":    "updation of",
    }
    for pattern, rep in replacements.items():
        def _replace(m, r=rep):
            return r if random.random() < rate else m.group(0)
        text = re.sub(pattern, _replace, text, flags=re.IGNORECASE)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 3. TELEMETRY / METADATA
# ─────────────────────────────────────────────────────────────────────────────

def compute_augmentation_stats(original: str, augmented: str) -> dict:
    """Return a metadata dict documenting augmentation decisions."""
    orig_words  = original.split()
    aug_words   = augmented.split()
    return {
        "original_word_count":  len(orig_words),
        "augmented_word_count": len(aug_words),
        "delta_words":          len(aug_words) - len(orig_words),
        "target_verb_agreement_rate": VERB_AGREEMENT_RATE,
        "discourse_markers_available": len(DISCOURSE_MARKERS),
        "uncountable_substitutions":   len(UNCOUNTABLE_PLURALS),
        "prepositional_verb_rules":    len(PREP_VERBS),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def augment(raw_text: str) -> tuple[str, dict]:
    """
    Full Phase-1 augmentation pipeline.

    Returns
    -------
    augmented_text : str
    metadata       : dict
    """
    # ── Sentence tokenisation (simple rule-based) ─────────────────────────
    sentences = re.split(r'(?<=[.!?])\s+', raw_text.strip())

    # ── Step 1: Discourse markers + burstiness ────────────────────────────
    sentences = inject_discourse_markers(sentences)
    sentences = inject_burstiness(sentences)

    # ── Rejoin to string for token-level operations ───────────────────────
    text = ' '.join(sentences)

    # ── Step 2: Syntactic structural variations ───────────────────────────
    text = pluralize_uncountables(text)
    text = apply_stative_progressive(text)
    text = handle_articles(text)
    text = inject_urdu_lexis(text)
    text = add_prepositional_verbs(text)
    text = inject_pake_hedges(text)

    # ── Step 3: Verb-agreement variation (14% rate) ───────────────────────
    text = inject_verb_agreement_variations(text, rate=VERB_AGREEMENT_RATE)

    # ── Step 4: Hyphenated Neologisms (Ref: Buriro 2023) ──────────────────
    if random.random() < 0.25:
        text = re.sub(r"\bproject\b", "mega-project", text, flags=re.IGNORECASE)
        text = re.sub(r"\bpeople\b", "anti-people", text, flags=re.IGNORECASE)

    # ── Metadata ──────────────────────────────────────────────────────────
    meta = compute_augmentation_stats(raw_text, text)

    return text, meta


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Accept input text from CLI argument or fall back to a built-in sample
    if len(sys.argv) > 1:
        raw_input = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        raw_input = (
            "The committee comprises several members who stress the importance of "
            "sharing information with all staff. Research indicates that software "
            "development requires adequate equipment and furniture in office spaces. "
            "They are responsible for providing feedback on the work submitted. "
            "The team was asked to contact the department and discuss the matter. "
            "He is confident that the knowledge gained will benefit everyone. "
            "We are planning to update the records by next week. "
            "The manager explained the new policy to all employees. "
            "It is essential that advice from experts is followed carefully. "
            "Please reply to this email at the earliest convenience."
        )
        print("[INFO] No input file provided. Using built-in sample text.\n")

    augmented_text, metadata = augment(raw_input)

    # ── Save outputs ──────────────────────────────────────────────────────
    out_dir = Path("output/phase1")
    out_dir.mkdir(parents=True, exist_ok=True)

    aug_path  = out_dir / "augmented_text.txt"
    meta_path = out_dir / "augmentation_metadata.json"

    aug_path.write_text(augmented_text, encoding="utf-8")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("══════════════════════════════════════════════════════")
    print("  Phase 1 — PakE Augmentation Complete")
    print("══════════════════════════════════════════════════════")
    print(f"  Output text   : {aug_path}")
    print(f"  Metadata JSON : {meta_path}")
    print(f"  Original words: {metadata['original_word_count']}")
    print(f"  Augmented words: {metadata['augmented_word_count']}")
    print("──────────────────────────────────────────────────────")
    print("\n[Augmented Text Preview]\n")
    print(augmented_text[:800], "…" if len(augmented_text) > 800 else "")
