from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Iterable, Optional, Any

import spacy
import re
import unicodedata
from collections import defaultdict
import math
from rapidfuzz.distance import Levenshtein
from wordfreq import zipf_frequency
from dataclasses import dataclass
from symspellpy import Verbosity

from rapidfuzz.distance import Levenshtein
from difflib import SequenceMatcher


WORDLIKE_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")

# --- Regex patterns ---
MULTI_PUNCT_RE = re.compile(r"([,;:.!?])\1+")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;:.!?])")
SPACE_AFTER_OPEN_RE = re.compile(r"([(\[\{])\s+")
SPACE_BEFORE_CLOSE_RE = re.compile(r"\s+([)\]\}])")

# Hyphenated line breaks: "Guer-\nra" -> "Guerra"
HYPHEN_LINEBREAK_RE = re.compile(r"(\w+)-\s*\n\s*(\w+)")


WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")

# Any remaining newlines
NEWLINE_RE = re.compile(r"\s*\n\s*")

# Repeated characters: "llllllll", "iiiiii", "-----" -> single char
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")  # 3+ repetitions




# --- Regex patterns (keep yours or use these) ---
NBSP_RE = re.compile(r"[\u00A0\u2007\u202F]")          # NBSP variants
MULTI_SPACE_RE = re.compile(r"[ \t]+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;:.!?])")
SPACE_AFTER_OPEN_RE = re.compile(r"([(\[\{])\s+")
SPACE_BEFORE_CLOSE_RE = re.compile(r"\s+([)\]\}])")

# Hyphenated line breaks: "Guer-\nra" -> "Guerra" (letters only, avoid numbers)
HYPHEN_LINEBREAK_RE = re.compile(
    r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,})-\s*\n\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,})"
)

# Collapse extreme punctuation runs, but keep ellipses optionally
MULTI_PUNCT_RE = re.compile(r"([,;:!?])\1{1,}")         # "!!"->"!"
MULTI_DOT_RE = re.compile(r"\.{4,}")                    # "...."->"..."
MULTI_DASH_RE = re.compile(r"[-‐-‒–—]{3,}")             # long dash runs -> "—"
MULTI_UNDERSCORE_RE = re.compile(r"_{3,}")              # "____" -> "_"

# Normalize whitespace around newlines (don’t erase by default)
NEWLINE_TRIM_RE = re.compile(r"[ \t]*\n[ \t]*")



def norm(tok: str) -> str:
    return tok.lower()


def doc_tokens(doc) -> List[str]:
    return [t.text for t in doc]


@dataclass(frozen=True)
class MinedPatch:
    raw_span: Tuple[str, ...]
    fixed_span: Tuple[str, ...]
    count: int


@dataclass(frozen=True)
class MinedWordReplacement:
    raw: str
    fixed: str
    count: int
    raw_total_occurrences: int
    fixed_given_raw_rate: float


def mine_patches_and_words(
    text_pairs: List[Dict[str, str]],
    nlp,
    raw_key: str = "raw_text",
    fixed_key: str = "gpt_fixed_conservative",
    phrase_min_span: int = 2,
    phrase_max_span: int = 5,
    include_newlines_in_phrases: bool = False,
    lowercase_word_key: bool = False,
) -> Tuple[
    Counter[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    Counter[Tuple[str, str]],
    Counter[str],
]:
    """
    Returns:
      phrase_counter: counts of (raw_span_tokens, fixed_span_tokens) where either side length in [2,5]
      word_pair_counter: counts of (raw_token, fixed_token) for 1->1 replacements
      word_raw_totals: total occurrences of raw_token in 1->1 replacement events (for P(fixed|raw))
    """
    phrase_counter: Counter[Tuple[Tuple[str, ...], Tuple[str, ...]]] = Counter()
    word_pair_counter: Counter[Tuple[str, str]] = Counter()
    word_raw_totals: Counter[str] = Counter()

    for ex in text_pairs:
        raw_doc = nlp(ex[raw_key])
        fix_doc = nlp(ex[fixed_key])

        raw_toks = doc_tokens(raw_doc)
        fix_toks = doc_tokens(fix_doc)

        sm = SequenceMatcher(
            a=[norm(t) for t in raw_toks],
            b=[norm(t) for t in fix_toks],
            autojunk=False,
        )

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue

            raw_span = tuple(raw_toks[i1:i2])
            fix_span = tuple(fix_toks[j1:j2])

            # ---- Word replacements (high precision): only 1->1 replace ----
            if tag == "replace" and len(raw_span) == 1 and len(fix_span) == 1:
                raw_tok = raw_span[0]
                fix_tok = fix_span[0]

                # ignore case-only changes for replacements (optional)
                if raw_tok.lower() == fix_tok.lower():
                    continue

                key_raw = raw_tok.lower() if lowercase_word_key else raw_tok
                word_pair_counter[(key_raw, fix_tok)] += 1
                word_raw_totals[key_raw] += 1
                continue  # a 1->1 replace is not a phrase patch

            # ---- Phrase patches (2–5 tokens): mostly from replace, but you can include insert/delete if you want ----
            if tag == "replace":
                raw_len, fix_len = len(raw_span), len(fix_span)

                in_bounds = (
                    phrase_min_span <= raw_len <= phrase_max_span
                    or phrase_min_span <= fix_len <= phrase_max_span
                )
                big_enough = (raw_len >= phrase_min_span) or (fix_len >= phrase_min_span)

                if in_bounds and big_enough:
                    if not include_newlines_in_phrases:
                        # spaCy usually keeps newlines in whitespace_, not token.text,
                        # so this is conservative: skip spans that *contain* '\n' inside tokens.
                        if any("\n" in t for t in raw_span) or any("\n" in t for t in fix_span):
                            continue

                    phrase_counter[(raw_span, fix_span)] += 1

    return phrase_counter, word_pair_counter, word_raw_totals


def finalize_phrase_patches(
    phrase_counter: Counter[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    min_count: int = 2
) -> List[MinedPatch]:
    patches = [
        MinedPatch(raw_span=k[0], fixed_span=k[1], count=v)
        for k, v in phrase_counter.items()
        if v >= min_count
    ]
    patches.sort(key=lambda p: p.count, reverse=True)
    return patches


def finalize_word_replacements(
    word_pair_counter: Counter[Tuple[str, str]],
    word_raw_totals: Counter[str],
    min_count: int = 2,
    min_precision: float = 0.90,
) -> List[MinedWordReplacement]:
    out: List[MinedWordReplacement] = []
    for (raw_tok, fix_tok), cnt in word_pair_counter.items():
        if cnt < min_count:
            continue
        total_for_raw = word_raw_totals[raw_tok]
        prec = cnt / total_for_raw if total_for_raw else 0.0
        if prec < min_precision:
            continue
        out.append(
            MinedWordReplacement(
                raw=raw_tok,
                fixed=fix_tok,
                count=cnt,
                raw_total_occurrences=total_for_raw,
                fixed_given_raw_rate=prec,
            )
        )
    out.sort(key=lambda r: (r.count, r.fixed_given_raw_rate), reverse=True)
    return out


def build_word_map(word_replacements: List[MinedWordReplacement]) -> Dict[str, str]:
    """
    If multiple fixed candidates remain for a raw token (rare after precision filtering),
    keep the most frequent.
    """
    best: Dict[str, Tuple[str, int]] = {}
    for r in word_replacements:
        if (r.raw not in best) or (r.count > best[r.raw][1]):
            best[r.raw] = (r.fixed, r.count)
    return {raw: fixed for raw, (fixed, _) in best.items()}

def preclean_ocr_text(text: str, *, collapse_newlines: bool = True) -> str:
    """
    Conservative pre-clean for *dictionary building*.

    Goals:
      - normalize Unicode + whitespace
      - repair hyphen-newline word breaks
      - reduce obvious punctuation noise
      - avoid changing letter sequences (no repeated-letter collapsing)

    Params:
      collapse_newlines:
        - True: replace newlines with a single space (safe for dict building)
        - False: keep newlines but normalize surrounding whitespace
    """
    if not text:
        return ""

    # 1) Unicode normalization (stable)
    text = unicodedata.normalize("NFC", text)

    # 2) Normalize NBSP variants -> normal space
    text = NBSP_RE.sub(" ", text)

    # 3) Fix hyphenated line-break words: "Guer-\nra" -> "Guerra"
    text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    # 4) Normalize whitespace around newlines
    text = NEWLINE_TRIM_RE.sub("\n", text)

    # 5) Optionally collapse newlines to spaces (dictionary building usually wants this)
    if collapse_newlines:
        text = text.replace("\n", " ")

    # 6) Reduce punctuation noise (don’t touch letters)
    text = MULTI_PUNCT_RE.sub(r"\1", text)       # "!!!!" -> "!"
    text = MULTI_DOT_RE.sub("...", text)         # "....." -> "..."
    text = MULTI_DASH_RE.sub("—", text)          # "-----" -> "—"
    text = MULTI_UNDERSCORE_RE.sub("_", text)    # "____" -> "_"

    # 7) Fix spacing artifacts (safe)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = SPACE_BEFORE_CLOSE_RE.sub(r"\1", text)

    return text.strip()


def simple_tok(text: str):
    for w in WORD_RE.findall(text):
        if len(w) < 3:
            continue
        yield w




@dataclass
class ScoredCandidate:
    term: str
    score: float
    breakdown: Dict[str, float]
    edit_distance: Optional[int] = None
    symspell_count: Optional[int] = None


def alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    a = sum(ch.isalpha() for ch in s)
    return a / len(s)


def weirdness_penalty(s: str) -> float:
    """
    Penalize candidates that look non-word-like.
    Returns a non-negative penalty to subtract from score.
    """
    if not s:
        return 5.0

    pen = 0.0

    # Too many non-letters for a word candidate
    ar = alpha_ratio(s)
    if ar < 0.6:
        pen += (0.6 - ar) * 6.0  # scale

    # Mixed digits in candidate (often bad for Spanish word correction)
    if any(ch.isdigit() for ch in s):
        pen += 1.5

    # Excess punctuation inside the candidate
    punct = sum((not ch.isalnum()) and (ch not in "áéíóúüñÁÉÍÓÚÜÑ-") for ch in s)
    if punct:
        pen += min(2.0, 0.5 * punct)

    # Very long tokens are suspicious (but don't over-penalize)
    if len(s) >= 20:
        pen += 1.0

    return pen


def zipf_score(term: str, lang: str = "es") -> float:
    """
    Zipf frequency is ~0..7+; higher = more common.
    For OCR, function words should score high, rare proper nouns lower.
    """
    return zipf_frequency(term.lower(), lang)


def confusion_bonus(
    original: str,
    candidate: str,
    confusion_pairs: Optional[Dict[str, str]] = None,
) -> float:
    """
    Optional reward if candidate is explainable by common OCR confusions.
    confusion_pairs maps 'src' -> 'dst' (e.g., {'rn': 'm', '0': 'O', 'l': 'I'}).

    This is a light heuristic (fast). It does not compute full edit paths.
    """
    if not confusion_pairs:
        return 0.0

    o = original
    c = candidate

    bonus = 0.0

    # Reward if applying a known confusion replacement moves original closer to candidate.
    # We'll measure closeness by Levenshtein distance change.
    base = Levenshtein.distance(o, c)

    for src, dst in confusion_pairs.items():
        if src in o:
            o2 = o.replace(src, dst)
            d2 = Levenshtein.distance(o2, c)
            if d2 < base:
                # improvement => reward
                bonus += min(1.0, (base - d2) / max(1, base))  # 0..1
                base = d2  # update baseline, prevents double counting too much

    # Scale bonus modestly (we don't want this to override frequency entirely)
    return 0.8 * bonus


def score_candidate(
    original: str,
    candidate: str,
    *,
    symspell_edit_distance: Optional[int] = None,
    symspell_count: Optional[int] = None,
    lang: str = "es",
    confusion_pairs: Optional[Dict[str, str]] = None,
    w_zipf: float = 1.2,
    w_edit: float = 1.0,
    w_count: float = 0.15,
    w_conf: float = 1.0,
) -> ScoredCandidate:
    """
    Higher score = better.

    Components:
    - Zipf frequency bonus (wordfreq)
    - Edit distance penalty
    - SymSpell dictionary count bonus (weak signal; your merged dict dominates quality)
    - Weirdness penalties
    - Confusion bonus (optional)
    """
    # Use SymSpell distance if provided; otherwise compute Levenshtein distance
    ed = symspell_edit_distance
    if ed is None:
        ed = Levenshtein.distance(original, candidate)

    z = zipf_score(candidate, lang=lang)

    # SymSpell counts can be huge; compress via log
    cnt = symspell_count or 0
    cnt_feat = math.log10(cnt + 1)  # 0.. (small-ish)

    conf = confusion_bonus(original, candidate, confusion_pairs=confusion_pairs)

    weird_pen = weirdness_penalty(candidate)

    # Score assembly
    score = (
        w_zipf * z
        - w_edit * ed
        + w_count * cnt_feat
        + w_conf * conf
        - weird_pen
    )

    breakdown = {
        "zipf": w_zipf * z,
        "edit_penalty": -w_edit * ed,
        "count_bonus": w_count * cnt_feat,
        "confusion_bonus": w_conf * conf,
        "weirdness_penalty": -weird_pen,
        "raw_zipf": z,
        "raw_edit_distance": float(ed),
        "raw_count_log10": float(cnt_feat),
    }

    return ScoredCandidate(
        term=candidate,
        score=score,
        breakdown=breakdown,
        edit_distance=ed,
        symspell_count=cnt if symspell_count is not None else None,
    )


def choose_best_candidate(
    original: str,
    symspell_suggestions: Iterable,  # Suggestion objects from symspellpy.lookup()
    *,
    lang: str = "es",
    confusion_pairs: Optional[Dict[str, str]] = None,
    min_score_gain: float = 1.0,
    allow_no_change: bool = True,
) -> Tuple[str, Optional[ScoredCandidate], Optional[ScoredCandidate]]:
    """
    Returns:
      (chosen_term, best_scored, original_scored)

    If allow_no_change is True, we will keep original unless best candidate improves
    score by >= min_score_gain.
    """
    # Score original baseline
    orig_sc = score_candidate(
        original,
        original,
        symspell_edit_distance=0,
        symspell_count=None,
        lang=lang,
        confusion_pairs=confusion_pairs,
    )

    best: Optional[ScoredCandidate] = None

    for sug in symspell_suggestions:
        cand = sug.term
        sc = score_candidate(
            original,
            cand,
            symspell_edit_distance=getattr(sug, "distance", None),
            symspell_count=getattr(sug, "count", None),
            lang=lang,
            confusion_pairs=confusion_pairs,
        )
        if (best is None) or (sc.score > best.score):
            best = sc

    if best is None:
        return original, None, orig_sc

    if allow_no_change:
        if (best.score - orig_sc.score) < min_score_gain:
            return original, best, orig_sc

    return best.term, best, orig_sc

PUNCT_STRIP_RE = re.compile(r"^[,.;:!?]+|[,.;:!?]+$")

def _norm_tok(t: str) -> str:
    # normalize for matching only (not output)
    return PUNCT_STRIP_RE.sub("", t).lower()


RawNormSpan = Tuple[str, ...]
FixedSpan   = Tuple[str, ...]


@dataclass(frozen=True)
class PhrasePatchIndex:
    patch_map: Dict[RawNormSpan, FixedSpan]
    by_first: Dict[str, List[RawNormSpan]]  # first_norm_tok -> list of raw_norm spans (longest-first)


def build_phrase_patch_index(phrase_patches) -> PhrasePatchIndex:
    """
    Build once, reuse across docs.
    phrase_patches: list[MinedPatch] (raw_span, fixed_span)
    """
    patch_map: Dict[RawNormSpan, FixedSpan] = {}
    by_first: Dict[str, List[RawNormSpan]] = defaultdict(list)

    for p in phrase_patches:
        raw_norm = tuple(_norm_tok(t) for t in p.raw_span)
        if not raw_norm:
            continue
        patch_map[raw_norm] = p.fixed_span
        by_first[raw_norm[0]].append(raw_norm)

    # longest match wins
    for k in by_first:
        by_first[k].sort(key=len, reverse=True)

    return PhrasePatchIndex(patch_map=dict(patch_map), by_first=dict(by_first))


def apply_phrase_patches_doc(
    doc: Doc,
    *,
    nlp,  # used only to create the output Doc via tokenizer
    patch_index: PhrasePatchIndex,
) -> Doc:
    """
    Applies phrase patches to an *already-parsed* spaCy Doc.
    Returns a new Doc (tokenizer-only) with whitespace preserved via re-tokenization.
    """
    toks = [t.text for t in doc]
    wss  = [t.whitespace_ for t in doc]
    toks_norm = [_norm_tok(t) for t in toks]

    out_parts: List[str] = []
    i = 0
    n = len(toks)

    while i < n:
        first_norm = toks_norm[i]
        matched = False

        for raw_norm in patch_index.by_first.get(first_norm, []):
            L = len(raw_norm)
            if i + L > n:
                continue
            if tuple(toks_norm[i:i+L]) == raw_norm:
                fixed_span = patch_index.patch_map[raw_norm]
                tail_ws = wss[i + L - 1]  # preserve whitespace after replaced span

                if fixed_span:
                    for j, ft in enumerate(fixed_span):
                        out_parts.append(ft)
                        out_parts.append(tail_ws if j == len(fixed_span) - 1 else " ")
                else:
                    out_parts.append(tail_ws)

                i += L
                matched = True
                break

        if not matched:
            out_parts.append(toks[i])
            out_parts.append(wss[i])
            i += 1

    fixed_text = "".join(out_parts)

    # tokenizer-only doc creation (fast), preserves whitespace from fixed_text
    return nlp.make_doc(fixed_text)


CLEAN_WORD_RE = re.compile(r"^[a-záéíóúüñ]+$", re.IGNORECASE)

def build_vocab_set(
    fixed_counts: Counter,
    raw_counts: Counter,
    *,
    fixed_min_count: int = 10,
    raw_min_count: int = 50,
    min_len: int = 3,
) -> set[str]:
    vocab = set()

    # 1) trusted: corrected side
    for w, c in fixed_counts.items():
        if c >= fixed_min_count and len(w) >= min_len and CLEAN_WORD_RE.match(w):
            vocab.add(w.lower())

    # 2) expanded: raw side, but only frequent + clean tokens
    for w, c in raw_counts.items():
        if c >= raw_min_count and len(w) >= min_len and CLEAN_WORD_RE.match(w):
            vocab.add(w.lower())

    return vocab


# --- helpers -------------------------------------------------------------

def strip_affixes(token_text: str) -> tuple[str, str, str]:
    """Return (prefix, core, suffix) where prefix/suffix are non-alnum runs."""
    if not token_text:
        return "", "", ""
    i, j = 0, len(token_text)

    while i < j and not token_text[i].isalnum():
        i += 1
    while j > i and not token_text[j - 1].isalnum():
        j -= 1

    return token_text[:i], token_text[i:j], token_text[j:]


def is_protected(tok) -> bool:
    """Hard 'do not touch' gate."""
    if tok.is_space or tok.is_punct:
        return True
    if tok.like_num:
        return True
    # keep very short tokens (function words, initials, abbreviations)
    if len(tok.text) <= 2:
        return True
    # keep short ALLCAPS acronyms
    if tok.text.isupper() and len(tok.text) <= 4:
        return True
    return False


def is_suspect(
    tok,
    *,
    vocab_set: Set[str],
    min_len: int = 3,
) -> bool:
    """Conservative suspect heuristic (includes mixed-case anomaly)."""
    text = tok.text

    if is_protected(tok):
        return False

    if len(text) < min_len:
        return False

    # mixed-case anomaly: spSundatUmÍDa
    has_lower = any(ch.islower() for ch in text)
    has_upper = any(ch.isupper() for ch in text)
    if has_lower and has_upper:
        if not (text.istitle() or text.isupper() or (text[0].isupper() and text[1:].islower())):
            return True

    alpha = sum(ch.isalpha() for ch in text)
    if alpha == 0:
        return False

    alpha_ratio = alpha / len(text)
    if alpha_ratio < 0.6:
        return True

    lower = text.lower()
    if lower in vocab_set:
        return False

    if any(ch.isdigit() for ch in text):
        return True

    if any(ch in "._,:;|/\\~" for ch in text[1:-1]):
        return True

    if len(text) >= 15:
        return True

    return False



def fix_ocr_doc(
    doc: Doc,
    nlp,
    patch_index,                 # PhrasePatchIndex from build_phrase_patch_index(...)
    symspell,
    vocab_set: Set[str],
    word_map: Optional[Dict[str, str]] = None,
    use_phrase_patches: bool = True,
    use_word_map: bool = False,
    use_symspell: bool = True,
    symspell_verbosity=Verbosity.CLOSEST,
    max_edit_distance: int = 2,
    min_score_gain: float = 1.0,
    confusion_pairs: Optional[Dict[str, str]] = None,
    lowercase_lookup: bool = True,
) -> Doc:
    """
    Cleaner that operates on a spaCy Doc and returns a new Doc.
    Preserves whitespace/newlines by reconstructing text and re-tokenizing with nlp.make_doc().
    """

    # 1) phrase patches (doc -> doc)
    if use_phrase_patches and patch_index is not None:
        doc = apply_phrase_patches_doc(doc, nlp=nlp, patch_index=patch_index)

    # 2) token loop (word_map and/or symspell)
    if not (use_word_map or use_symspell):
        return doc

    out_parts: list[str] = []

    for tok in doc:
        t = tok.text

        if tok.is_space:
            out_parts.append(t)
            continue

        # optional word_map pre-pass
        if use_word_map and word_map and not is_protected(tok):
            mapped = word_map.get(t) or word_map.get(t.lower())
            if mapped and mapped.lower() != t.lower():
                out_parts.append(mapped + tok.whitespace_)
                continue

        # optional symspell pass
        if use_symspell and is_suspect(tok, vocab_set=vocab_set):
            prefix, core, suffix = strip_affixes(t)
            if core:
                lookup_term = core.lower() if lowercase_lookup else core
                suggestions = symspell.lookup(
                    lookup_term,
                    symspell_verbosity,
                    max_edit_distance=max_edit_distance,
                )

                chosen, _, _ = choose_best_candidate(
                    lookup_term,
                    suggestions,
                    confusion_pairs=confusion_pairs,
                    min_score_gain=min_score_gain,
                )

                # restore basic casing if we lowercased for lookup
                if lowercase_lookup and chosen != lookup_term:
                    if core.istitle():
                        chosen = chosen[:1].upper() + chosen[1:]
                    elif core.isupper():
                        chosen = chosen.upper()

                out_parts.append((prefix + chosen + suffix) + tok.whitespace_)
                continue

        out_parts.append(t + tok.whitespace_)

    return nlp.make_doc("".join(out_parts))



def evaluate_ocr_fix(raw_doc, gold_doc, fixed_doc) -> dict:
    raw_text   = raw_doc.text
    gold_text  = gold_doc.text
    fixed_text = fixed_doc.text

    # --- CER / WER -------------------------------------------------------
    def cer(a: str, b: str) -> float:
        return 0.0 if not b else Levenshtein.distance(a, b) / len(b)

    def wer(a: str, b: str) -> float:
        aw, bw = a.split(), b.split()
        return 0.0 if not bw else Levenshtein.distance(aw, bw) / len(bw)

    cer_raw   = cer(raw_text, gold_text)
    cer_fixed = cer(fixed_text, gold_text)
    cer_abs_impr = cer_raw - cer_fixed
    cer_pct_impr = (cer_abs_impr / cer_raw) if cer_raw > 0 else 0.0

    wer_raw   = wer(raw_text, gold_text)
    wer_fixed = wer(fixed_text, gold_text)

    # --- token sequences -------------------------------------------------
    raw_t   = raw_text.split()
    gold_t  = gold_text.split()
    fixed_t = fixed_text.split()

    # --- Edit precision/recall using edit operations --------------------
    def editops_as_set(src, tgt):
        ops = Levenshtein.editops(src, tgt)  # list[Editop(tag, src_pos, dest_pos)]
        out = set()
        for op in ops:
            tag, i, j = op.tag, op.src_pos, op.dest_pos
            if tag == "replace":
                out.add(("replace", i, src[i], tgt[j]))
            elif tag == "delete":
                out.add(("delete", i, src[i], ""))
            elif tag == "insert":
                out.add(("insert", i, "", tgt[j]))
        return out

    gold_ops = editops_as_set(raw_t, gold_t)
    sys_ops  = editops_as_set(raw_t, fixed_t)

    tp = len(gold_ops & sys_ops)
    edit_precision = tp / len(sys_ops) if sys_ops else 1.0
    edit_recall    = tp / len(gold_ops) if gold_ops else 1.0

    # --- Damage rate with alignment (fixes off-by-one) -------------------
    # Step 1: find "clean" raw indices where raw token == gold token *in alignment*
    sm_rg = SequenceMatcher(a=raw_t, b=gold_t, autojunk=False)
    clean_raw_idx = set()
    for tag, i1, i2, j1, j2 in sm_rg.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                clean_raw_idx.add(i1 + k)

    # Step 2: build a mapping raw_index -> fixed_token (aligned), allowing shifts
    # We align op spans; for replace, pair up min(len_raw, len_fixed)
    sm_rf = SequenceMatcher(a=raw_t, b=fixed_t, autojunk=False)
    raw_to_fixed = {}  # raw_i -> fixed token or None (deleted/unmapped)

    for tag, i1, i2, j1, j2 in sm_rf.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                raw_to_fixed[i1 + k] = fixed_t[j1 + k]
        elif tag == "replace":
            Lr, Lf = (i2 - i1), (j2 - j1)
            L = min(Lr, Lf)
            for k in range(L):
                raw_to_fixed[i1 + k] = fixed_t[j1 + k]
            for k in range(L, Lr):
                raw_to_fixed[i1 + k] = None
        elif tag == "delete":
            for k in range(i1, i2):
                raw_to_fixed[k] = None
        elif tag == "insert":
            # no raw indices to map
            pass

    # Step 3: count damage only on clean raw tokens
    damaged = 0
    total_clean = 0
    for i in sorted(clean_raw_idx):
        total_clean += 1
        fixed_tok = raw_to_fixed.get(i, None)
        if fixed_tok is None or fixed_tok != raw_t[i]:
            # print("Damaged:", fixed_tok, raw_t[i])
            damaged += 1

    damage_rate = damaged / total_clean if total_clean else 0.0

    return {
        "CER_raw": cer_raw,
        "CER_fixed": cer_fixed,
        "CER_improvement_abs": cer_abs_impr,
        "CER_improvement_pct": cer_pct_impr,

        "WER_raw": wer_raw,
        "WER_fixed": wer_fixed,

        "edit_precision": edit_precision,
        "edit_recall": edit_recall,

        "damage_rate": damage_rate,
        "damage_count": damaged,
        "clean_token_count": total_clean,
    }


def is_word(s: str) -> bool:
    """
    True if the string contains at least one alphabetic Spanish word
    (i.e., not purely numbers or punctuation).
    """
    if not s:
        return False
    return WORD_RE.search(s) is not None