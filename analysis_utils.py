import spacy 

nlp = spacy.blank("es")

def contains_search_word(doc, targets):
    return any(token.lower_ in targets for token in doc)

def get_surrounding_spans(doc, targets, n_window):
    """
    Returns a list[Span] of windows around target tokens.
    """
    n = len(doc)

    # collect (start, end) windows
    windows = []
    for tok in doc:
        if tok.lower_ in targets:
            start = tok.i - n_window
            if start < 0:
                start = 0
            end = tok.i + n_window
            if end > n:
                end = n
            windows.append((start, end))

    if not windows:
        return []

    return [doc[s:e] for s, e in windows]


def get_surrounding_spans_as_doc(doc, targets, n_window):
    """
    Returns a single Doc made by concatenating the extracted spans (preserves token attrs).
    """
    spans = get_surrounding_spans(doc, targets, n_window)
    if not spans:
        return spacy.tokens.Doc(doc.vocab, words=[])

    # from_docs preserves token attributes best; fewer spans (via merge) => faster
    return spacy.tokens.Doc.from_docs([span.as_doc() for span in spans])



def spacy_tokenizer(doc):
    # print(type(doc))
    # print(doc)
    tokens = [
        token.lower_.strip()
        for token in doc
        if token.lower_.strip() not in nlp.Defaults.stop_words and
           token.lower_.strip() not in '.,;:!?()[]{}"\'`~@#$%^&*+=-_<>|\\' and
           # Remove non-alphabetic tokens, and single-character tokens (optional)
           token.is_alpha and len(token.lower_.strip()) > 1
    ]
    return tokens