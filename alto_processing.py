from lxml import etree

def xp(node, path):
    return node.xpath(path, namespaces={'a': 'http://www.loc.gov/standards/alto/ns-v2#'})


def _int_attr(el, name, default=0):
    v = el.get(name)
    return int(float(v)) if v is not None else default

def iter_line_tokens(textline):
    """
    Yield tokens in the order they appear inside the TextLine:
    - <String CONTENT="...">
    - <SP> (space)
    - <HYP CONTENT="-"> (hyphen, sometimes)
    """
    # We read children in-document order.
    for child in textline:
        tag = etree.QName(child).localname
        if tag == "String":
            yield ("STR", child.get("CONTENT", ""))
        elif tag == "SP":
            yield ("SP", " ")
        elif tag == "HYP":
            # Some ALTO uses CONTENT, some not; default to hyphen.
            yield ("HYP", child.get("CONTENT") or "-")

def text_of_line(textline):
    parts = []
    for kind, val in iter_line_tokens(textline):
        parts.append(val)
    # Normalize: collapse repeated spaces 
    return "".join(parts).strip()

def line_sort_key(textline):
    # Standard ALTO geometry attributes are VPOS/HPOS (y/x), WIDTH/HEIGHT
    y = _int_attr(textline, "VPOS", 0)
    x = _int_attr(textline, "HPOS", 0)
    return (y, x)


def extract_block_text(textblock, join_with="\n"):
    """
    Extract text for a single <TextBlock>:
    - sorts TextLine by (VPOS, HPOS)
    - joins lines with newline (default) or space
    """
    lines = xp(textblock, "./a:TextLine")
    lines_sorted = sorted(lines, key=line_sort_key)
    # print(lines_sorted)
    out_lines = []
    for tl in lines_sorted:
        s = text_of_line(tl)
        if s:
            out_lines.append(s)

    # Optional: handle end-of-line hyphenation:
    # if a line ends with '-' and next line starts with a letter, merge them.
    merged = []
    i = 0
    while i < len(out_lines):
        cur = out_lines[i]
        if i < len(out_lines) - 1 and cur.endswith("-"):
            nxt = out_lines[i + 1].lstrip()
            if nxt and nxt[0].isalpha():
                merged.append(cur[:-1] + nxt)  # drop hyphen + merge
                i += 2
                continue
        merged.append(cur)
        i += 1

    return join_with.join(merged)


def extract_text_blocks_from_path(path):
    tree = etree.parse(path)
    root = tree.getroot()
    
    processed_blocks = []
    blocks = xp(root, './/a:TextBlock')
    for b in blocks:
        text = extract_block_text(b, join_with="\n")
        processed_blocks.append({
            'id': b.get('ID'),
            'height': b.get('HEIGHT'),
            'width': b.get('WIDTH'),
            'hpos': b.get('HPOS'),
            'vpos': b.get('VPOS'),
            'text': text
        })
    return processed_blocks