import re
"""Rough phonetic transliteration of the 9 Brahmic scripts used in the data into Latin.
All Brahmic blocks share the ISCII-derived layout, so each code point is mapped to
Devanagari by its offset inside its 128-char block, then a single Devanagari table is used."""
BLOCK_BASES = [0x900, 0x980, 0xA00, 0xA80, 0xB00, 0xB80, 0xC00, 0xC80, 0xD00]
SPECIAL = {  # characters whose offset does not line up with Devanagari
    '\u0D7A': 'n', '\u0D7B': 'n', '\u0D7C': 'r', '\u0D7D': 'l', '\u0D7E': 'l', '\u0D7F': 'k',  # Malayalam chillu
    '\u09CE': 't', '\u09DF': 'y', '\u0B5F': 'y', '\u0B71': 'v', '\u0A70': 'n', '\u0A71': '',
    '\u0A72': '', '\u0A73': '', '\u0A74': 'om', '\u200C': '', '\u200D': '',
}
V_IND = {0x05: 'a', 0x06: 'a', 0x07: 'i', 0x08: 'i', 0x09: 'u', 0x0A: 'u', 0x0B: 'ri', 0x0C: 'li', 0x0D: 'e',
         0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x11: 'o', 0x12: 'o', 0x13: 'o', 0x14: 'au', 0x60: 'ri', 0x61: 'li'}
MATRA = {0x3E: 'a', 0x3F: 'i', 0x40: 'i', 0x41: 'u', 0x42: 'u', 0x43: 'ri', 0x44: 'ri', 0x45: 'e', 0x46: 'e',
         0x47: 'e', 0x48: 'ai', 0x49: 'o', 0x4A: 'o', 0x4B: 'o', 0x4C: 'au', 0x62: 'li', 0x63: 'li', 0x4E: 'e',
         0x55: '', 0x56: 'ai', 0x57: 'au'}
CONS = {0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'n', 0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j', 0x1D: 'jh',
        0x1E: 'n', 0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n', 0x24: 't', 0x25: 'th', 0x26: 'd',
        0x27: 'dh', 0x28: 'n', 0x29: 'n', 0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm', 0x2F: 'y',
        0x30: 'r', 0x31: 'r', 0x32: 'l', 0x33: 'l', 0x34: 'l', 0x35: 'v', 0x36: 'sh', 0x37: 'sh', 0x38: 's',
        0x39: 'h', 0x58: 'q', 0x59: 'kh', 0x5A: 'g', 0x5B: 'z', 0x5C: 'r', 0x5D: 'rh', 0x5E: 'f', 0x5F: 'y'}
NUKTA_MOD = {'j': 'z', 'ph': 'f', 'k': 'q', 'd': 'r', 'dh': 'rh', 'g': 'g', 'kh': 'kh'}
SIGNS = {0x01: 'n', 0x02: 'n', 0x03: 'h'}
VIRAMA, NUKTA = 0x4D, 0x3C


def _offset(ch):
    o = ord(ch)
    for b in BLOCK_BASES:
        if b <= o < b + 0x80:
            return o - b
    return None


def translit(word):
    out = []
    pending = None  # consonant waiting for its vowel
    for ch in word:
        if ch in SPECIAL:
            if pending is not None:
                out.append(pending); pending = None
            out.append(SPECIAL[ch]); continue
        off = _offset(ch)
        if off is None:
            if pending is not None:
                out.append(pending + 'a'); pending = None
            if ch.isalnum():
                out.append(ch.lower())
            continue
        if off in CONS:
            if pending is not None:
                out.append(pending + 'a')
            pending = CONS[off]
        elif off == NUKTA:
            if pending is not None:
                pending = NUKTA_MOD.get(pending, pending)
        elif off == VIRAMA:
            if pending is not None:
                out.append(pending); pending = None
        elif off in MATRA:
            if pending is not None:
                out.append(pending + MATRA[off]); pending = None
            else:
                out.append(MATRA[off])
        elif off in V_IND:
            if pending is not None:
                out.append(pending + 'a'); pending = None
            out.append(V_IND[off])
        elif off in SIGNS:
            if pending is not None:
                out.append(pending + 'a'); pending = None
            out.append(SIGNS[off])
        elif 0x66 <= off <= 0x6F:
            if pending is not None:
                out.append(pending + 'a'); pending = None
            out.append(str(off - 0x66))
        else:
            if pending is not None:
                out.append(pending + 'a'); pending = None
    if pending is not None:
        out.append(pending)  # schwa deletion at word end
    return ''.join(out)


def skeleton(latin):
    """Consonant skeleton with voicing merged: robust key for comparing English words with back-transliterations."""
    s = latin.lower()
    s = re.sub(r'c(?=[eiy])', 's', s)  # English soft c
    for a, b in (('chh', 'c'), ('ch', 'c'), ('sh', 's'), ('ph', 'f'), ('kh', 'k'), ('gh', 'g'), ('th', 't'),
                 ('dh', 'd'), ('bh', 'b'), ('jh', 'j'), ('ck', 'k'), ('x', 'ks'), ('q', 'k'), ('w', 'v'), ('z', 'j')):
        s = s.replace(a, b)
    table = str.maketrans({'g': 'k', 'c': 'k', 'j': 'k', 'd': 't', 'b': 'p', 'f': 'p', 'v': 'p', 'z': 's'})
    s = s.translate(table)
    s = ''.join(c for c in s if c not in 'aeiouyh')
    # collapse doubles
    out = []
    for c in s:
        if not out or out[-1] != c:
            out.append(c)
    return ''.join(out)
