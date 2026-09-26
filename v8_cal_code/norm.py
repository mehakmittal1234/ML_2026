import re, unicodedata, pickle, os
from translit import translit

_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work', 'data')
INDIC_RE = re.compile(r'[\u0900-\u0DFF]')
NAME_DICT = pickle.load(open(os.path.join(_D, 'indic_name_dict.pkl'), 'rb'))
_raw = pickle.load(open(os.path.join(_D, 'indic_dict_raw.pkl'), 'rb'))['addr']
# native-script state names -> latin state name (multi-word states completed)
_STATE_FIX = {'nadu': 'tamil nadu', 'bengal': 'west bengal', 'andhra': 'andhra pradesh', 'orissa': 'odisha'}
ADDR_DICT = {k: _STATE_FIX.get(v[0], v[0]) for k, v in _raw.items()}

# ---------------------------------------------------------------- shared helpers
def strip_accents(s):
    if s.isascii():
        return s
    s = s.replace('\u0153', 'oe').replace('\u00e6', 'ae').replace('\u00df', 'ss').replace('\u00b0', ' ')
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

_DOTTED = re.compile(r'\b(?:[a-z]\.){1,}[a-z]\b\.?')          # l.l.c. / s.a.s / d.o.
_INWORD_JUNK = re.compile(r'(?<=\w)[#@*](?=\w)')                  # #jaintechn#logies
_NONALNUM = re.compile(r'[^a-z0-9]+')

def _indic_map(text, table, fallback=True):
    if not INDIC_RE.search(text):
        return text
    out = []
    for w in text.split():
        if INDIC_RE.search(w):
            core = w.strip('()[]{},;:"\'')
            v = table.get(core)
            if v is None:
                v = NAME_DICT.get(core) or (translit(core) if fallback else '')
            out.append(v)
        else:
            out.append(w)
    return ' '.join(out)

def _clean_text(s):
    s = strip_accents(s.lower())
    s = s.replace('&', ' and ').replace("'", '').replace('\u2019', '')
    s = _DOTTED.sub(lambda m: m.group(0).replace('.', ''), s)
    s = _INWORD_JUNK.sub('', s)
    return s

# ---------------------------------------------------------------- names
ALIAS_RE = re.compile(r'\s(?:a/k/a|aka|d/b/a|dba|doing business as|trading as|t/a|f/k/a|fka|formerly known as|formerly|also known as)\s', re.I)
IDTAG_RE = re.compile(r'\(\s*id[^)]*\)?', re.I)
PHONE_RE = re.compile(r'\s[-\u2013|:]+\s*\d{6,}\s*$|\b\d{5,}\b')
DOMAIN_RE = re.compile(r'^\W*(?:www\.)?([a-z0-9-]+)\.com\W*$')
LEGAL_CANON = {
    'private': 'pvt', 'pvt': 'pvt', 'pte': 'pvt', 'limited': 'ltd', 'ltd': 'ltd', 'ltda': 'ltd',
    'incorporated': 'inc', 'inc': 'inc', 'corporation': 'corp', 'corp': 'corp', 'company': 'co', 'co': 'co',
    'llc': 'llc', 'llp': 'llp', 'pllc': 'pllc', 'plc': 'plc', 'lp': 'lp', 'pc': 'pc', 'pa': 'pa', 'opc': 'opc',
    'sa': 'sa', 'sas': 'sas', 'sasu': 'sasu', 'sarl': 'sarl', 'eurl': 'eurl', 'sci': 'sci', 'snc': 'snc',
    'scp': 'scp', 'selarl': 'selarl', 'scop': 'scop', 'gie': 'gie', 'sca': 'sca', 'gmbh': 'gmbh',
}
NAME_STOP = {'the', 'of', 'and', 'a', 'an', 'de', 'du', 'des', 'la', 'le', 'les', 'et', 'd', 'l', 'en', 'au',
             'aux', 'smt', 'mr', 'mrs', 'ms', 'ms', 'messrs', 'm', 's'}

def _name_tokens(s):
    toks = _NONALNUM.split(s)
    out = []
    for t in toks:
        if not t:
            continue
        out.append(LEGAL_CANON.get(t, t))
    return out

def norm_name(name):
    """returns (tokens, core_tokens, alt_core_tokens, domain_label)"""
    s = _indic_map(name, NAME_DICT)
    s = IDTAG_RE.sub(' ', s)
    s = PHONE_RE.sub(' ', s)
    s = _clean_text(s)
    dom = ''
    m = DOMAIN_RE.match(s.strip())
    if m:
        dom = m.group(1).replace('-', '')
        return [dom], [dom], [], dom
    alt = []
    parts = ALIAS_RE.split(' ' + s + ' ')
    if len(parts) > 1:
        alt_s = parts[-1]
        alt = [t for t in _name_tokens(alt_s) if t not in NAME_STOP and t not in LEGAL_CANON.values()]
        s = ' '.join(parts)
    toks = _name_tokens(s)
    core = [t for t in toks if t not in NAME_STOP and t not in LEGAL_CANON.values()]
    return toks, core, alt, dom

# ---------------------------------------------------------------- addresses
US_STATES = {'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca', 'colorado': 'co',
             'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
             'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la',
             'maine': 'me', 'maryland': 'md', 'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn',
             'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
             'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
             'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
             'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx',
             'utah': 'ut', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv',
             'wisconsin': 'wi', 'wyoming': 'wy', 'district of columbia': 'dc', 'puerto rico': 'pr'}
IN_STATES = {'andhra pradesh': 'ap', 'arunachal pradesh': 'ar', 'assam': 'as', 'bihar': 'br', 'chhattisgarh': 'cg',
             'chattisgarh': 'cg', 'goa': 'ga', 'gujarat': 'gj', 'haryana': 'hr', 'himachal pradesh': 'hp',
             'jharkhand': 'jh', 'karnataka': 'ka', 'kerala': 'kl', 'madhya pradesh': 'mp', 'maharashtra': 'mh',
             'manipur': 'mn', 'meghalaya': 'ml', 'mizoram': 'mz', 'nagaland': 'nl', 'odisha': 'od', 'orissa': 'od',
             'punjab': 'pb', 'rajasthan': 'rj', 'sikkim': 'sk', 'tamil nadu': 'tn', 'telangana': 'ts',
             'tripura': 'tr', 'uttar pradesh': 'up', 'uttarakhand': 'uk', 'uttaranchal': 'uk', 'west bengal': 'wb',
             'delhi': 'dl', 'new delhi': 'new dl', 'jammu and kashmir': 'jk', 'jammu kashmir': 'jk',
             'chandigarh': 'ch', 'puducherry': 'py', 'pondicherry': 'py', 'ladakh': 'la', 'daman and diu': 'dd',
             'dadra and nagar haveli': 'dn', 'andaman and nicobar islands': 'an', 'lakshadweep': 'ld',
             'tg': 'ts', 'or': 'od', 'ch': 'ch'}
US_STREET = {'street': 'st', 'str': 'st', 'avenue': 'ave', 'av': 'ave', 'avn': 'ave', 'road': 'rd', 'drive': 'dr',
             'drv': 'dr', 'lane': 'ln', 'court': 'ct', 'crt': 'ct', 'circle': 'cir', 'circ': 'cir',
             'boulevard': 'blvd', 'boul': 'blvd', 'place': 'pl', 'parkway': 'pkwy', 'pkway': 'pkwy',
             'highway': 'hwy', 'terrace': 'ter', 'trail': 'trl', 'square': 'sq', 'point': 'pt', 'mount': 'mt',
             'mountain': 'mtn', 'heights': 'hts', 'expressway': 'expy', 'freeway': 'fwy', 'turnpike': 'tpke',
             'crossing': 'xing', 'plaza': 'plz', 'center': 'ctr', 'centre': 'ctr', 'suite': 'ste', 'apartment': 'apt',
             'building': 'bldg', 'floor': 'fl', 'room': 'rm', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
             'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw', 'township': 'twp',
             'county': 'co', 'fort': 'ft', 'saint': 'st', 'junction': 'jct', 'estates': 'est', 'meadows': 'mdws',
             'ridge': 'rdg', 'valley': 'vly', 'creek': 'crk', 'lake': 'lk', 'springs': 'spgs', 'village': 'vlg',
             'extension': 'extn', 'ext': 'extn', 'sector': 'sec', 'nagar': 'ngr', 'colony': 'col', 'opposite': 'opp',
             'near': 'nr', 'post': 'po', 'office': 'off', 'district': 'dist', 'dt': 'dist', 'taluka': 'tal',
             'tq': 'tal', 'tehsil': 'teh', 'industrial': 'indl', 'ind': 'indl', 'estate': 'est', 'area': 'area',
             'main': 'main', 'cross': 'crs', 'phase': 'ph', 'first': '1st', 'second': '2nd', 'third': '3rd',
             'ground': 'grd', 'gr': 'grd', 'bengaluru': 'bangalore', 'gurugram': 'gurgaon', 'calcutta': 'kolkata',
             'bombay': 'mumbai', 'madras': 'chennai', 'poona': 'pune', 'trivandrum': 'thiruvananthapuram',
             'mysuru': 'mysore', 'baroda': 'vadodara', 'vizag': 'visakhapatnam', 'cochin': 'kochi'}
FR_STREET = {'rue': 'r', 'r': 'r', 'avenue': 'av', 'av': 'av', 'ave': 'av', 'boulevard': 'bd', 'bd': 'bd',
             'blvd': 'bd', 'bld': 'bd', 'route': 'rte', 'rte': 'rte', 'chemin': 'ch', 'ch': 'ch', 'chem': 'ch',
             'impasse': 'imp', 'imp': 'imp', 'allee': 'all', 'allees': 'all', 'all': 'all', 'place': 'pl',
             'pl': 'pl', 'square': 'sq', 'sq': 'sq', 'quai': 'quai', 'qu': 'quai', 'cours': 'crs', 'crs': 'crs',
             'faubourg': 'fg', 'fbg': 'fg', 'fg': 'fg', 'residence': 'res', 'res': 'res', 'lotissement': 'lot',
             'lot': 'lot', 'saint': 'st', 'st': 'st', 'sainte': 'ste', 'ste': 'ste', 'passage': 'pass',
             'pass': 'pass', 'promenade': 'prom', 'prom': 'prom', 'hameau': 'ham', 'ham': 'ham', 'lieu': 'lieu',
             'dit': 'dit', 'zone': 'z', 'za': 'za', 'zi': 'zi', 'zac': 'zac', 'batiment': 'bat', 'bat': 'bat',
             'escalier': 'esc', 'esc': 'esc', 'etage': 'etg', 'appartement': 'apt', 'appt': 'apt', 'apt': 'apt',
             'general': 'gen', 'gen': 'gen', 'marechal': 'mal', 'mal': 'mal', 'docteur': 'dr', 'dr': 'dr',
             'professeur': 'pr', 'pr': 'pr', 'president': 'pdt', 'pdt': 'pdt', 'grande': 'gde', 'gde': 'gde',
             'petite': 'pte', 'pte': 'pte', 'voie': 'voie', 'sentier': 'sent', 'sent': 'sent', 'rond': 'rd',
             'point': 'pt', 'pt': 'pt', 'cedex': 'cedex'}
ADDR_PREFIX_DROP = {'no', 'nr', 'num', 'number', 'hno', 'h', 'door', 'plot', 'flat', 'shop', 'n', 'null', 'none',
                    'nan', 'city', 'ciity', 'cty'}
ADDR_STOP_FR = {'de', 'du', 'des', 'la', 'le', 'les', 'd', 'l', 'et', 'a', 'au', 'aux', 'en', 'sur', 'sous'}
ADDR_STOP_EN = {'the', 'of', 'and'}
def _multi_re(tbl):
    keys = [k for k in tbl if ' ' in k]
    return re.compile(r'\b(' + '|'.join(sorted(map(re.escape, keys), key=len, reverse=True)) + r')\b')
_US_MULTI, _IN_MULTI = _multi_re(US_STATES), _multi_re(IN_STATES)
DIGITS_RE = re.compile(r'\d+')

def norm_addr(addr, country):
    """returns (tokens, digit_groups) ; tokens canonicalized, components separated by ','"""
    if addr is None or addr == '':
        return [], []
    s = _indic_map(addr, ADDR_DICT, fallback=True)
    s = _clean_text(s)
    s = s.replace('n\u00b0', ' ').replace('\u00b0', ' ')
    digits = DIGITS_RE.findall(s)
    if country == 'US':
        s = _US_MULTI.sub(lambda m: ' ' + US_STATES[m.group(1)] + ' ', s)
        tbl_state, tbl_street, stop = US_STATES, US_STREET, ADDR_STOP_EN
    elif country == 'India':
        s = _IN_MULTI.sub(lambda m: ' ' + IN_STATES[m.group(1)] + ' ', s)
        tbl_state, tbl_street, stop = IN_STATES, US_STREET, ADDR_STOP_EN
    else:
        tbl_state, tbl_street, stop = {}, FR_STREET, ADDR_STOP_FR
    out = []
    for comp in s.split(','):
        ctoks = []
        for t in _NONALNUM.split(comp):
            if not t or t in ADDR_PREFIX_DROP or t in stop:
                continue
            if t in tbl_state:
                t = tbl_state[t]
                if ' ' in t:
                    ctoks.extend(t.split()); continue
            else:
                t = tbl_street.get(t, t)
            ctoks.append(t)
        if ctoks:
            out.append(' '.join(ctoks))
    return ' , '.join(out).split(' '), digits
