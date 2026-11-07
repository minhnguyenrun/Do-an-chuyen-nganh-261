import re
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from difflib import SequenceMatcher
from shapely.geometry import Polygon, box
from ..spatial.coordinates import polygon_bbox


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.rows, self.cells, self.text = [], [], [], []
        self.valid = True
        self.tables = 0

    def handle_starttag(self,tag,attrs):
        if tag in ('table','thead','tbody','tfoot','tr','td','th'):
            if tag == 'table':
                self.tables += 1
            if tag == 'tr':
                if not self.stack or self.stack[-1] not in ('table','thead','tbody','tfoot'):
                    self.valid = False
                self.rows.append([])
            if tag in ('td','th'):
                if not self.stack or self.stack[-1] != 'tr' or not self.rows:
                    self.valid = False
                else:
                    attributes = dict(attrs)
                    try:
                        span = int(attributes.get('colspan','1'))
                        rowspan = int(attributes.get('rowspan','1'))
                    except ValueError:
                        span,rowspan = 0,0
                    # Span geometry needs a richer grid matcher: abstain for now.
                    if span != 1 or rowspan != 1:
                        self.valid = False
                    self.rows[-1].append('')
            self.stack.append(tag)

    def handle_endtag(self,tag):
        if tag in ('table','thead','tbody','tfoot','tr','td','th'):
            if not self.stack or self.stack[-1] != tag:
                self.valid = False
            else:
                self.stack.pop()

    def handle_data(self,data):
        self.text.append(data)
        if self.stack and self.stack[-1] in ('td','th') and self.rows and self.rows[-1]:
            self.rows[-1][-1] += data


def parse_table(content):
    parser = TableParser()
    parser.feed(content)
    parser.close()
    return parser


def extract_table_features(html_content,line_count):
    parser = parse_table(html_content)
    counts = [len(row) for row in parser.rows]
    valid = parser.valid and not parser.stack and parser.tables == 1 and bool(counts) and min(counts)>0 and len(set(counts)) == 1
    cells = sum(counts)
    return dict(is_valid_html=float(valid),row_count=float(len(counts)),col_count=float(max(counts,default=0)),
                cell_count=float(cells),cell_density=min(1.,cells/max(1,line_count)))


def extract_formula_features(content):
    stack = []
    pairs = {'}':'{',']':'[',')':'('}
    balanced = True
    for char in content:
        if char in '{[(':
            stack.append(char)
        elif char in '}])':
            if not stack or stack.pop() != pairs[char]:
                balanced = False
    tokens = re.findall(r'\\(?:frac|sum|int|sqrt|prod|lim)|[=+^_∑∫√]',content)
    return dict(is_balanced=float(balanced and not stack),has_math_tokens=float(bool(tokens)),
                operator_density=len(tokens)/max(1,len(content)))


def plain_text(content):
    return re.sub(r'<[^>]*>',' ',content)


def calculate_content_consistency(probe_text,line_transcripts):
    normalize = lambda s: re.sub(r'\s+','',plain_text(s)).casefold()
    expected = normalize(' '.join(line_transcripts))
    if not expected:
        return 0.0  # Absence of independent evidence is not perfect agreement.
    return SequenceMatcher(None,normalize(probe_text),expected,autojunk=False).ratio()


def formula_content_consistency(content,transcripts):
    def tokens(text):
        text = unicodedata.normalize('NFKC',text)
        for command,symbol in {'sum':'∑','int':'∫','sqrt':'√','alpha':'α','beta':'β','gamma':'γ',
                               'theta':'θ','pi':'π','times':'*','cdot':'*','leq':'≤','geq':'≥'}.items():
            text = re.sub(r'\\'+command+r'\b',lambda _:symbol,text)
        text = re.sub(r'\\(?:frac|dfrac|tfrac|left|right|mathrm|mathbf|text|displaystyle)\b','',text)
        return Counter(re.findall(r'[A-Za-zα-ωΑ-Ω]|\d+|[∑∫√=+*≤≥-]',text))
    predicted,expected = tokens(content),tokens(' '.join(transcripts))
    common = sum((predicted & expected).values())
    return 2*common/max(1,sum(predicted.values())+sum(expected.values()))


def spatial_grid(lines):
    """Cluster actual line boxes into rows, then verify repeated column centers."""
    boxes = sorted((polygon_bbox(l.polygon) for l in lines),key=lambda b:(b[1],b[0]))
    rows = []
    for b in boxes:
        cy = (b[1]+b[3])/2
        row = next((row for row in rows if abs(cy-row[0]) <= max(3,(b[3]-b[1])*.5)),None)
        if row is None:
            rows.append([cy,[b]])
        else:
            row[1].append(b)
    ordered = [sorted(row[1]) for row in rows]
    if len(ordered) < 2 or min(map(len,ordered),default=0) < 2 or len(set(map(len,ordered))) != 1:
        return 0,0,0.0
    width = max(b[2] for b in boxes)-min(b[0] for b in boxes)
    for col in zip(*ordered):
        centers = [(b[0]+b[2])/2 for b in col]
        if max(centers)-min(centers) > max(5,width*.08):
            return len(rows),len(ordered[0]),0.0
    return len(rows),len(ordered[0]),1.0


def table_alignment(content,lines):
    rows,cols,alignment = spatial_grid(lines)
    features = extract_table_features(content,len(lines))
    return alignment if (features['is_valid_html'] and rows == features['row_count'] and cols == features['col_count']) else 0.0
