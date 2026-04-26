import xml.etree.ElementTree as ET

tree = ET.parse(r'e:\school\mypaper\unpacked2\word\document.xml')
root = tree.getroot()

ns = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'o': 'urn:schemas-microsoft-com:office:office',
    'v': 'urn:schemas-microsoft-com:vml',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
}

# parse rels for image mapping
rels_tree = ET.parse(r'e:\school\mypaper\unpacked2\word\_rels\document.xml.rels')
rels_root = rels_tree.getroot()
rel_ns = 'http://schemas.openxmlformats.org/package/2006/relationships'
rels = {}
for rel in rels_root.findall(f'{{{rel_ns}}}Relationship'):
    rels[rel.get('Id')] = rel.get('Target')

def get_image_refs(elem):
    refs = []
    for blip in elem.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
        rid = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
        if rid and rid in rels:
            refs.append(rels[rid])
    for imgdata in elem.iter('{urn:schemas-microsoft-com:vml}imagedata'):
        rid = imgdata.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        if rid and rid in rels:
            refs.append(rels[rid])
    return refs

def has_ole_object(elem):
    for obj in elem.iter('{urn:schemas-microsoft-com:office:office}OLEObject'):
        return True
    return False

def has_math(elem):
    for obj in elem.iter('{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'):
        return True
    for obj in elem.iter('{http://schemas.openxmlformats.org/officeDocument/2006/math}oMathPara'):
        return True
    return False

def extract_math_text(elem):
    parts = []
    for node in elem.iter():
        tag = node.tag.split('}')[-1] if '}' in node.tag else node.tag
        if tag == 't':
            if node.text:
                parts.append(node.text)
    return ''.join(parts)

lines = []
body = root.find('.//w:body', ns)

img_counter = 0
formula_counter = 0

for child in body:
    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

    if tag == 'p':
        parts = []
        para_has_formula = False
        para_img_refs = []

        for elem in child:
            etag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if etag == 'r':
                for t in elem.findall('.//w:t', ns):
                    if t.text:
                        parts.append(t.text)
                refs = get_image_refs(elem)
                if refs:
                    para_img_refs.extend(refs)
                    img_counter += 1
                    parts.append(f'[图片:{refs[0]}]')
                elif has_ole_object(elem):
                    refs2 = get_image_refs(elem)
                    if refs2:
                        para_img_refs.extend(refs2)
                        img_counter += 1
                        parts.append(f'[图片:{refs2[0]}]')
                if has_math(elem):
                    math_text = extract_math_text(elem)
                    if math_text:
                        parts.append(f'[公式: {math_text}]')
                    else:
                        parts.append('[公式]')
                    para_has_formula = True
                    formula_counter += 1

            elif etag == 'hyperlink':
                for r in elem.findall('.//w:t', ns):
                    if r.text:
                        parts.append(r.text)

            elif etag in ('oMath', 'oMathPara'):
                math_text = extract_math_text(elem)
                if math_text:
                    parts.append(f'[公式: {math_text}]')
                else:
                    parts.append('[公式]')
                para_has_formula = True
                formula_counter += 1

        text = ''.join(parts)
        lines.append(text)

    elif tag == 'tbl':
        lines.append('[=== 表格开始 ===]')
        for row in child.findall('.//w:tr', ns):
            cells = []
            for cell in row.findall('.//w:tc', ns):
                cell_parts = []
                for t in cell.findall('.//w:t', ns):
                    if t.text:
                        cell_parts.append(t.text)
                if has_math(cell):
                    math_text = extract_math_text(cell)
                    if math_text:
                        cell_parts.append(f'[公式:{math_text}]')
                refs = get_image_refs(cell)
                if refs:
                    cell_parts.append(f'[图片:{refs[0]}]')
                cells.append(''.join(cell_parts).strip())
            lines.append(' | '.join(cells))
        lines.append('[=== 表格结束 ===]')

while lines and not lines[-1].strip():
    lines.pop()

output = '\n'.join(lines)
with open(r'e:\school\mypaper\doc2_content.txt', 'w', encoding='utf-8') as f:
    f.write(output)

print(f"Total lines: {len(lines)}")
print(f"Images found: {img_counter}")
print(f"Formulas found: {formula_counter}")
print(f"Tables found: {sum(1 for l in lines if '=== ' in l) // 2}")
