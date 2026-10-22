"""Bake luminance soft masks into icon PNG alpha so WebKit renders the icons.

pdftocairo emits each icon as a colour <image> under a <mask> whose content is a
second (greyscale) image run through filter-color-to-alpha. WebKit ignores that
filter inside a mask, so the icons vanish in Safari. Folding the greyscale image
into the colour image's alpha removes the need for the mask.
"""
import base64, io, re, sys
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def img_data(i):
    m = re.search(r'<image id="%s" [^>]*xlink:href="data:image/png;base64,([^"]+)"' % i, s)
    return m, Image.open(io.BytesIO(base64.b64decode(m.group(1))))


mask_re = re.compile(
    r'<mask id="(mask-\d+)">\s*<g filter="url\(#filter-remove-color\)">\s*'
    r'<use xlink:href="#(source-\d+)" filter="url\(#filter-color-to-alpha\)" transform="([^"]+)"/>\s*</g>\s*</mask>\s*')
masks = list(mask_re.finditer(s))
total = len(masks)
print(f'{total} icon masks found', flush=True)
ok = 0
for n, m in enumerate(masks, 1):
    mask_id, alpha_id, tf = m.groups()
    use_re = re.compile(r'<g mask="url\(#%s\)">\s*<use xlink:href="#(source-\d+)" transform="%s"/>\s*</g>'
                        % (mask_id, re.escape(tf)))
    u = use_re.search(s)
    if not u:
        print(f'[{n:2}/{total}] {mask_id}: no matching use, skipped', flush=True)
        continue
    color_id = u.group(1)
    _, a = img_data(alpha_id)
    cm, c = img_data(color_id)
    if a.size != c.size:
        print(f'[{n:2}/{total}] {mask_id}: size mismatch {a.size} vs {c.size}, skipped', flush=True)
        continue
    rgba = c.convert('RGBA')
    rgba.putalpha(a.convert('L'))
    buf = io.BytesIO()
    rgba.save(buf, 'PNG', optimize=True)
    s = s.replace(cm.group(1), base64.b64encode(buf.getvalue()).decode(), 1)
    s = s.replace(u.group(0), '<use xlink:href="#%s" transform="%s"/>' % (color_id, tf), 1)
    s = s.replace(m.group(0), '', 1)
    s = re.sub(r'<image id="%s" [^>]*/>\s*' % alpha_id, '', s, count=1)
    ok += 1
    print(f'[{n:2}/{total}] {mask_id}: {color_id} now carries alpha from {alpha_id}', flush=True)
open(dst, 'w').write(s)
print(f'{ok} fixed, {total - ok} skipped', flush=True)


# Second pass: each icon group also sits under a full-canvas outer mask whose
# alpha only repeats the icon's own shape. WebKit drops the group under it, so
# unwrap those uses and delete the now-unused mask and its 8k-wide image.
s = open(dst).read()
icon_groups = re.findall(r'<g id="(source-\d+)" clip-path="url\(#clip-\d+\)">\s*<g clip-path="url\(#clip-\d+\)">\s*<use xlink:href="#source-\d+" transform="matrix\([\d.]+, 0, 0, [\d.]+,', s)
print(f'{len(icon_groups)} icon groups under outer masks', flush=True)
for n, gid in enumerate(icon_groups, 1):
    u = re.search(r'<g mask="url\(#(mask-\d+)\)">\s*(<use xlink:href="#%s" transform="[^"]+"/>)\s*</g>' % gid, s)
    if not u:
        print(f'[{n:2}/{len(icon_groups)}] {gid}: no outer mask', flush=True)
        continue
    mask_id = u.group(1)
    s = s.replace(u.group(0), u.group(2), 1)
    mm = re.search(r'<mask id="%s">\s*<g filter="url\(#filter-remove-color\)">\s*<use xlink:href="#(source-\d+)"[^>]*/>\s*</g>\s*</mask>\s*' % mask_id, s)
    s = s.replace(mm.group(0), '', 1)
    if s.count('#%s"' % mm.group(1)) == 0:
        s = re.sub(r'<image id="%s" [^>]*/>\s*' % mm.group(1), '', s, count=1)
    print(f'[{n:2}/{len(icon_groups)}] {gid}: unwrapped from {mask_id}', flush=True)
open(dst, 'w').write(s)
