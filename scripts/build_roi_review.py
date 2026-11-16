"""Render deterministic ROI overlays, overview sheets and an offline review editor.

Usage: python scripts/build_roi_review.py --config configs/camera_rois.draft.json
No inference configuration is changed by this tool.
"""
import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#36e8ae', '#5ab5ff', '#ffca62', '#e694ff']


def font(size):
    path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def area(points):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1]))) / 2


def simple_polygon(points):
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def on_segment(a, b, p):
        return (abs(orient(a, b, p)) < 1e-10
                and min(a[0], b[0])-1e-10 <= p[0] <= max(a[0], b[0])+1e-10
                and min(a[1], b[1])-1e-10 <= p[1] <= max(a[1], b[1])+1e-10)

    def crosses(a, b, c, d):
        return ((orient(a, b, c)*orient(a, b, d) < 0
                 and orient(c, d, a)*orient(c, d, b) < 0)
                or on_segment(a, b, c) or on_segment(a, b, d)
                or on_segment(c, d, a) or on_segment(c, d, b))

    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            if crosses(points[i], points[(i+1) % n], points[j], points[(j+1) % n]):
                return False
    return True


def validate(config):
    seen = set()
    for camera in config['cameras']:
        assert camera['camera_id'] not in seen, 'Duplicate camera ID'
        seen.add(camera['camera_id'])
        assert (ROOT / camera['reference_image']).is_file(), camera['reference_image']
        w,h = camera['reference_size']
        assert w > 0 and h > 0
        for region in camera['regions']:
            pts = region['points_normalized']
            assert len(pts) >= 3 and all(0<=x<=1 and 0<=y<=1 for x,y in pts)
            assert area(pts) > .0001, 'Degenerate polygon'
            assert simple_polygon(pts), f"Self-intersection: {camera['camera_id']}"


def overlay(camera):
    image = Image.open(ROOT / camera['reference_image']).convert('RGBA')
    layer = Image.new('RGBA',image.size)
    draw = ImageDraw.Draw(layer)
    w,h = image.size
    for i,region in enumerate(camera['regions']):
        points = [(round(x*w),round(y*h)) for x,y in region['points_normalized']]
        color = COLORS[i%len(COLORS)]
        rgb = tuple(int(color[j:j+2],16) for j in (1,3,5))
        draw.polygon(points, fill=rgb+(40,))
        draw.line(points+[points[0]],fill=rgb+(255,),width=2)
        for x,y in points:
            draw.ellipse((x-2,y-2,x+2,y+2),fill=rgb+(255,))
        # Label near the bottom-most edge, clamped to image extent.
        x,y = max(points,key=lambda p:p[1])
        x,y = min(max(x,5),w-38), min(max(y-18,20),h-20)
        draw.rectangle((x-2,y-2,x+29,y+15),fill=(12,20,30,230))
        draw.text((x,y),f'R{i+1}',font=font(12),fill=rgb+(255,))
    return Image.alpha_composite(image,layer).convert('RGB')


def build(config, output):
    validate(config)
    output.mkdir(parents=True,exist_ok=True)
    previews = output / 'previews'
    previews.mkdir(exist_ok=True)
    browser = json.loads(json.dumps(config))
    for camera,bc in zip(config['cameras'],browser['cameras']):
        image = overlay(camera)
        image.save(previews / f"{camera['index']:02d}_{camera['camera_id']}.jpg",quality=95)
        source = (ROOT / camera['reference_image']).read_bytes()
        bc['image_data'] = 'data:image/jpeg;base64,'+base64.b64encode(source).decode()
    for start in range(0,len(config['cameras']),9):
        group = config['cameras'][start:start+9]
        sheet = Image.new('RGB',(min(3,len(group))*512,((len(group)+2)//3)*380),'#111925')
        d = ImageDraw.Draw(sheet)
        for i,camera in enumerate(group):
            x,y = (i%3)*512,(i//3)*380
            name = camera['name'].replace('_',' ')
            d.text((x+8,y+7),f"{camera['index']:02d}  {name}",font=font(15),fill='white')
            sheet.paste(overlay(camera),(x,y+34))
            d.text((x+8,y+327),f"{len(camera['regions'])} ROI | DỰ THẢO | {camera['confidence']}",font=font(14),fill='#ffce75')
            d.text((x+8,y+350),camera['camera_id'],font=font(12),fill='#aab5c5')
        sheet.save(output / f'overview_{start//9+1:02d}.jpg',quality=95)
    template = (Path(__file__).with_name('roi_review_template.html')).read_text()
    # JSON cannot close the data script, even if camera names contain HTML.
    encoded = json.dumps(browser,ensure_ascii=False).replace('<','\\u003c')
    (output / 'review.html').write_text(template.replace('__ROI_DATA__',encoded),encoding='utf-8')
    print('Review:',output / 'review.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/camera_rois.draft.json')
    p.add_argument('--output',type=Path,default=ROOT/'reports/roi_review')
    args=p.parse_args()
    build(json.loads(args.config.read_text()),args.output)
