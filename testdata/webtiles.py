from pathlib import Path
import PIL
from PIL import Image


def is_layer_png(level):
    return level <= 11


def downsample(root, in_level):
    out_level = in_level - 1
    in_root = Path(root) / str(in_level)
    out_root = Path(root) / str(out_level)
    out_root.mkdir(exist_ok=True)
    all = set()
    src_ext = ".png" if is_layer_png(in_level) else ".jpeg"
    for fn in in_root.glob('**/*' + src_ext):
        x = int(fn.parent.name)
        y = int(fn.stem)
        all.add((x // 2, y // 2))
    for dst in all:
        dst_x, dst_y = dst
        src_x, src_y = dst
        src_x, src_y = src_x * 2, src_y * 2
        out = Image.new('RGBA', (256, 256))
        nsrc = 0
        for cy in range(2):
            for cx in range(2):
                try:
                    src = Image.open(str(in_root / str(src_x + cx) / str(src_y + cy)) + src_ext)
                    src = src.resize((128, 128), PIL.Image.BOX)
                    out.paste(src, (cx * 128, cy * 128))
                    nsrc += 1
                except:
                    pass
        if nsrc == 0:
            raise ValueError('Expected at least 1 source tile for target tile {},{}'.format(dst_x, dst_y))
        (out_root / str(dst_x)).mkdir(exist_ok=True)
        if is_layer_png(out_level):
            out.save(str(out_root / str(dst_x) / str(dst_y)) + '.png', compress_level=1)
        else:
            out = out.convert('RGB')
            out.save(str(out_root / str(dst_x) / str(dst_y)) + '.jpeg', quality=90)


for level in range(15, 1, -1):
    print("Downsampling level {}".format(level))
    downsample('/home/ben/inf/webtiles', level)