import os
from pathlib import Path
import PIL
from PIL import Image


def is_layer_png(level):
    return level <= 22


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
    all = list(all)
    for idst in range(len(all)):
        dst = all[idst]
        if idst % 100 == 0:
            print("{}/{}\r".format(idst, len(all)), end="", flush=True)
        dst_x, dst_y = dst
        src_x, src_y = dst
        src_x, src_y = src_x * 2, src_y * 2
        dst_path = str(out_root / str(dst_x) / str(dst_y)) + ('.png' if is_layer_png(out_level) else '.jpeg')
        dst_stat = None
        try:
            dst_stat = os.stat(dst_path)
        except:
            pass
        n_src_newer = 0
        # Count the number of tiles where the source is newer than the destination (ie invalid, needs to be downsampled)
        if dst_stat is None:
            # Just make it an arbitrary non-zero value, if dst does not yet exist
            n_src_newer = 1
        else:
            for cy in range(2):
                for cx in range(2):
                    try:
                        src_stat = os.stat(str(in_root / str(src_x + cx) / str(src_y + cy)) + src_ext)
                        if src_stat.st_mtime > dst_stat.st_mtime:
                            n_src_newer += 1
                    except:
                        # assume that any exception to stat(src) means the source does not exist, which is fine (equivalent to a blank tile)
                        pass

        if n_src_newer == 0:
            # destination has already been built, and is newer than src
            continue

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
            out.save(dst_path, compress_level=1)
        else:
            out = out.convert('RGB')
            out.save(dst_path, quality=90)


for level in range(25, 1, -1):
    print("Downsampling level {}".format(level))
    downsample('/home/ben/inf/webtiles', level)