from pathlib import Path
import json

srcRoot = Path("C:\\mldata\\labels\\DSCF3022.MOV")
srcFiles = srcRoot.glob("*.json")

dstRoot = Path("T:\\IMQS8_Data\\ML\\labels\\DSCF3022.MOV")
dstFiles = dstRoot.glob("*.json")

dstNames = {x.name: x for x in dstFiles}
#print(dstNames)

for src in srcFiles:
    with open(src) as f:
        jsrc = json.load(f)
    if jsrc["labels"] is None:
        continue
    polys = []
    for lab in jsrc["labels"]:
        if "poly" in lab:
            polys.append(lab)
    if len(polys) == 0:
        continue

    dst = dstRoot / src.name

    if src.name in dstNames:
        print("merge ", src.name, " to ", dst)
        with open(dst) as f:
            jdst = json.load(f)
        jdst["labels"] += polys
    else:
        print("copy ", src.name, " to ", dst)
        jdst = jsrc

    with open(dst, "w") as f:
        f.write(json.dumps(jdst, indent="    "))
