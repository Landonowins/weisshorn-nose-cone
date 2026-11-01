import re


def sort_xy(src, dst, label):
    """Rewrite a Fluent XY file with its points sorted by x and the curve renamed."""
    text = open(src).read()
    head, _, rest = text.partition("((xy/key/label")
    body = rest.split("\n", 1)[1].rsplit(")", 1)[0]
    pts = sorted((float(a), float(b)) for a, b in (l.split() for l in body.strip().splitlines()))
    with open(dst, "w") as f:
        f.write(head)
        f.write('((xy/key/label "%s")\n' % label)
        f.write("".join("%.6g\t%.6g\n" % p for p in pts))
        f.write(")\n")
