from PIL import Image
# from functools import reduce
import numpy as np


# load the bitmap
im = Image.open("/src/sample/test.png").convert('L')
p = np.array(im)
res = []

for r in p:
    for c in r:
        if c==0:
            res.append(1)
        else:
            res.append(0)

print(res)
# print(reduce(lambda x, y: x+y, p))
