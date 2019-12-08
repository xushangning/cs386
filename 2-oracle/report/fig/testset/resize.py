import os
from PIL import Image

if __name__ == "__main__":
  fnames = os.listdir()
  for fname in fnames:
      if fname == 'resize.py': continue
      img = Image.open(fname)
      img = img.resize((64, 64), Image.ANTIALIAS)
      img.save(fname)
