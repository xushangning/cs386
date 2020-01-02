from pathlib import Path

import cv2 as cv

input_dir = Path('pred_results')

for p in input_dir.glob('**/*.png'): 
   img = cv.imread(str(p)) 
   res = cv.resize(img, None, fx=4, fy=4, interpolation=cv.INTER_CUBIC) 
   cv.imwrite(str(p), res) 
