# 3-UHD
Please organize the project folder for 3-UHD like this:
```
3-UHD
|----code
|    |
|    |----images
|    |      |----4k (original "homework" download)
|    |      |----720P
|    |      |----1080P
|    |
|    |----faker.m
|    |
|    |...
|
|----report
     |...
```
## Run Application
We provide both applications for the simple thresholding classification and SVM classification.

The following command runs a simple threshold classification on a folder of images using area average reference interpolation and a reference rate of 2.
It takes ??? seconds per image.
```sh
python DCT_Judger.py \
     --input_folder=image/folder \
     --ref_method=AR \
     --ref_rate=2 \
     --mask_method=L1
```
The `mask_method` specifies which of the L1 and L2 energy spectrums to use.
We provide following pairs of references:
| Reference Method | Reference Rate |
| :---: | :---: |
| AR | 2 |
| NN | 2 |
| BL | 2 |
| BC | 2 |
| AR | 3 |

For further information, run `python DCT_Judger.py --help` to see more optional arguments.

The following command runs the SVM classifier stored in `svm_complete.model` to classify images in the given folder.
It takes ??? seconds per image.
```
python SVM_app.py \
     --input_folder=image/folder
```

An output.csv file will be generated containing the classification results after running the above commands.

## Dependencies
Our application requires Python 3.x, and following libraries are required for the core programs:
- Numpy
- Pandas
- Sklearn
- Matplotlib
- OpenCV
