# 2-Oracle
Please organize the project folder for 2-oracle like this:
```
2-oracle
|
|----code
|       |----models
|       |...
|
|----report
|       |...
|
|----dataset
        |----0102
        |----0103
        |----0104
        |...
```
DO NOT ORGANIZE CODE WITHIN ANOTHER LAYER OF FOLDERS IN THE "CODE" FOLDER, OTHERWISE THE "Dataset" CLASS MAY NOT WORK.

## Source Code
The main source code files include the follows:
- **OBC_Recognition_app.py**: The main application.
- **Dataset.py**: The code for loading data and data augmentation.
- **OracleNet.py**: The code for network structure and network training.
- **template_matching.py**: The code implementing template matching.
- **utils.py**: Some utility functions.

## Run Application
We have trained models saved in the directory `./code/models`.
To test our DNN models, you can run the following command:
```sh
python OBC_Recognition_app.py \
    --input_folder=test_images \
    --model_path=weights_norm_cat[10/40]_v2.hdf5 \
    --method=DNN \
    --num_cat=[10/40]
```

To test the template matching model, you can run the following command:
```sh
python OBC_Recognition_app.py \
    --input_folder=test_images \
    --method=TM \
    --tm_method=MSE \
    --num_cat=[10/40]
```

An output.csv file will be generated containing the classification results after running the above commands.

For more details of the application, please run the command `python OBC_Recognition_app.py --help`.

## Dependencies
Our application requires Python 3.x, and following libraries are required:
- Numpy
- Pandas
- Tensorflow
- Keras
- Sklearn
- Matplotlib
- PIL
- Imgaug
- OpenCV
