# <p align="center">Weather classification network based on YOLO11<br>基于YOLO11的天气分类网络</p>

This is a weather classification network based on the [YOLO11l-cls](https://docs.ultralytics.com/models/yolo11) model. It is trained using a [custom dataset](https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset). The model after transfer learning is located in the project root directory, named `yolo11-best.pt`.
<br>
这是一个基于[YOLO11l-cls](https://docs.ultralytics.com/models/yolo11)模型的天气分类网络。使用一个[自定义数据集](https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset)训练而成。迁移学习后的模型位于项目根目录，名为 `yolo11-best.pt` 。

- Dataset download link 数据集下载链接

    ```
    https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset
    ```
- About `YOLO11l-cls` model 关于 `YOLO11l-cls` 模型

    ```
    https://docs.ultralytics.com/models/yolo11
    ```

## Environment 运行环境

This network was developed and tested in the following environment:
<br>
这个网络在以下环境开发并测试：

- macOS Tahoe 26.0.1
- Apple M2 chip

With the following package:
<br>
使用以下套件开发：

- Python 3.13.7
- Pytorch 2.8.0
- Ultralytics 8.3.207

## Direct inference 直接使用

If you want to use the trained model to perform inference directly, please refer to the "[Weather classification](#weather-classification-天气分类)" part.
<br>
如果想要使用已经训练好的模型直接进行推理，请参考“[天气分类](#weather-classification-天气分类)”部分。


## Prepare the dataset 准备数据集

After downloading and unzipping the dataset, the dataset structure should look like this:
<br>
下载好并解压数据集之后，数据集的结构应该是这样：

```
multiclass-weather-dataset
├─ alien_test
│    ├─ Cloud_1.png
│    ├─ foggy_10.jpg
│    └─ ...
├─ cloudy
│    ├─ cloudy1.jpg
│    ├─ cloudy10.jpg
│    └─ ...
├─ foggy
│    ├─ foggy1.jpg
│    ├─ foggy10.jpg
│    └─ ...
├─ rainy
│    ├─ rain1.jpg
│    ├─ rain10.jpg
│    └─ ...
├─ shine
│    ├─ shine1.jpg
│    ├─ shine10.jpg
│    └─ ...
├─ sunrise
│    ├─ sunrise1.jpg
│    ├─ sunrise10.jpg
│    └─ ...
└─ test.csv
```

This dataset contains 5 weather conditions, each of which is located in a corresponding folder.
<br>
这个数据集包含5种天气状况，每种天气状况都分别位于对应的文件夹中。

- cloudy
- foggy
- rainy
- shine
- sunrise

These 5 weather conditions and the images in the corresponding folders will be used for training.
<br>
这5种天气条件和对应文件夹内的图片将会被用作训练。

## Split the dataset 分割数据集

To split the dataset, in the `split_dataset.py`, set  `dataset_dir` and `output_dir` parameters:
<br>
为了分割数据集，在 `split_dataset.py` 中修改 `dataset_dir` 和 `output_dir` 参数：

```
dataset_dir = "path/to/dataset"
```

and

```
output_dir = "./weather_split"
```

Then, run `split_dataset.py`. The split dataset will be in the directory specified by `output_dir`.
<br>
然后，运行 `split_dataset.py` 。分割后的数据集将会在 `output_dir` 所指定的目录中。

The structure of the split dataset will be like this:
<br>
分割后的数据集结构看起来将会像这样：

```
weather_split
├── train
│   ├── cloudy
│   │   ├── cloudy1.jpg
│   │   └── ...
│   ├── foggy
│   │   ├── foggy2.jpg
│   │   └── ...
│   ├── rainy
│   │   ├── rainy1.jpg
│   │   └── ...
│   ├── shine
│   │   ├── shine2.jpg
│   │   └── ...
│   └── sunrise
│       ├── sunrise1.jpg
│       └── ...
├── val
│   ├── cloudy
│   │   └── ...
│   ├── foggy
│   │   └── ...
│   ├── rainy
│   │   └── ...
│   ├── shine
│   │   └── ...
│   └── sunrise
│       └── ...
└── test
    ├── cloudy
    │   └── ...
    ├── foggy
    │   └── ...
    ├── rainy
    │   └── ...
    ├── shine
    │   └── ...
    └── sunrise
        └── ...
```

Now, the dataset is ready.
<br>
现在，数据集已经准备好了。

## Training model 训练模型

The training script is provided in the `train.py` file.
<br>
训练模型的代码在 `train.py` 文件内。

Before running `train.py`, you need to check:
<br>
在运行 `train.py` 前，需要检查:

```
dir_dataset = "./weather_split"
```
Please make sure the `dir_dataset` parameter is the path of the split dataset.
<br>
请确保 `dir_dataset` 参数确实指向的是分割后数据集的路径。

Then, run `train.py` to train the model.
<br>
然后，运行 `train.py` 以训练模型。

After training, the best model weights will be saved at:
<br>
训练完成后，最佳的权重文件将会被保存在：

```
runs/classify/train/weights/best.pt
```

And the testing result details, such as confusion matrix, will be saved at:
<br>
测试细节（例如混淆矩阵）会保存在：

```
runs/classify/test_result
```

## Weather classification 天气分类

There are two classification modes provided, and their codes are stored in `infer_one_pic.py` and `infer_folder.py` respectively:
<br>
有两种分类模式，它们的代码分别保存在 `infer_one_pic.py` 和 `infer_folder.py` 中：

- Single image classification 单张图片分类

    Its code is located in `infer_one_pic.py`. Its function is to input a picture, classify the weather in the picture, and output it to the console.
    <br>
    它的代码位于 `infer_one_pic.py` 。它的功能是输入一张图片，分类图片中的天气，并在控制台输出。

- Folder image classification 文件夹图片分类

    Its code is located in `infer_folder.py`. By entering the folder path, classify the weather of the pictures in the folder, and saves the results in a JSON file.
    <br>
    它的代码位于 `infer_folder.py` 中。通过输入文件夹路径，分类文件夹中图片的天气。并将结果保存在JSON文件中。


> For direct inference, by default, the trained model is located in the project root directory and is named `yolo11-best.pt`. Modify the `weights_path` parameter to the path to this model.
> <br>
> 如果想直接使用已经训练好的模型，默认情况下，训练好的模型位于项目根目录，名为 `yolo11-best.pt` 。将 `weights_path` 参数修改为这个模型的路径即可。



### Single image classification 单张图片分类

In the `infer_one_pic.py` file, make sure `weights_path` parameter is the path to the model; `image_path` parameter is the path to the image you want to classify.
<br>
在 `infer_one_pic.py` 文件中，确保 `weights_path` 参数为模型的路径；`image_path` 参数为欲分类的图片的路径。

```
weights_path = "./best.pt"   # Change this to your actual model path
image_path = "./weather_split/test/rainy/rain275.jpg"  # Change this to your image
```

Run `infer_one_pic.py`.
<br>
运行 `infer_one_pic.py`。

The output in console will look like this: 
<br>
在控制台的输出看起来会像这样：

```
```

## Work in progress...