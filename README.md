# A Bio-inspired Monogenic CNN Layer for Illumination-Contrast Invariance

## Introduction
Deep learning (DL) is attracting considerable interest as it currently achieves remarkable performance in many branches of science and technology. However, current DL cannot guarantee capabilities of the mammalian visual systems such as lighting changes and rotation equivariance. This work proposes an entry layer capable of classifying images even with low contrast conditions. We achieve this by means of an improved version of monogenic wavelets inspired by some physiological experimental results (such as the stronger response of primary visual cortex to oriented lines and edges). We have simulated the atmospheric degradation of the CIFAR-10 and the Dogs and Cats datasets to generate realistic illumination degradations of the images. The most important result is that the accuracy gained by using our layer is substantially more robust to illumination changes than nets without such a layer.

Some evidence supports that the depth (number of layers), and the data augmentation during the training process, can occasionally provide invariance or equivariance relative to some class of transformations, the reasons for that behaviour does not seem to be well understood. Some investigations indicate that the learning of an invariance response may fail even with very deep Convolutional Neural Networks or by large data augmentations in the training.

To overcome these shortcomings, one idea is to embrace suitable geometric methods where the main techniques are real algebraic varieties and methods of computer algebra. The main strong points for Geometric Deep Learning is the long history of achievements in a great variety of fields that it includes the complex numbers and the quaternions as very special cases; and the fact that there is a well-developed theory of Geometric Calculus (GC) wavelets with the potential to be applied to Deep Learning (DL) much as scalar wavelets are used in current DL techniques. An additional bonus of GC is that the representation of the signals occurs in a higher dimensional space and hence they provide naturally a stronger discrimination capacity.

In this work, as the first step in this general strategy, we work with Hamilton’s quaternions H, which is the most straightforward geometric calculus beyond the complex numbers C. The main results are the design and implementation of a CNN layer (M6) based on the monogenic signal. These layers substantially enhance the invariance response to illumination-contrast. As we will see, they reproduce to a good extent characteristic properties of the mammal primary cortex V1.

Up till now, quaternions have been used with fully connected NNs and, more recently, with CNNs. In this context, the method proposed in this work is the first, to the best of our knowledge, that combines a CNN with local phase computations using quaternions.

On the experimental side, to evaluate the predictive performance of M6, we have simulated illumination-contrast changes using the atmospheric degradation model over two image-datasets, the CIFAR-10 and the Dogs and Cats.

## Dataset
The dataset is available in the public bucket:
 *  gs://um-bucket-2019/cifar_haze_levels_32_june_2019/
 *  gs://um-bucket-2019/cad_haze_levels_224_june_2019/
 
In order to download the dataset the [`gsutil tool`](https://cloud.google.com/storage/docs/gsutil) will be required.

You can download the dataset as follows:
 
 ```
 > gsutil -m cp -r gs://um-bucket-2019/cifar_haze_levels_32_june_2019/ .
 ```
 
 Each folder contains 4 hdf5 files in which illumination-contrast changes (atmospheric degradation model) were applied. Each file has different levels of degradation:

 * No atmospheric degradation level (0_3)
 * Atmospheric degradation level 1 (1_3)
 * Atmospheric degradation level 2 (3_3)
 * Atmospheric degradation level 3 (3_3)

Yo can see some examples of images below.

![](images/atmosphericdegradation.png?raw=true)

Randomly degraded versions of the images in Figure 5: (a) From CIFAR-10; (b) From Dogs and Cats.

![](images/oneimagedegradation.png?raw=true)

From left to right the different levels of atmospheric degradation are shown.
 
## Scripts
There are 6 different scripts in the [models](models) folder from which the results of this work were obtained.

**Important**: for a correct execution of the scripts these require two files, [hdf5_utilities.py](tools/hdf5_utilities.py) and [monogenic_functions.py](tools/monogenic_functions.py). Both are available in the [tools](tools) folder. These files must be in the same directory of each model below. Our recommendation is just to copy-paste these files into the directory of the model to be tested.

1. [CIFAR-10 No Monogenic](models/cifar_models/cifar_rgb_haze.py). Training and validation by using CIFAR-10 dataset with atmospheric degradation without applying monogenic wavelets.
2. [CIFAR-10 Monogenic](models/cifar_models/cifar_rgb_haze_monogenic.py). Training and validation by using CIFAR-10 dataset with atmospheric degradation and monogenic wavelets.
3. [Dogs and Cats No Monogenic](models/cads_models/cads_rgb_haze.py). Training and validation by using Dogs and Cats dataset with atmospheric degradation without applying monogenic wavelets.
4. [Dogs and Cats Monogenic](models/cads_models/cads_rgb_haze_monogenic.py). Training and validation by using Dogs and Cats dataset with atmospheric degradation and monogenic wavelets.
5. [CIFAR-10 No Monogenic and Resnet V2](models/cifar_resnet20_models/cifar_rgb_haze_resnet.py). Training and validation by using CIFAR-10 dataset model with atmospheric degradation without applying monogenic wavelets. ResNet V2 model is used.
6. [CIFAR-10 Monogenic and Resnet V2](models/cifar_models/cifar_rgb_haze_monogenic.py). Training and validation by using CIFAR-10 dataset with atmospheric degradation and monogenic wavelets. ResNet V2 model is used.

## Usage
For a correct execution we recommend the following tree structure:

```
/                                [Directory]
  cifar_rgb_haze.py              [File]
  hdf5_utilities.py              [File]
  monogenic_functions.py         [File]
  data/                          [Directory]
       cifar_haze_0_3-32-32.h5   [File]
```

In order to test the models you need to give some parameters:

 * Job ID (Integer number, i.e 1)
 * Epochs (Integer number, i.e 100)
 * Batch size (Integer number, i.e 32)
 * Learning rate (Float, i.e 0.0001)

For example:
```
> cifar_rgb_haze.py 1 100 32 0.0001
```

The results will be stored in an auto generated folder called *job_out*.
