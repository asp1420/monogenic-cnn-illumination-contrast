# A Bio-inspired Monogenic CNN Layer for Illumination-Contrast Invariance

## Abstract
Deep learning (DL) is attracting considerable interest as it currently achieves remarkable performance in many branches of science and technology. However, current DL cannot guarantee capabilities of the mammalian visual systems such as lighting changes and rotation equivariance. This work proposes an entry layer capable of classifying images even with low contrast conditions. We achieve this by means of an improved version of monogenic wavelets inspired by some physiological experimental results (such as the stronger response of primary visual cortex to oriented lines and edges). We have simulated the atmospheric degradation of the CIFAR-10 and the Dogs and Cats datasets to generate realistic illumination degradations of the images. The most important result is that the accuracy gained by using our layer is substantially more robust to illumination changes than nets without such a layer.

## More content of the paper
(TODO)

## Dataset
The dataset is available in the public bucket:
 *  gs://um-bucket-2019/cifar_haze_levels_32_june_2019/
 *  gs://um-bucket-2019/cad_haze_levels_224_june_2019/
 
In order to download the dataset the [`gsutil tool`](https://cloud.google.com/storage/docs/gsutil) will be required.

You can download the dataset as follows:
 
 ```
 > gsutil -m cp -r gs://um-bucket-2019/cifar_haze_levels_32_june_2019/ .
 ```
 
 Dataset `cifar_haze_levels_32_june_2019` contains four hdf5 files ... (TODO)
 Dataset `cad_haze_levels_224_june_2019` contains four hdf5 files ... (TODO)
 
 ## Test
 (TODO)
