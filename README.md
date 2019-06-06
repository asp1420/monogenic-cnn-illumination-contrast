# A Bio-inspired Monogenic CNN Layer for Illumination-Contrast Invariance

## Abstract
Deep learning (DL) is attracting considerable interest as it currently achieves remarkable performance in many branches of science and technology. However, current DL cannot guarantee capabilities of the mammalian visual systems such as lighting changes and rotation equivariance. This work proposes an entry layer capable of classifying images even with low contrast conditions. We achieve this by means of an improved version of monogenic wavelets inspired by some physiological experimental results (such as the stronger response of primary visual cortex to oriented lines and edges). We have simulated the atmospheric degradation of the CIFAR-10 and the Dogs and Cats datasets to generate realistic illumination degradations of the images. The most important result is that the accuracy gained by using our layer is substantially more robust to illumination changes than nets without such a layer.

Some evidence supports that the depth (number of layers), and the data augmentation during the training process, can occasionally provide invariance or equivariance relative to some class of transformations, the reasons for that behaviour does not seem to be well understood. Some investigations indicate that the learning of an invariance response may fail even with very deep Convolutional Neural Networks or by large data augmentations in the training.

To overcome these shortcomings, one idea is to embrace suitable geometric methods where the main techniques are real algebraic varieties and methods of computer algebra. The main strong points for Geometric Deep Learning is the long history of achievements in a great variety of fields that it includes the complex numbers and the quaternions as very special cases; and the fact that there is a well-developed theory of Geometric Calculus (GC) wavelets with the potential to be applied to Deep Learning (DL) much as scalar wavelets are used in current DL techniques. An additional bonus of GC is that the representation of the signals occurs in a higher dimensional space and hence they provide naturally a stronger discrimination capacity.

In this work, as the first step in this general strategy, we work with Hamilton’s quaternions H, which is the most straightforward geometric calculus beyond the complex numbers C. The main results are the design and implementation of a CNN layer (M6) based on the monogenic signal. These layers substantially enhance the invariance response to illumination-contrast. As we will see, they reproduce to a good extent characteristic properties of the mammal primary cortex V1.

Up till now, quaternions have been used with fully connected NNs and, more recently, with CNNs. In this context, the method proposed in this work is the first, to the best of our knowledge, that combines a CNN with local phase computations using quaternions.

On the experimental side, to evaluate the predictive performance of M6, we have simulated illumination-contrast changes using the atmospheric degradation model over two image-datasets, the CIFAR-10 and the Dogs and Cats.

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
