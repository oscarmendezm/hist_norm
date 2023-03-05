# Histogram Equalization Example
This is a simple implementation of histogram equalisation. The program works by loading an image, and applying both Global and Sliding Window Adaptive Histogram Equalisation. The program works on either a single image or a directory of images, and will output the equalised images as PNG files, along with a PLT display of the histogram equalisation.

## Installation
The program requires 3 libraries: 
1) Pillow (Basic Image IO)
2) OpenCV (Colorspace Manipulation)
3) MatplotLib (Display)

The program is trivially modifiable to only require OpenCV, or indeed no library, but the current implementation allows the most flexibility for displaying results.

Easy installation is done usign Anaconda.

Create a new Anaconda environment: 
```
conda create --name histnorm python==3.10.9
conda activate histnorm
conda install opencv pillow matplotlib
```

## How to Run:

The simplest version of the code runs as 

```
python main.py --file <FILENAME>
```

for example:
```
python main.py --file ./data/SIPI/4.2.07.tiff
```

alternatively, an entire directory can be passed to the program:

```
python main.py --file ./data/SIPI/
```
which will find all files in the directory and attempt to apply histogram equalisation.

## Notes
There was little consideration given to runtime performance. The program focuses on performing an easy to understand version of the algorithms. Using numpy more efficiently, extracting the adaptive histogram equalisation to c/c++/java or even using PyTorch's GPU bindings would siginificantly improve performance.
Similarly, there was no implementation of Contrast Limiting, which would improve the resulsting SWAHE image.

