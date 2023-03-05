import numpy as np
import img_load as iml
import hist_norm as hst
import cv2
from pathlib import Path

def do_eq_cv2(I, params):
    img = I.to_numpy()
    img_bw = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_eq = cv2.equalizeHist(img_bw).astype(np.uint8)
    Ieq = iml.ImageData(img_np=img_eq)
    Ieq.imshow('OpenCV')
    Ieq.imsave(params['output_file']+"_cv2.png")
    return img_eq

def do_eq(I, params):
    #Create Equalizer
    h = hst.HistEQ(params)

    #Equalize
    img = I.to_numpy()
    imgh = h.equalize(img)

    #Create new Image for Display
    Ieq = iml.ImageData(img_np=imgh)
    Ieq.imshow()

    #Get Filename based on params and Save
    filename = params['output_file']
    for k, v in params.items():
        if v is True:
            filename+="_{}".format(k) 
    filename+=".png"
    Ieq.imsave(filename)
    return imgh

def hist_equalise(input_filename, output_filename):
    I = iml.ImageData(input_filename)
    I.imsave(output_filename+".png") #Save as PNG
    I.imshow()

    # Perform Global Histogram Equalisation (BW)
    params = {'colour': False,
              'adaptive': False,
              'window': 16,
              'output_file': output_filename+"_eq"}
    eq_me = do_eq(I, params)

    # Perform SW Adaptive Histogram Equalisation (BW)
    params['adaptive'] = True
    params['colour'] = False
    do_eq(I, params)

    # Perform Global Histogram Equalisation (Colour)
    params['adaptive'] = False
    params['colour'] = True
    do_eq(I, params)

    # Perform SW Adaptive Histogram Equalisation (Colour)
    params['adaptive'] = True
    params['colour'] = True
    do_eq(I, params)

    #Sanity Check Error vs. OpenCV (Global BW)
    eq_cv = do_eq_cv2(I, params)
    diff = np.abs(eq_cv - eq_me)
    print("Average Photometric Error: {}".format(np.mean(diff.flatten())))

def main(args):
    filename = Path(args.file)
    if filename.is_dir():
        files = filename.glob("*")
        for file in files:
            output_filename = file.with_suffix("")
            hist_equalise(file.as_posix(), output_filename.as_posix())
            
    elif filename.is_file():
        file = filename
        output_filename = file.with_suffix("")
        hist_equalise(file.as_posix(), output_filename.as_posix())
    else:
        raise RuntimeError("Input is neither a file nor a directory.")


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="./data/SIPI/7.1.07.tiff", help="Image or Directory of Images in which to apply Equalisation")
    args = parser.parse_args()
    main(args)
    print("Done.")