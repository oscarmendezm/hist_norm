import numpy as np
import colorsys as cs
import cv2

def get_wrapped(img, x, y, wx, wy):
  m, n = img.shape
  rows = (np.arange(x-wx, x+wx) % m) if wx !=0 else x % m
  cols = (np.arange(y-wy, y+wy) % n) if wy !=0 else y % n
  return img[rows][:, cols]

class HistEQ():
    def __init__(self, params):
        self.params = params
    
    def equalize(self, img):

        ###Prepare Image###
        #if we want to do colour, convert to HSV and keep value/luminance
        if self.params['colour'] and img.shape[-1]==3: #ensure we actually have RGB
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_eq = img_hsv[:,:,2].astype(img.dtype)
        else:
            #keep image if its alread bw otherwise convert to grayscale
            img_eq = img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)      
            img_eq = img_eq.astype(img.dtype)

        ###Apply EQ###
        if self.params['adaptive']:
            img_out = self.equalise_img_adaptive(img_eq)
        else:
            img_out = self.equalize_img(img_eq)

        ###Restore V for HSV Image and Convert Back###
        if self.params['colour'] and img.shape[-1]==3:
            img_hsv[:,:,2] = img_out
            img_out = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            
        return img_out.astype(img.dtype)

    def get_eq_params(self, img):
        L = np.iinfo(img.dtype).max # Get the max value of the input image
        cdf = self.get_cdf(img)
        cdf_min, denom = self.get_cdf_params(cdf)
        return [L, cdf, cdf_min, denom] # Add eps to prevent divide by zero
    
    def get_cdf_params(self, cdf):
        cdf_min = np.min(cdf[np.nonzero(cdf)])
        cdf_max = np.max(cdf[np.nonzero(cdf)])
        denom = (cdf_max - cdf_min) + 1e-5  # Wikipedia uses (W*H) - cdf_min as the denominator, but it relies on CDF max being equal to W*H
                                            # add small epsilon to prevent divide by zero, when e.g. all pixel values are the same so cdf_min == cdf_max as we ignore zero values
        return [cdf_min, denom]
    
    def apply_eq(self, val, L, cdf, cdf_min, A): #applies eq to either a single pixel or a np.array of them
        return (((cdf[val]-cdf_min)/(A))) * (L)

    def equalize_img(self, img_eq):
        eq_params = self.get_eq_params(img_eq)
        return self.apply_eq(img_eq, *eq_params)
    
    def equalise_img_adaptive(self,img_eq):
        #get some parameters
        W,H = img_eq.shape[0:2]
        d = self.params['window']

        #Estimate Adaptive Histnorm
        img_out = img_eq.copy() #Make a copy to avoid in-place modifications
        for i in range(0,W):
            #Get first patch and estimate Params, these will be updated as we move across the row
            p_0 = get_wrapped(img_eq, i, 0, d, d) #get 0th patch on this row
            eq_0 = self.get_eq_params(p_0) #get 0th eq params
            img_out[i][0] = self.apply_eq(img_eq[i][0], *eq_0) #equalize 0th pixel 
            for j in range(1,H): 
                #Get Changed Columns
                first_col = get_wrapped(img_eq, i, j-d-1, d, 0) # Get the first column in the current "patch"
                last_col = get_wrapped(img_eq, i, j+d-1, d, 0) #get the last column in the next "patch"

                #Get CDF for Changed columns
                cdf_f = self.get_cdf(first_col) #get cdf params for first column
                cdf_l = self.get_cdf(last_col) #get cdf params for last column

                #Update CDF
                eq_0[1] = eq_0[1] - cdf_f + cdf_l # Update CDF: Take CDF, remove first column, add last row, should be equal to accumulating histograms
                eq_0[2], eq_0[3] = self.get_cdf_params(eq_0[1]) # Update min and denom
                
                #equalize current pixel
                img_out[i][j] = self.apply_eq(img_eq[i][j], *eq_0)

        return img_out

    def get_cdf(self, img):
        hist, bin_edges = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum() 
        return cdf
