from PIL import Image
import numpy as np

class ImageData():
    def __init__(self, filename=None, img_np=None):
        self.pil_img = None
        self.filename = filename
        if self.filename is not None:
            self.pil_img = self.imload(self.filename)
        elif img_np is not None: 
            self.pil_img = self.from_numpy(img_np)
        else:
            raise RuntimeError("No filename or Numpy Image!")
        
        #NOTE: I don't use the PIL "convert" as it doesn't support HSV directly. Instead apply directly to numpy

    def imshow(self, title='Image'):
        #Calculate Histogram for Display
        hist = self.pil_img.convert('L').histogram()

        #loading here to avoid unnecesary dependencies.
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.tight_layout()
        fig.suptitle(title)
        ax1.imshow(self.pil_img)
        ax2.bar(range(len(hist)), hist)
        plt.draw()
        plt.pause(0.001)

    def imsave(self, filename):
        print("Saving image to: {}".format(filename))
        self.pil_img.save(filename)

    def imload(self, filename):
        print("Loading image: {}".format(filename))
        return Image.open(filename)

    def to_numpy(self):
        return np.array(self.pil_img)
    
    def from_numpy(self, img):
        return Image.fromarray(img)
