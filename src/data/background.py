from subprocess import call
from glob import glob
import matplotlib.image as mpimg
from matplotlib.pyplot import plt
import pickle
import random
import os
from pathlib import Path
import logging

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


class Backgrounds():
    def __init__(self, background_pck="data/raw/backgrounds/backgrounds.pck"):
        self.background_pck = background_pck
        self._images = pickle.load(open(background_pck, 'rb'))
        self._nb_images = len(self._images)
        logger.info("Nb of images loaded :", self._nb_images)

    def get_random(self, display=False):
        bg = self._images[random.randint(0, self._nb_images-1)]
        if display:
            plt.imshow(bg)
        return bg

    def download(self):
        call(["wget", "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"])
        call(["tar", "xf", "dtd-r1.0.1.tar.gz"])

        dtd_dir = "dtd/images/"
        bg_images = []
        for subdir in glob(dtd_dir+"/*"):
            for f in glob(subdir+"/*.jpg"):
                bg_images.append(mpimg.imread(f))

        logger.info("Nb of images loaded :", len(bg_images))
        logger.info("Saved in :", self.background_pck)

        if not Path(os.path.dirname(self.background_pck)).exists():
            os.mkdirs(os.path.dirname(self.background_pck), exist_ok=True)

        pickle.dump(bg_images, open(self.backgrounds_pck_fn, 'wb'))

        call(["rm", "dtd-r1.0.1.tar.gz"])
        call(["rm", "-r", "dtd"])


if __name__ == '__main__':

    b = Backgrounds()
    b.download()
    b.get_random(display=True)
