from psslib import make_spectrum_cube as makesc
from settings import settings, tuning_settings

import numpy as np


class Tools(object):

    def __init__(self, spectrum_cube):
        self.spectrum_cube = spectrum_cube.spectrum_cube
        self.material = spectrum_cube.material
        self.comp = spectrum_cube.comp

    def reshape(self,
                height=tuning_settings.hyper_img_height,
                width=tuning_settings.hyper_img_width,
                depth=len(settings.wavelength),
                h_start=tuning_settings.hyper_img_height_start,
                w_start=tuning_settings.hyper_img_width_start) -> makesc.SpectrumCube:

        reshape_spectrum_cube = np.empty([height, width, depth], dtype=np.uint8)

        for i in range(depth):
            img = self.spectrum_cube[:, :, i]
            img = img[
                  h_start: h_start + height,
                  w_start: w_start + width,
                  ]
            reshape_spectrum_cube[:, :, i] = img

        return makesc.SpectrumCube(spectrum_cube=reshape_spectrum_cube,
                                   material=self.material,
                                   comp=self.comp)
