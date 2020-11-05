import os

from settings import settings

import cv2
import numpy as np
# from nptyping import NDArray
import pandas as pd


class SpectrumCube(object):

    def __init__(self, spectrum_cube, material="csv", comp=True, show=True):
        self.material = material
        self.comp = comp

        if show is True:
            print(
                "=" * 25 + "\n",
                "=" * 25 + "\n",
                "material:{}\n".format(material),
                "max:{}\n".format(np.max(spectrum_cube)),
                "min:{}\n".format(np.min(spectrum_cube)),
                "shape:{}\n".format(spectrum_cube.shape),
                "composition:{}\n".format(self.comp),
                "=" * 25 + "\n",
                "=" * 25 + "\n",
                )

        if np.max(spectrum_cube) > 255 and self.comp is True:
            spectrum_cube = spectrum_cube * 256 // 1024
            print("=" * 5 + "最大値が255を超えるため下記へ圧縮しました。" + "=" * 5 + "\n",
                  "=" * 25 + "\n",
                  "material:{}\n".format(material),
                  "max:{}\n".format(np.max(spectrum_cube)),
                  "min:{}\n".format(np.min(spectrum_cube)),
                  "shape:{}\n".format(spectrum_cube.shape),
                  "composition:{}\n".format(self.comp),
                  "=" * 25 + "\n",
                  )
            self.spectrum_cube = spectrum_cube.astype(np.uint8)

        elif np.max(spectrum_cube) > 255 and self.comp is False:
            self.spectrum_cube = spectrum_cube.astype(np.uint16)

        else:
            self.spectrum_cube = spectrum_cube.astype(np.uint8)

    def show_bin_image(self, num=10, threshold=30, show=True):
        bin_image = np.where(self.spectrum_cube[:, :, num] < threshold, 0, 255)
        bin_image = bin_image.astype(np.uint8)

        if show is True:
            self.show_image(image=bin_image, name="bin_image_{}_{}".format(num, threshold))

        return bin_image

    def show_gray_image(self, num=10, show=True):
        gray_image = self.spectrum_cube[:, :, num]

        if show is True:
            self.show_image(image=gray_image, name="gray_image_{}".format(num))

        return gray_image

    def show_image(self, image, name="image"):
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class MakeSpectrumCube(object):

    def __init__(self, input_path=settings.input_path):
        self.input_path = input_path
        self.spectrum_cube = np.empty([settings.img_height, settings.img_width, len(settings.wavelength)],
                                      dtype=np.uint)

    def from_bmp(self) -> SpectrumCube:

        bmp_input_path = self.input_path + "/bmp"
        file_list = os.listdir(bmp_input_path)

        for i, w in enumerate(settings.wavelength):
            for file_name in file_list:
                if str(w) + "nm" in file_name:
                    img = cv2.imread(bmp_input_path + "/" + file_name)
                    self.spectrum_cube[:, :, i] = img[:, :, 0]

        return SpectrumCube(spectrum_cube=self.spectrum_cube, material="bmp")

    def from_csv(self) -> SpectrumCube:

        csv_input_path = self.input_path + "/csv"
        file_list = os.listdir(csv_input_path)

        column_list = []

        for i in range(0, settings.img_width):
            column_list.append(i)

        for i, w in enumerate(settings.wavelength):
            for file_name in file_list:
                if str(w) + "nm" in file_name:
                    img = pd.read_csv(
                        csv_input_path + "/" + file_name,
                        delimiter=',',
                        skiprows=11,
                        usecols=column_list,
                        header=None,
                        )
                    self.spectrum_cube[:, :, i] = img.values

        return SpectrumCube(spectrum_cube=self.spectrum_cube, material="csv")
