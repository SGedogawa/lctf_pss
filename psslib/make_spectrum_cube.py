import os

from settings import settings

import cv2
import numpy as np
import pandas as pd


class SpectrumCube(object):

    def __init__(self, spectrum_cube, material, comp):
        self.spectrum_cube = spectrum_cube
        self.material = material
        self.comp = comp

    def bin_image(self, num=10, threshold=30, show=False):
        if self.spectrum_cube.dtype != 'uint8':
            print("error message:8bitへ圧縮してください")
        else:
            bin_image = np.where(self.spectrum_cube[:, :, num] < threshold, 0, 255)

            if show is True:
                self.show_image(image=bin_image, name="bin_image_{}_{}".format(num, threshold))
                return bin_image

    def gray_image(self, num=10, show=False):
        if self.spectrum_cube.dtype != 'uint8':
            print("error message:8bitへ圧縮してください")
        else:
            gray_image = self.spectrum_cube[:, :, num]
            if show is True:
                self.show_image(image=gray_image, name="gray_image_{}".format(num))
                return gray_image

    def show_image(self, image, name="image"):
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_info(self):
        print(
            "=" * 25 + "\n",
            "material:{}\n".format(self.material),
            "max:{}\n".format(np.max(self.spectrum_cube)),
            "min:{}\n".format(np.min(self.spectrum_cube)),
            "shape:{}\n".format(self.spectrum_cube.shape),
            "type:{}\n".format(self.spectrum_cube.dtype),
            "composition:{}\n".format(self.comp),
            "=" * 25 + "\n",
        )


class MakeSpectrumCube(object):

    def __init__(self, input_path=settings.input_path, comp=False, show=True):
        self.input_path = input_path
        self.spectrum_cube = np.empty([settings.img_height, settings.img_width, len(settings.wavelength)],
                                      dtype=np.uint)
        self.comp = comp
        self.show = show

    def from_bmp(self) -> SpectrumCube:

        bmp_input_path = self.input_path + "/bmp"
        file_list = os.listdir(bmp_input_path)

        for i, w in enumerate(settings.wavelength):
            for file_name in file_list:
                if str(w) + "nm" in file_name:
                    img = cv2.imread(bmp_input_path + "/" + file_name)
                    self.spectrum_cube[:, :, i] = img[:, :, 0]

        spectrum_cube = self.uint16to8(spectrum_cube=self.spectrum_cube)
        self.show_info(spectrum_cube=spectrum_cube, material="bmp")

        return SpectrumCube(spectrum_cube=spectrum_cube, material="bmp", comp=self.comp)

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

        spectrum_cube = self.uint16to8(spectrum_cube=self.spectrum_cube)
        self.show_info(spectrum_cube=spectrum_cube, material="csv")

        return SpectrumCube(spectrum_cube=spectrum_cube, material="csv", comp=self.comp)

    def show_info(self, spectrum_cube, material):
        if self.show is True:
            print(
                "=" * 25 + "\n",
                "material:{}\n".format(material),
                "max:{}\n".format(np.max(spectrum_cube)),
                "min:{}\n".format(np.min(spectrum_cube)),
                "shape:{}\n".format(spectrum_cube.shape),
                "type:{}\n".format(spectrum_cube.dtype),
                "composition:{}\n".format(self.comp),
                "=" * 25 + "\n",
            )

    def uint16to8(self, spectrum_cube):
        if np.max(spectrum_cube) > 255 and self.comp is True:

            spectrum_cube = (spectrum_cube * 255 // 1024).astype(np.uint8)

            if self.show is True:
                print(
                    "=" * 45 + "\n",
                    "=" * 5 + "最大値が255を超えるため下記へ圧縮しました。" + "=" * 5 + "\n",
                    "=" * 45 + "\n",
                )

        elif np.max(spectrum_cube) <= 255:
            spectrum_cube = spectrum_cube.astype(np.uint8)

        return spectrum_cube
