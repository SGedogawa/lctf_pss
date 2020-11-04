import os

from settings import settings

import cv2
import numpy as np
import pandas as pd


class SpectrumCube(object):

    def __init__(self, spectrum_cube, material="csv", comp=True, show=True):
        self.material = material
        self.comp = comp

        if show is True:
            print(
                "=" * 25 + "\n",
                "material:{}\n".format(material),
                "max:{}\n".format(np.max(spectrum_cube)),
                "min:{}\n".format(np.min(spectrum_cube)),
                "shape:{}\n".format(spectrum_cube.shape),
                "=" * 25 + "\n",
            )

        if np.max(spectrum_cube) > 255 and self.comp is True:
            spectrum_cube = spectrum_cube * 256 / 1024
            print("=" * 5 + "最大値が255を超えるため圧縮しました。" + "=" * 5 + "\n")
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

    def to_hyper_spectrum_cube(self):
        return MakeHyperSpectrumCube(spectrum_cube=self.spectrum_cube)


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


class MakeHyperSpectrumCube(object):

    def __init__(self, spectrum_cube):
        spectrum_cube = SpectrumCube(spectrum_cube, show=False)
        hyper_spectrum_cube = np.empty([settings.hyper_img_height, settings.hyper_img_width, len(settings.wavelength)],
                                       dtype=np.uint8)

        for i in range(len(settings.wavelength)):
            gray_img = spectrum_cube.show_gray_image(num=i, show=False)
            img = gray_img[50: 450, 50: 650]
            hyper_spectrum_cube[:, :, i] = img
        '''
            mask = spectrum_cube.show_bin_image(num=i, threshold=220, show=False)
            dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            hyper_spectrum_cube[:, :, i] = dst
        '''

        self.hyper_spectrum_cube = hyper_spectrum_cube

    def akaze_matching(self, first=10, second=11):
        first_img = self.hyper_spectrum_cube[:, :, first]
        second_img = self.hyper_spectrum_cube[:, :, second]

        # A-KAZE検出器の生成
        akaze = cv2.AKAZE_create()

        # 特徴量の検出と特徴量ベクトルの計算
        kp1, des1 = akaze.detectAndCompute(first_img, None)
        kp2, des2 = akaze.detectAndCompute(second_img, None)

        # Brute-Force Matcher生成
        bf = cv2.BFMatcher()

        # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
        matches = bf.knnMatch(des1, des2, k=2)

        # データを間引きする
        ratio = 0.5
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append([m])

        # 対応する特徴点同士を描画
        akaze_matching_img = cv2.drawMatchesKnn(first_img, kp1, second_img, kp2, good, None, flags=2)
        self.show_image(image=akaze_matching_img, name="_{}_and_{}_akaze_matching_image".format(first, second))

    def show_image(self, image, name="image"):
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
