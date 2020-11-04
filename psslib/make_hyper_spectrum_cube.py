from settings import settings
from psslib import make_spectrum_cube as sc

import cv2
import numpy as np


class MakeHyperSpectrumCube(object):

    def __init__(self, spectrum_cube):
        spectrum_cube = sc.SpectrumCube(spectrum_cube, show=False)
        hyper_spectrum_cube = np.empty([settings.hyper_img_height, settings.hyper_img_width, len(settings.wavelength)],
                                       dtype=np.uint8)

        for i in range(len(settings.wavelength)):
            gray_img = spectrum_cube.show_gray_image(num=i, show=False)
            img = gray_img[50: 450, 150: 650]
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
        ratio = 2.0
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
