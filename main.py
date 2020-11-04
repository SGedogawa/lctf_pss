from make_hyper_spectrum_cube import make_hyper_spectrum_cube as pss


if __name__ == "__main__":
    spectrum_cube_from_csv = pss.MakeSpectrumCube().from_csv()
    spectrum_cube_from_bmp = pss.MakeSpectrumCube().from_bmp()
