from psslib import make_spectrum_cube as sc
from psslib import make_hyper_spectrum_cube as hsc


if __name__ == "__main__":
    spectrum_cube_from_csv = sc.MakeSpectrumCube().from_csv()
    spectrum_cube_from_bmp = sc.MakeSpectrumCube().from_bmp()

    hsc.MakeHyperSpectrumCube(spectrum_cube=spectrum_cube_from_csv.spectrum_cube).reshape()
