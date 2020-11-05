from psslib import make_spectrum_cube as makesc
from psslib import cube_tools


if __name__ == "__main__":
    sc = makesc.MakeSpectrumCube(comp=True).from_csv()
    reshape = cube_tools.Tools(sc).reshape()
    reshape.show_info()
