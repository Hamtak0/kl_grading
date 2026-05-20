import numpy as np
import pydicom as pdc
from pathlib import Path

def normalize(img: np.array) -> np.array:
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    return img

# -- PixelSpacing
# Physical distance in the patient between the center of each pixel, specified by a numeric pair -
# adjacent row spacing (delimiter) adjacent column spacing in mm.
# The first value is the row spacing in mm, that is the spacing between the centers of adjacent rows, or vertical spacing.
# The second value is the column spacing in mm, that is the spacing between the centers of adjacent columns, or horizontal spacing.
def mm_to_pixel(ds: pdc.Dataset, x: float, y: float) -> tuple[int, int]:
    ipp = np.array(ds.PixelSpacing)
    pixel_x = int(x / ipp[0])
    pixel_y = int(y / ipp[1])
    return pixel_x, pixel_y

def photometric(ds: pdc.Dataset, image: np.array) -> np.array:
    return 1 - image if ds.PhotometricInterpretation == "MONOCHROME1" else image

def read_dicom_image(file: Path) -> tuple[pdc.Dataset, np.array]:
    ds = pdc.dcmread(file)
    image = normalize(ds.pixel_array)
    image = photometric(ds, image)
    return ds, image
