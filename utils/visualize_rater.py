import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes

from utils.cropping import cropping_points, center_to_corners, get_all_points
from utils.dicom_cut import read_dicom_image, mm_to_pixel

def draw_box(
        ax: Axes,
        color: str,
        title: str | None = None,
        box: tuple[float, float, float, float] | list[float] | np.ndarray | None = None,
        points: np.ndarray | None = None
    ) -> patches.Rectangle | None:
    # Helper function to draw bounding boxes
    # inputs:
    #   pre-calculated box [xmin, ymin, xmax, ymax]
    #   or 2D numpy array of points [[x1, y1], [x2, y2], ...]
    if box is not None:
        xmin, ymin, xmax, ymax = box
    elif points is not None:
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0])
        ymin = np.min(points[:, 1])
        ymax = np.max(points[:, 1])
    else:
        return None

    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=2,
        edgecolor=color,
        facecolor='none',
        label=title
    )
    ax.add_patch(rect)
    return rect

def main(specific: str | None = None) -> None:
    kt_path = Path("./dataset/landmarks")
    so_path = Path("./dataset/SO_landmarks")

    kt_files = list(kt_path.glob("ID*.mrk.json"))
    so_files = list(so_path.glob("ID*.mrk.json"))
    kt_names = set(x.name for x in kt_files)
    so_names = set(x.name for x in so_files)
    both = sorted(kt_names & so_names) if not specific else [specific]

    for file in both:
        print(f"X-ray of {file}")
        with open(kt_path / file, "r") as f:
            data1 = json.load(f)
        with open(so_path / file, "r") as f:
            data2 = json.load(f)

        root = Path("./dataset/bilateral_standing_AP")
        file_name = Path(f"{file.split('.')[0]}.dcm")
        ds, image = read_dicom_image(root / file_name)

        #! rescaled with mm_to_pixel
        # show all manual points
        point1 = np.array([mm_to_pixel(ds, *point) for point in get_all_points(data1).values()])
        point2 = np.array([mm_to_pixel(ds, *point) for point in get_all_points(data2).values()])

        # create border corners
        left_crop_kt, right_crop_kt = cropping_points(data1)
        left_crop_so, right_crop_so = cropping_points(data2)

        left_crop_kt = np.array([mm_to_pixel(ds, *point) for point in center_to_corners(*left_crop_kt)])
        right_crop_kt = np.array([mm_to_pixel(ds, *point) for point in center_to_corners(*right_crop_kt)])
        left_crop_so = np.array([mm_to_pixel(ds, *point) for point in center_to_corners(*left_crop_so)])
        right_crop_so = np.array([mm_to_pixel(ds, *point) for point in center_to_corners(*right_crop_so)])

        fig, ax = plt.subplots()
        plt.title(f"{file_name.stem}", fontsize=16)
        ax.imshow(image, cmap="gray")

        # plt.scatter(left_crop_kt[:, 0], left_crop_kt[:, 1], color="red")
        # plt.scatter(right_crop_kt[:, 0], right_crop_kt[:, 1], color="red")
        # plt.scatter(left_crop_so[:, 0], left_crop_so[:, 1], color="blue")
        # plt.scatter(right_crop_so[:, 0], right_crop_so[:, 1], color="blue")

        ax.scatter(point1[:, 0], point1[:, 1], color="red")
        ax.scatter(point2[:, 0], point2[:, 1], color="blue")

        draw_box(ax, "red", points=left_crop_kt)
        draw_box(ax, "red", points=right_crop_kt)
        draw_box(ax, "blue", points=left_crop_so)
        draw_box(ax, "blue", points=right_crop_so)
        plt.show()

if __name__ == "__main__":
    id = input("Input ID: ")
    id = None if id == "" else "ID" + id + ".mrk.json"
    main(id)
