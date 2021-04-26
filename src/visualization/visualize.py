from glob import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.data.card import CardSet, FrenchCards
from pathlib import Path
from uuid import uuid4


def display_image(
    img: np.array,
    polygons: list = [],
    channels: str = "bgr",
    size: int = 9
):
    """Function to display an inline image, and draw optional polygons
    (bounding boxes, convex hulls) on it. Use the param 'channels' to specify
    the order of the channels ("bgr" for an image coming from OpenCV world)


    Args:
        img (np.array): [description]
        polygons (list, optional): [description]. Defaults to [].
        channels (str, optional): [description]. Defaults to "bgr".
        size (int, optional): [description]. Defaults to 9.
    """
    if not isinstance(polygons, list):
        polygons = [polygons]
    if channels == "bgr":  # bgr (cv2 image)
        nb_channels = img.shape[2]
        if nb_channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_facecolor((0, 0, 0))
    ax.imshow(img)
    for polygon in polygons:
        # An polygon has either shape (n,2),
        # either (n,1,2) if it is a cv2 contour (like convex hull).
        # In the latter case, reshape in (n,2)
        if len(polygon.shape) == 3:
            polygon = polygon.reshape(-1, 2)
        patch = patches.Polygon(polygon, linewidth=1,
                                edgecolor='g', facecolor='none')
        ax.add_patch(patch)

    return fig, ax


def display_random_card(
    card_set: CardSet = FrenchCards(),
    card_filter: str = "*",
    input_dir: str = "data/processed/cards",
    fig_path: Path = f"data/test/random_card_{uuid4()}.png"
):

    image_files = glob(input_dir + f"/{card_filter}/*.png")
    selected_file = random.choice(image_files)
    fig, ax = display_image(
        cv2.imread(selected_file, cv2.IMREAD_UNCHANGED),
        polygons=[
            card_set.cards["7s"].ref_box_tl(),
            card_set.cards["7s"].ref_box_br()
        ]
    )

    fig.savefig(fig_path)


if __name__ == '__main__':

    display_random_card(card_filter="Ks")
