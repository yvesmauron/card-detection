import numpy as np
import cv2
import os
from uuid import uuid4
from src.data.card import ReferenceCard
import shutil
import logging
from src.visualization.visualize import display_image


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def alphamask(
    ref_card: ReferenceCard = ReferenceCard(),
    bord_size: int = 2
):
    alphamask = np.ones(
        (ref_card.height, ref_card.width),
        dtype=np.uint8
    ) * 255
    cv2.rectangle(
        alphamask,
        (0, 0),
        (ref_card.width-1, ref_card.height-1),
        0,
        bord_size
    )
    cv2.line(
        alphamask,
        (bord_size*3, 0),
        (0, bord_size*3),
        0,
        bord_size
    )
    cv2.line(
        alphamask,
        (ref_card.width - bord_size*3, 0),
        (ref_card.width, bord_size*3),
        0,
        bord_size
    )
    cv2.line(
        alphamask,
        (0, ref_card.height-bord_size*3),
        (bord_size*3, ref_card.height),
        0,
        bord_size
    )
    cv2.line(
        alphamask,
        (ref_card.width-bord_size*3, ref_card.height),
        (ref_card.width, ref_card.height-bord_size*3),
        0,
        bord_size
    )

    return alphamask


def varianceOfLaplacian(img):
    """
    Compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian
    Source: A.Rosebrock,
    https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()


def extract_card(
        img,
        output_path=None,
        ref_card: ReferenceCard = ReferenceCard(),
        min_focus=120,
        debug=False
):
    """
    """
    imgwarp = None
    # Check the image is not too blurry
    focus = varianceOfLaplacian(img)
    if focus < min_focus:
        if debug:
            print("Focus too low :", focus)
        return False, None

    # Convert in gray color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise-reducing and edge-preserving filter
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge extraction
    edge = cv2.Canny(gray, 30, 200)

    # Find the contours in the edged image
    cnts, _ = cv2.findContours(
        edge.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # We suppose that the contour with largest area corresponds
    # to the contour delimiting the card
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # We want to check that 'cnt' is the contour of a rectangular shape
    # First, determine 'box', the minimum area bounding rectangle of 'cnt'
    # Then compare area of 'cnt' and area of 'box'
    # Both areas sould be very close
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    areaCnt = cv2.contourArea(cnt)
    areaBox = cv2.contourArea(box)
    valid = areaCnt / areaBox > 0.95

    if valid:
        # We want transform the zone inside the contour into the reference
        # rectangle of dimensions (cardW,cardH)
        ((xr, yr), (wr, hr), thetar) = rect
        # Determine 'Mp' the transformation that transforms 'box' into the
        # reference rectangle
        if wr > hr:
            Mp = cv2.getPerspectiveTransform(
                np.float32(box),
                ref_card.card()
            )
        else:
            Mp = cv2.getPerspectiveTransform(
                np.float32(box),
                ref_card.card_rotated()
            )
        # Determine the warped image by applying the transformation
        # to the image
        imgwarp = cv2.warpPerspective(
            img,
            Mp,
            (ref_card.width, ref_card.height)
        )
        # Add alpha layer
        imgwarp = cv2.cvtColor(imgwarp, cv2.COLOR_BGR2BGRA)

        # Shape of 'cnt' is (n,1,2), type = int with n  =  number of points
        # We reshape into (1,n,2), type = float32, before
        # feeding to perspectiveTransform
        cnta = cnt.reshape(1, -1, 2).astype(np.float32)
        # Apply the transformation 'Mp' to the contour
        cntwarp = cv2.perspectiveTransform(cnta, Mp)
        cntwarp = cntwarp.astype(np.int)

        # We build the alpha channel so that we have transparency on the
        # external border of the card
        # First, initialize alpha channel fully transparent
        alphachannel = np.zeros(imgwarp.shape[:2], dtype=np.uint8)
        # Then fill in the contour to make opaque this zone of the card
        cv2.drawContours(alphachannel, cntwarp, 0, 255, -1)

        # Apply the alphamask onto the alpha channel to clean it
        alphachannel = cv2.bitwise_and(alphachannel, alphamask())

        # Add the alphachannel to the warped image
        imgwarp[:, :, 3] = alphachannel

        # Save the image to file
        cv2.imwrite(output_path, imgwarp)

    if debug:
        cv2.imshow("Gray", gray)
        cv2.imshow("Canny", edge)
        edge_bgr = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(edge_bgr, [box], 0, (0, 0, 255), 3)
        cv2.drawContours(edge_bgr, [cnt], 0, (0, 255, 0), -1)
        cv2.imshow("Contour with biggest area", edge_bgr)
        if valid:
            cv2.imshow("Alphachannel", alphachannel)
            cv2.imshow("Extracted card", imgwarp)

    return valid, imgwarp


def extract_cards_from_video(
    video_file: str,
    output_dir: str,
    ref_card: ReferenceCard = ReferenceCard(),
    keep_ratio: int = 5,
    min_focus: int = 120,
    debug: bool = False
):
    """Extract cards from media file 'video_file'
        If 'output_dir' is specified, the cards are saved in 'output_dir'.
        One file per card with a random file name
        Because 2 consecutives frames are probably very similar, we don't use
        every frame of the video,
        but only one every 'keep_ratio' frames

        Returns list of extracted images

    Args:
        video_file ([type]): [description]
        output_dir ([type], optional): [description]. Defaults to None.
        keep_ratio (int, optional): [description]. Defaults to 5.
        min_focus (int, optional): [description]. Defaults to 120.
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    if not os.path.isfile(video_file):
        print(f"Video file {video_file} does not exist !!!")
        return -1, []

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_file)

    frame_nb = 0
    imgs_list = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        # Work on every 'keep_ratio' frames
        if frame_nb % keep_ratio == 0:
            output_path = os.path.join(output_dir, str(uuid4()) + ".png")
            valid, card_img = extract_card(
                img,
                output_path,
                ref_card=ref_card,
                min_focus=min_focus,
                debug=debug
            )

            if valid:
                imgs_list.append(card_img)
        frame_nb += 1

    if debug:
        cap.release()
        cv2.destroyAllWindows()

    return imgs_list


if __name__ == '__main__':
    DEBUG = False
    img = cv2.imread("data/test/scene.png")
    display_image(img)
    valid, card = extract_card(img, "test/extracted_card.png", debug=DEBUG)

    if valid:
        fig, _ = display_image(card)
        fig.savefig("data/test/test_fig.png")
