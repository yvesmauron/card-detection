from abc import ABC
import numpy as np
import cv2


class CardIdentfierBox():

    def __init__(
        self,
        width: float,
        height: float,
        box_x_border: float,
        box_x_width: float,
        box_y_border: float,
        box_y_height: float,
        zoom: int = 4
    ):
        self.width = int(width * zoom)
        self.height = int(height * zoom)
        self.box_x_border = int(box_x_border * zoom)
        self.box_x_width = int(box_x_width * zoom)
        self.box_y_border = int(box_y_border * zoom)
        self.box_y_height = int(box_y_height * zoom)

    def ref_box_tl(self):
        return np.array(
            [
                [self.box_x_border, self.box_y_border],
                [self.box_x_width, self.box_y_border],
                [self.box_x_width, self.box_y_height],
                [self.box_x_border, self.box_y_height]
            ], dtype=np.float32)

    def ref_box_br(self):
        return np.array(
            [
                [self.width - self.box_x_border,
                 self.height - self.box_y_border],
                [self.width - self.box_x_width,
                 self.height - self.box_y_border],
                [self.width - self.box_x_width,
                 self.height - self.box_y_height],
                [self.width - self.box_x_border,
                 self.height - self.box_y_height]
            ],
            dtype=np.float32)

    def ref_boxes(self):
        return np.array([self.ref_box_hl(), self.ref_box_lr()])

    def ref_hull(self, img: np.array, box: list = None):
        """
            Find in the zone 'box' of image 'img' and return, the convex hull
            delimiting the value and suit symbols
            'box' (shape (4,2)) is an array of 4 points delimiting a
            rectangular zone, takes one of the 2 possible values : refboxHL or
            refboxLR
        """

        if box is None:
            box = self.ref_box_tl()

        kernel = np.ones((3, 3), np.uint8)
        box = box.astype(np.int)

        # We will focus on the zone of 'img' delimited by 'box'
        x1, y1 = box[0]
        x2, y2 = box[2]
        w = x2-x1
        h = y2-y1
        zone = img[y1:y2, x1:x2].copy()

        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        thld = cv2.Canny(gray, 30, 200)
        thld = cv2.dilate(thld, kernel, iterations=1)

        # Find the contours
        contours, _ = cv2.findContours(
            thld.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # We will reject contours with small area. TWEAK, 'zoom' dependant
        min_area = 30
        # Reject contours with a low solidity. TWEAK
        min_solidity = 0.3

        # We will aggregate in 'concat_contour' the contours
        # that we want to keep
        concat_contour = None

        for c in contours:
            area = cv2.contourArea(c)

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area
            # Determine the center of gravity (cx,cy) of the contour
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #  abs(w/2-cx)<w*0.3 and abs(h/2-cy)<h*0.4 : TWEAK, the idea here
            # is to keep only the contours which are closed to the center of
            # the zone
            if area >= min_area \
                    and abs(w/2-cx) < w*0.3 \
                    and abs(h/2-cy) < h*0.4 \
                    and solidity > min_solidity:
                if concat_contour is None:
                    concat_contour = c
                else:
                    concat_contour = np.concatenate((concat_contour, c))

        if concat_contour is not None:
            # At this point, we suppose that 'concat_contour' contains only
            # the contours corresponding the value and suit symbols
            # We can now determine the hull
            hull = cv2.convexHull(concat_contour)
            hull_area = cv2.contourArea(hull)
            # If the area of the hull is to small or too big, there may
            # be a problem
            min_hull_area = 520  # TWEAK, deck and 'zoom' dependant
            max_hull_area = 2120  # TWEAK, deck and 'zoom' dependant
            if hull_area < min_hull_area or hull_area > max_hull_area:
                return None
            # So far, the coordinates of the hull are relative to 'zone'
            # We need the coordinates relative to the image -> 'hull_in_img'
            hull_in_img = hull+box[0]
        else:
            return None

        return hull_in_img


class CardSet(ABC):
    def __init__(
        self,
        width: float = 57,
        height: float = 87,
        zoom: int = 4
    ):
        self.width = int(width * zoom)
        self.height = int(height * zoom)

    def ref_card(self):

        return np.array(
            [[0, 0],
             [self.width, 0],
             [self.width, self.height],
             [0, self.height]],
            dtype=np.float32
        )

    def ref_card_rotated(self):
        return np.array(
            [[self.width, 0],
             [self.width, self.height],
             [0, self.height],
             [0, 0]],
            dtype=np.float32)


class FrenchCards(CardSet):

    def __init__(
        self,
        width: float = 57,
        height: float = 87,
        zoom: int = 4,
        custom_bbox=None
    ):
        super().__init__(
            width=width,
            height=height,
            zoom=zoom
        )

        if custom_bbox is not None:
            assert type(custom_bbox) == dict, "custom_bbox should be a dict"

            self.cards = custom_bbox

        else:

            card_suit_values = [
                value+suit for value in ['A', 'K', 'Q', 'J',
                                         '10', '9', '8', '7', '6']
                for suit in ['s', 'h', 'd', 'c']
            ]

            self.cards = dict()
            for card in card_suit_values:

                if card[0] in ['K', 'Q', 'J']:
                    box_x_border = 3
                    box_x_width = 27
                    box_y_border = 5
                    box_y_height = 25
                elif card[0] == "A":
                    box_x_border = 2
                    box_x_width = 12
                    box_y_border = 4
                    box_y_height = 13
                else:
                    box_x_border = 3
                    box_x_width = 22
                    box_y_border = 5
                    box_y_height = 22

                self.cards[card] = CardIdentfierBox(
                    width=width,
                    height=height,
                    box_x_border=box_x_border,
                    box_x_width=box_x_width,
                    box_y_border=box_y_border,
                    box_y_height=box_y_height,
                    zoom=zoom
                )


if __name__ == '__main__':

    fc = FrenchCards()

    print(fc.cards["7s"])
