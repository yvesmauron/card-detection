from abc import ABC
import numpy as np


class Card(ABC):

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

    def ref_box_hl(self):
        return np.array(
            [
                [self.box_x_border, self.box_y_border],
                [self.box_x_width, self.box_y_border],
                [self.box_x_width, self.box_y_height],
                [self.box_x_border, self.box_y_height]
            ], dtype=np.float32)

    def ref_box_lr(self):
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


class FrenchCard(Card):

    def __init__(
        self,
        width: float = 57,
        height: float = 87,
        box_x_border: float = 3,
        box_x_width: float = 54,
        box_y_border: float = 3,
        box_y_height: float = 43.5,
        zoom: int = 4
    ):
        super().__init__(
            width=width,
            height=height,
            box_x_border=box_x_border,
            box_x_width=box_x_width,
            box_y_border=box_y_border,
            box_y_height=box_y_height,
            zoom=zoom
        )
