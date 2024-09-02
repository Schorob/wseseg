import numpy as np
import cv2
np.random.seed(12345)


class Clicker(object):
    def __init__(self, gt_mask, ignore_label=-1, click_indx_offset=0):
        """
        :param gt_mask:
            [H, W] ground truth mask. Possible values: {0, 1, ignore_label}
        :param ignore_label:
            A numerical (integer) label of which click to ignore.
        """
        self.click_indx_offset = click_indx_offset
        self.gt_mask = gt_mask == 1
        self.valid_mask = gt_mask != ignore_label
        self.not_clicked_map = np.full(gt_mask.shape, True, dtype=np.bool)


    def get_next_click(self, pred_mask, ij=True):
        """
        :param pred_mask:
            The previously predicted boolean mask of shape [H, W]
        :return:
            The (i, j) or (x, y) coordinates of the next click, and whether the click is a foreground
            or background click.
        """

        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.valid_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.valid_mask)

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        # Since multiple points may correspond to the maximum of the DT at once
        # we will choose some at random
        p_x = coords_x[np.random.randint(coords_x.shape[0])]
        p_y = coords_y[np.random.randint(coords_y.shape[0])]

        if ij:
            return p_y, p_x, is_positive
        else:
            return p_x, p_y, is_positive