import numpy as np
import torch
from utils import clicker
import torch.nn.functional as F
from typing import Tuple
from copy import deepcopy
import segment_anything
import segment_anything_hq
from datasets import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import scipy.ndimage as ndimage
import cv2
import time
import mreplay


# We are in a permanent testing situation; avoid randomness for reproducibility
torch.manual_seed(12345)
np.random.seed(12345)


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.device)
        self.memory_replay = mreplay.MemoryReplay(self.config.replay_size)

        if self.config.backbone_size == "b":
            self.sam_model = segment_anything.build_sam_vit_b(checkpoint=self.config.checkpoint).to(self.device)
        elif self.config.backbone_size == "b_hq":
            self.sam_model = segment_anything_hq.build_sam_vit_b(checkpoint=self.config.checkpoint).to(self.device)
        else:
            raise NotImplementedError("The selected model size is not available. Resort to 'b' or 'b_hq'. ")

        # In order to reset the decoder after each image, we need a copy of the state dict
        if self.config.reset_decoder_after_each_image:
            self.decoder_state_dict = self.sam_model.mask_decoder.state_dict()

        sam_params = sum(p.numel() for p in self.sam_model.parameters())
        sam_decoder_params = sum(p.numel() for p in self.sam_model.mask_decoder.parameters())
        print("Decoder Params | rest : ", sam_decoder_params, "|", sam_params - sam_decoder_params)

        if self.config.backbone_size[-3:] == "_hq":
            self.sam_predictor = segment_anything_hq.predictor.SamPredictor(self.sam_model, hq_token_use=self.config.hq_token_use)
        else:
            self.sam_predictor = segment_anything.predictor.SamPredictor(self.sam_model)

        self.optimizer = optim.Adam(list(self.sam_model.mask_decoder.parameters()), lr=self.config.learning_rate)

        self.sam_input_image_size = self.sam_model.image_encoder.img_size

        # Load the dataset
        self.test_dataset = sports_tools.SportsTools(self.config.dataset_path, self.config.sports_tools_class)
        self.test_dataset.dataset_samples = np.random.permutation(np.array(self.test_dataset.dataset_samples))
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=1)

        self.criterion = nn.BCEWithLogitsLoss(reduction="none").to(self.device)


    def adaptive_test(self):
        """
        This function carries out a test using an adaptation method described in
        the paper.
        :return:
            avg_noc : The average NoC@IoU on the dataset (averaged over all images.
            failure_ratio : The percentage of images for whom the click limit
            set in the config file was not sufficient to attain the indicated
            IoU level.
        """
        total_noc = 0
        total_imgs = 0
        failures = 0

        sum_opt_duration = 0.0
        num_opt_durations = 0

        pbar = tqdm(self.test_loader)
        for image, gt_mask in pbar:
            image = image.cpu().numpy()[0]
            gt_mask = gt_mask.cpu().numpy()[0]

            noc, duration = self.test_single_image_predictor(image, gt_mask)
            if duration is not None:
                sum_opt_duration += duration
                num_opt_durations += 1

            if noc >= self.config.max_clicks:
                failures += 1
            total_noc += noc
            total_imgs += 1
            pbar.set_description(str(noc) + " NoC")

        avg_noc = total_noc / total_imgs
        failure_ratio = (failures / total_imgs) * 100
        avg_opt_duration = sum_opt_duration / num_opt_durations
        print("Avg NoC: {:10.3f}".format(avg_noc))
        print("Failure Ratio: {:10.2f}".format(failure_ratio))
        print("Average duration (sec): ", avg_opt_duration)

        return avg_noc, failure_ratio

    def test_single_image_predictor(self, image, gt_mask):
        """
        :param image:
            [H, W, 3] image with values in [0,1]
        :param gt_mask:
            [H, W] ground truth mask; binarized to 0/1
        :return:
            NoC@IoU for the configuration fitted IoU (for this single image).
        """
        # The predictor model expects the image to have dtype uint8 and values in [0, 255]
        image = (image * 255.0).astype(np.uint8)
        self.sam_predictor.set_image(image)

        # In order to reset the decoder after each image, we need a copy of the state dict
        if self.config.reset_decoder_after_each_image:
            self.decoder_state_dict = self.sam_model.mask_decoder.state_dict()

        accumulated_points = [] # Will hold all points cumulatively as a [num_points, 2] list
        accumulated_labels = [] # List of the point labels (1 :  foreground; 0 : background)

        img_clicker = clicker.Clicker(gt_mask)
        initial_mask = np.zeros_like(gt_mask) # [H, W] int
        fc_x, fc_y, fc_positive = img_clicker.get_next_click(initial_mask == 1, ij=False)
        accumulated_points.append([fc_x, fc_y])
        accumulated_labels.append(fc_positive.astype(np.int32))

        # Assemble the input to the predictor for a single click iteration
        point_coords = np.array(accumulated_points)
        point_labels = np.array(accumulated_labels)
        mask_input, newh, neww = self.preprocess_mask(initial_mask)

        start_time = time.time()

        masks, iou_predictions, low_res_masks = self.sam_predictor.predict_with_gradient(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input = mask_input,
            box=None,
            multimask_output=False
        )

        duration = time.time() - start_time


        result_mask = masks[0, 0].detach().cpu().numpy()
        result_conf = torch.sigmoid(low_res_masks[0, 0].detach()).cpu().numpy()

        continue_clicking = (len(accumulated_points) < self.config.max_clicks) and \
                            self.config.min_iou > self.compute_iou(gt_mask, result_mask)

        if self.config.click_level_adaption:
            click_mask = self.clicks_to_mask(accumulated_points, accumulated_labels, gt_mask.shape)
            click_bce = self.balanced_ce(gt_mask=click_mask, low_res_logits=low_res_masks)

            # For PyTorch implementation reasons, a computational graph cannot be differentiated twice.
            if continue_clicking or not self.config.image_level_adaption:
                self.optimizer.zero_grad()
                click_bce.backward()
                self.optimizer.step()


        while continue_clicking:
            # Get the next click
            fc_x, fc_y, fc_positive = img_clicker.get_next_click(result_mask == 1, ij=False)
            accumulated_points.append([fc_x, fc_y])
            accumulated_labels.append(fc_positive.astype(np.int32))

            # Assemble the input to the predictor for a single click iteration
            point_coords = np.array(accumulated_points)
            point_labels = np.array(accumulated_labels)
            mask_input, newh, neww = self.preprocess_mask(result_mask)

            masks, iou_predictions, low_res_masks = self.sam_predictor.predict_with_gradient(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                box=None,
                multimask_output=False
            )

            result_mask = masks[0, 0].detach().cpu().numpy()
            result_conf = torch.sigmoid(low_res_masks[0, 0].detach()).cpu().numpy()

            continue_clicking = (len(accumulated_points) < self.config.max_clicks) and \
                self.config.min_iou > self.compute_iou(gt_mask, result_mask)

            if self.config.click_level_adaption:
                click_mask = self.clicks_to_mask(accumulated_points, accumulated_labels, gt_mask.shape)
                click_bce = self.balanced_ce(gt_mask=click_mask, low_res_logits=low_res_masks)

                # For PyTorch implementation reasons, a computational graph cannot be differentiated twice.
                if continue_clicking or not self.config.image_level_adaption:
                    self.optimizer.zero_grad()
                    click_bce.backward()
                    self.optimizer.step()

        if self.config.reset_decoder_after_each_image:
            self.sam_model.mask_decoder.load_state_dict(self.decoder_state_dict)
            masks, iou_predictions, low_res_masks = self.sam_predictor.predict_with_gradient(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=False
            )


        if self.config.filter_result_mask == 0: # (U)ntreated mask in the paper
            refined_mask = self.original_mask_and_click_incorporation(
                result_mask, accumulated_points, accumulated_labels, self.config.use_dense_mask_for_img_adapt
            )
        elif self.config.filter_result_mask == 1: # (E)roded mask in the paper
            refined_mask = self.mask_erosion_and_click_incorporation(
                result_mask, accumulated_points, accumulated_labels, self.config.use_dense_mask_for_img_adapt,
                erosion_iters=self.config.erosion_iters
            )
        elif self.config.filter_result_mask == 2: # (C)onfidence filtered mask in the paper
            refined_mask = self.confidence_filtering_and_click_incorporation(
                result_mask, result_conf, accumulated_points, accumulated_labels, self.config.use_dense_mask_for_img_adapt
            )
        else:
            raise NotImplementedError


        if self.config.image_level_adaption  and len(accumulated_points) < self.config.max_clicks:
            image_bce = self.balanced_ce(gt_mask=refined_mask, low_res_logits=low_res_masks)

            # If we have click adaptation, we have not yet differentiated the loss from the last click round
            if self.config.click_level_adaption and not self.config.reset_decoder_after_each_image:
                image_bce = image_bce + click_bce

            self.optimizer.zero_grad()
            image_bce.backward()
            self.optimizer.step()

            # Use and store in replay memory
            if self.config.replay_size > 0:
                if len(self.memory_replay.storage) > 0:
                    replay_loss = self.replay_loss()
                    self.optimizer.zero_grad()
                    replay_loss.backward()
                    self.optimizer.step()
                self.memory_replay.additem(
                    dict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        mask_input=mask_input,
                        refined_mask=refined_mask
                    )
                )


            self.additional_image_optimization_step(
                mask_input=mask_input, refined_mask=refined_mask,
                accumulated_points=accumulated_points, accumulated_labels=accumulated_labels)

        return len(accumulated_points), duration


    def additional_image_optimization_step(self, mask_input, refined_mask, accumulated_points, accumulated_labels):
        """
        Carry out an optimization step with the pseudo mask, after the interaction for one image is done.
        """
        point_coords = np.array(accumulated_points)
        point_labels = np.array(accumulated_labels)
        masks, iou_predictions, low_res_masks = self.sam_predictor.predict_with_gradient(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            box=None,
            multimask_output=False
        )

        click_bce = self.balanced_ce(gt_mask=refined_mask, low_res_logits=low_res_masks)

        self.optimizer.zero_grad()
        click_bce.backward()
        self.optimizer.step()


    def mask_erosion_and_click_incorporation(self,
        mask,
        accumulated_points,
        accumulated_labels,
        use_dense_mask,
        erosion_iters=5):
        """
        This corresponds to the mask erosion described in the paper. First the foreground
        and background mask are separated. Then, they two masks are iteratively eroded.
        The two resulting masks are then fused again.
        :param mask:
            A predicted mask which is assumed to at least have the configured IoU with the
            ground truth mask (binarized to 0/1). [H, W] np.bool
        :param accumulated_points:
            A N x 2 python list containing the clicked points in x, y format
        :param accumulated_labels:
            The point labels; 1 : foreground, 0 : background. A length N python list.
        :param use_dense_mask:
            Whether the dense mask should be incorporated at all. If not, only use the click mask.
        :return:
            A corroded version of this mask.
        """
        # 1. Create the dense mask
        fg_mask = (mask == 1) # [H, W] np.bool
        bg_mask = (mask == 0) # [H, W] np.bool
        fg_mask_eroded = ndimage.binary_erosion(fg_mask, iterations=erosion_iters) # [H, W] np.bool
        bg_mask_eroded = ndimage.binary_erosion(bg_mask, iterations=erosion_iters) # [H, W] np.bool

        size = mask.shape[0], mask.shape[1]

        dense_mask = np.full(shape=size, fill_value=-1, dtype=np.int32) # [H, W] np.int32
        dense_mask[fg_mask_eroded] = 1
        dense_mask[bg_mask_eroded] = 0

        # 2. Create the click mask
        click_mask = self.clicks_to_mask(accumulated_points, accumulated_labels, size=size)

        # 3. Fuse the masks accordingly (we know the click mask points to always be correct)
        if use_dense_mask:
            pseudo_mask = np.where(click_mask == -1, dense_mask, click_mask)
        else:
            pseudo_mask = click_mask

        return pseudo_mask

    def confidence_filtering_and_click_incorporation(self,
        mask, conf,
        accumulated_points,
        accumulated_labels,
        use_dense_mask,
        delta_conf=0.4):
        """
        Corresponds to the confidence based filtering in the paper. The argument conf
        is a map that contains the foreground-probability which is used for confidence
        filtering.
        :param mask:
            A predicted mask which is assumed to at least have the configured IoU with the
            ground truth mask (binarized to 0/1). [H, W] np.bool
        :param conf:
            The confidence mask which contains the probability of the pixels belonging
            to the foreground. Very low resolution, due to not being postprocessed yes.
            [64, 64] np.float
        :param accumulated_points:
            A N x 2 python list containing the clicked points in x, y format
        :param accumulated_labels:
            The point labels; 1 : foreground, 0 : background. A length N python list.
        :param use_dense_mask:
            Whether the dense mask should be incorporated at all. If not, only use the click mask.
        :param delta_conf:
            Since the masks confidence is expressed by a sigmoid (space [0,1]) the confidence
            is expressed by the distance from the 0.5 mark. The minimum distance for a pixel
            to be considered is delta_conf.
        :return:
            A confidence filtered version of this mask.
        """
        # 1. We have to expand the confidence mapt to the size of the mask.
        longer_side = max(mask.shape[0], mask.shape[1])
        size = mask.shape[0], mask.shape[1]
        conf = cv2.resize(conf, dsize=(longer_side, longer_side), interpolation=cv2.INTER_LINEAR)
        conf = conf[:mask.shape[0], :mask.shape[1]]

        # 2. Carry out the confidence filtering.
        confident_area = np.abs(0.5 - conf) > delta_conf
        dense_mask = np.full(shape=size, fill_value=-1, dtype=np.int32) # [H, W] np.int32
        dense_mask = np.where(confident_area, mask, dense_mask)

        # 3. Create the click mask
        click_mask = self.clicks_to_mask(accumulated_points, accumulated_labels, size=size)

        # 4. Fuse the masks accordingly (we know the click mask points to always be correct)
        if use_dense_mask:
            pseudo_mask = np.where(click_mask == -1, dense_mask, click_mask)
        else:
            pseudo_mask = click_mask

        return pseudo_mask

    def original_mask_and_click_incorporation(self,
        mask,
        accumulated_points,
        accumulated_labels,
        use_dense_mask):
        """
        :param mask:
            A predicted mask which is assumed to at least have the configured IoU with the
            ground truth mask (binarized to 0/1). [H, W] np.bool
        :param accumulated_points:
            A N x 2 python list containing the clicked points in x, y format
        :param accumulated_labels:
            The point labels; 1 : foreground, 0 : background. A length N python list.
        :param use_dense_mask:
            Whether the dense mask should be incorporated at all. If not, only use the click mask.

        """
        size = mask.shape[0], mask.shape[1]
        dense_mask = mask.astype(np.int32)

        # 1. Create the click mask
        click_mask = self.clicks_to_mask(accumulated_points, accumulated_labels, size=size)

        # 2. Fuse the masks accordingly (we know the click mask points to always be correct)
        if use_dense_mask:
            pseudo_mask = np.where(click_mask == -1, dense_mask, click_mask)
        else:
            pseudo_mask = click_mask

        return pseudo_mask


    def clicks_to_mask(self, accumulated_points, accumulated_labels, size, ignore_index = -1):
        """
        :param accumulated_points:
            A N x 2 python list containing the clicked points in x, y format
        :param accumulated_labels:
            The point labels; 1 : foreground, 0 : background. A length N python list.
        :param size:
            A 2-tuple containing the size of the created mask in (H, W) format.
        :param ignore_index:
            The scalar value which is to be written at any position where no click has occured.
        :return:
        """
        click_mask = np.full(shape=size, fill_value=ignore_index, dtype=np.int32)
        for p, l in zip(accumulated_points, accumulated_labels):
            i, j = p[1], p[0]
            click_mask[i, j] = l

        return click_mask


    def preprocess_mask(self, mask):
        """
        Resizes and pads the mask to be suitable as input for SAM.
        :param mask:
            (np.ndarray): A 0/1 mask for the image (not necessarily gt) in 1xHxW format

        """
        mask = torch.tensor(mask)[None] # [1, 1, H, W]
        # Compute the target resolution of the scaling applied to "image"
        oldh, oldw = mask.shape[1], mask.shape[2]
        scale = self.sam_input_image_size * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        # The input masks are assume to have the same ouput format as SAM (thus: 1/4 of the image size)
        scaled_mask = F.interpolate(mask[None].float(), size=(newh // 4, neww // 4)) # -> (1, 1, newh // 4, neww // 4)

        # SAM does not pad the mask to a quadratic shape internally; thus, we have to do it here manually first
        mask_padh = self.sam_input_image_size // 4 - newh // 4
        mask_padw = self.sam_input_image_size // 4- neww // 4
        scaled_mask = F.pad(scaled_mask, (0, mask_padw, 0, mask_padh))

        scaled_mask = scaled_mask.detach().cpu().numpy()[0]

        return scaled_mask, newh, neww




    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        oldh, oldw = original_size
        scale = self.sam_input_image_size * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (neww / oldw)
        coords[..., 1] = coords[..., 1] * (newh / oldh)
        return coords


    def compute_iou(self, mask, prediction):
        intersection = np.sum(np.logical_and(mask, prediction))
        union = np.sum(np.logical_or(mask, prediction))
        iou = intersection / union
        return iou

    def balanced_ce(self, gt_mask, low_res_logits):
        """
        :param gt_mask:
            (np.array): [H, W] ground truth mask; binarized to 0/1
        :param low_res_logits:
            (torch.Tensor): A predicted logit mask in in 1xHxW format
        """

        # The gt_mask is not in an appropriate shape to be compared against the logits
        oldh, oldw = gt_mask.shape
        scale = low_res_logits.shape[2] / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        gt_mask = torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0)
        gt_mask = F.interpolate(gt_mask.float(), size=(newh, neww))[0].long()

        padh = low_res_logits.shape[2] - newh
        padw = low_res_logits.shape[2] - neww
        gt_mask = F.pad(gt_mask, (0, padw, 0, padh), value=0)
        gt_mask = gt_mask.to(self.device)[None].float()

        fg_loss = self.criterion(low_res_logits, gt_mask) * (gt_mask == 1).int().float()
        bg_loss = self.criterion(low_res_logits, gt_mask) * (gt_mask == 0).int().float()

        total_loss = (fg_loss.mean() + bg_loss.mean()) / 2.0

        return total_loss

    def replay_loss(self):
        sample = self.memory_replay.sample_random()
        masks, iou_predictions, low_res_masks = self.sam_predictor.predict_with_gradient(
            point_coords=sample["point_coords"],
            point_labels=sample["point_labels"],
            mask_input=sample["mask_input"],
            multimask_output=False
        )
        refined_mask = sample["refined_mask"]
        image_bce = self.balanced_ce(gt_mask=refined_mask, low_res_logits=low_res_masks)
        return image_bce



if __name__ == "__main__":
    import config
    my_eval = Evaluator(config.Config())
    my_eval.adaptive_test()
