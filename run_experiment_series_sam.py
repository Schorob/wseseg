import evaluator as evaluator
import config as config
import numpy as np


#################################################################
######################## Config creators ########################
#################################################################
def alter_config_for_full_model_erosion(config_instance):
    """
    Alters the config file to correspond to the full model configuration.
    More precisely:
        - Only the decoder is adapted.
        - There is no form of regularization.
        - During the treatment of a single image, we use each given click as
        a GT to perform a single iteration on.
        - Directly after annotating/testing on a single image, the decoder parameters are
        reset to their state before the image.
        - After that, we subject the IntSeg-result mask to erosion. Then we carry out a single
        training step on the decoders parameters using the eroded mask.
    """

    config_instance.image_level_adaption = True
    config_instance.click_level_adaption = True
    config_instance.reset_decoder_after_each_image = True
    config_instance.use_dense_mask_for_img_adapt = True
    config_instance.filter_result_mask = 1
    config_instance.erosion_iters = 5
    config_instance.output_regularization = False
    config_instance.output_reg_coeff = 0.0
    config_instance.use_predictor = True

    return config_instance

def alter_config_for_full_model_conf(config_instance):
    """
    Alters the config file to correspond to the full model configuration.
    More precisely:
        - Only the decoder is adapted.
        - There is no form of regularization.
        - During the treatment of a single image, we use each given click as
        a GT to perform a single iteration on.
        - Directly after annotating/testing on a single image, the decoder parameters are
        reset to their state before the image.
        - After that, we subject the IntSeg-result mask to confidence based filtering. Then we carry out a single
        training step on the decoders parameters using the filtered mask.
    """

    config_instance.image_level_adaption = True
    config_instance.click_level_adaption = True
    config_instance.reset_decoder_after_each_image = True
    config_instance.use_dense_mask_for_img_adapt = True
    config_instance.filter_result_mask = 2
    config_instance.delta_conf = 0.4
    config_instance.output_regularization = False
    config_instance.output_reg_coeff = 0.0
    config_instance.use_predictor = True

    return config_instance

def alter_config_no_adaptation(config_instance):
    """
    We simply test on the model; there is no adaptation whatsoever.
    """

    config_instance.image_level_adaption = False
    config_instance.click_level_adaption = False
    config_instance.reset_decoder_after_each_image = False
    config_instance.use_dense_mask_for_img_adapt = False
    config_instance.use_predictor = True

    return config_instance

def alter_config_clickwise_resets(config_instance):
    """
    Alters the config file to correspond to the full model configuration.
    More precisely:
        - Only the decoder is adapted.
        - There is no form of regularization.
        - During the treatment of a single image, we use each given click as
        a GT to perform a single iteration on.
        - The decoder parameters are reset after each image, since the decoder
        has effectively overfit to the current image.
    """

    config_instance.image_level_adaption = False
    config_instance.click_level_adaption = True
    config_instance.reset_decoder_after_each_image = True
    config_instance.use_dense_mask_for_img_adapt = False
    config_instance.filter_result_mask = 2
    config_instance.delta_conf = 0.4
    config_instance.output_regularization = False
    config_instance.output_reg_coeff = 0.0
    config_instance.use_predictor = True

    return config_instance


def alter_config_clickwise_noresets(config_instance):
    """
    Alters the config file to correspond to the full model configuration.
    More precisely:
        - Only the decoder is adapted.
        - There is no form of regularization.
        - During the treatment of a single image, we use each given click as
        a GT to perform a single iteration on.
        - The decoder parameters are kept instead of being reset. The
        decoder thus learns from a continuous stream of clicks.
    """

    config_instance.image_level_adaption = False
    config_instance.click_level_adaption = True
    config_instance.reset_decoder_after_each_image = False
    config_instance.use_dense_mask_for_img_adapt = False
    config_instance.filter_result_mask = 2
    config_instance.delta_conf = 0.4
    config_instance.output_regularization = False
    config_instance.output_reg_coeff = 0.0
    config_instance.use_predictor = True

    return config_instance

def alter_config_resmaskonly_erosion(config_instance):
    """
    Alters the config file to correspond to the full model configuration.
    More precisely:
        - Only the decoder is adapted.
        - There is no form of regularization.
        - Directly after annotating/testing on a single image, the decoder parameters are
        reset to their state before the image.
        - After that, we subject the IntSeg-result mask to erosion. Then we carry out a single
        training step on the decoders parameters using the eroded mask.
    """

    config_instance.image_level_adaption = True
    config_instance.click_level_adaption = False
    config_instance.reset_decoder_after_each_image = False
    config_instance.use_dense_mask_for_img_adapt = True
    config_instance.filter_result_mask = 1 # Erosion
    config_instance.erosion_iters = 5
    config_instance.output_regularization = False
    config_instance.output_reg_coeff = 0.0
    config_instance.use_predictor = True

    return config_instance


def alter_config_resmaskonly_confidence(config_instance):
    """
    Alters the config file to correspond to the full model configuration.
    More precisely:
        - Only the decoder is adapted.
        - There is no form of regularization.
        - Directly after annotating/testing on a single image, the decoder parameters are
        reset to their state before the image.
        - After that, we subject the IntSeg-result mask to erosion. Then we carry out a single
        training step on the decoders parameters using the eroded mask.
    """

    config_instance.image_level_adaption = True
    config_instance.click_level_adaption = False
    config_instance.reset_decoder_after_each_image = False
    config_instance.use_dense_mask_for_img_adapt = True
    config_instance.filter_result_mask = 2
    config_instance.delta_conf = 0.4
    config_instance.output_regularization = False
    config_instance.output_reg_coeff = 0.0
    config_instance.use_predictor = True

    return config_instance

def alter_config_clickmaskonly(config_instance):
    """
    Alters the config file to correspond to the full model configuration.
    More precisely:
        - Only the decoder is adapted.
        - There is no form of regularization.
        - Directly after annotating/testing on a single image, the decoder parameters are
        reset to their state before the image.
        - After that, we create a mask from all the accumulated clicks. Then we carry out a single
        training step on the decoders parameters using the click-mask.
    """

    config_instance.image_level_adaption = True
    config_instance.click_level_adaption = False
    config_instance.reset_decoder_after_each_image = False
    config_instance.use_dense_mask_for_img_adapt = False
    config_instance.filter_result_mask = 1 # Erosion
    config_instance.erosion_iters = 5
    config_instance.output_regularization = False
    config_instance.output_reg_coeff = 0.0
    config_instance.use_predictor = True

    return config_instance


def alter_config_for_full_model_click_mask_only(config_instance):
    """
    Alters the config file to correspond to the full model configuration.
    More precisely:
        - Only the decoder is adapted.
        - There is no form of regularization.
        - During the treatment of a single image, we use each given click as
        a GT to perform a single iteration on.
        - Directly after annotating/testing on a single image, the decoder parameters are
        reset to their state before the image.
        - After that, we create a mask from all the accumulated clicks. Then we carry out a single
        training step on the decoders parameters using the click-mask.
    """

    config_instance.image_level_adaption = True
    config_instance.click_level_adaption = True
    config_instance.reset_decoder_after_each_image = True
    config_instance.use_dense_mask_for_img_adapt = False
    config_instance.filter_result_mask = 1
    config_instance.erosion_iters = 5
    config_instance.output_regularization = False
    config_instance.output_reg_coeff = 0.0
    config_instance.use_predictor = True

    return config_instance


def alter_config_for_full_model_nomasktreatment(config_instance):
    """
    Alters the config file to correspond to the full model configuration.
    More precisely:
        - Only the decoder is adapted.
        - There is no form of regularization.
        - During the treatment of a single image, we use each given click as
        a GT to perform a single iteration on.
        - Directly after annotating/testing on a single image, the decoder parameters are
        reset to their state before the image.
        - After that, we train on the IntSeg mask (no addtional mask treatment).
    """

    config_instance.image_level_adaption = True
    config_instance.click_level_adaption = True
    config_instance.reset_decoder_after_each_image = True
    config_instance.use_dense_mask_for_img_adapt = True
    config_instance.filter_result_mask = 0 # <- Important
    config_instance.output_regularization = False
    config_instance.output_reg_coeff = 0.0
    config_instance.use_predictor = True

    return config_instance



def run_sports_tools():
    #HYPERPARAMETERS
    BACKBONE_SIZE = "b_hq" # "b" for SAM; "b_hq" for HQ-SAM
    DATASET_NAME = "sports_tools" # There only is the sport_tools dataset
    MAX_CLICKS = 20 # The maximum number of clicks
    MIN_IOU = 0.85 # The IoU threshold to be reached
    DEVICE = "cuda:2" # The string representing the devices you want to use (torch.device(...))
    LEARNING_RATE = 5e-8 # For SAM 5e-8; for HQ-SAM 1e-6
    EROSION_ITERS = 4 # The number of iterations for iterative erosion
    DELTA_CONF = 0.45 # The distance from the decision boundary for confidence thresholding
    RUN_ID = 9 # Which of the proposed configurations from the array RUNS is to be used

    # Select one of the possible run configurations by setting the hyperparameter RUND_ID
    RUNS = [
        ("Full Model with Erosion: ", alter_config_for_full_model_erosion), # Table row 2
        ("No adaptation at all: ", alter_config_no_adaptation), # Table row 1
        ("Clickwise only (with resets): ", alter_config_clickwise_resets), # Table row 4
        ("Clickwise only (without resets): ", alter_config_clickwise_noresets), # Table row 5
        ("Result-Mask only with Erosion: ", alter_config_resmaskonly_erosion), # Table row 6
        ("Accumulated click-mask only: ", alter_config_clickmaskonly), # Table row 8
        ("Full Model with Click Mask: ", alter_config_for_full_model_click_mask_only), # Table row 9
        ("Full Model with untreated result mask: ", alter_config_for_full_model_nomasktreatment), # Table row 10
        ("Full Model with confidence threshold: ", alter_config_for_full_model_conf), # Table row 3
        ("Result-Mask only with confidence threshold: ", alter_config_resmaskonly_confidence) # Table row 7
    ]
    row_text, config_altering_function = RUNS[RUN_ID]

    # We will automatically iterate through this list
    # If you don't want to use the skis in the ski jumping context, comment out "yt_sports"
    SPORTS_TOOLS_CLASSES = list([
        "yt_sports",
        "flickr_bobsleighs", "flickr_curling_stone", "flickr_ski_helmets", "flickr_snow_kites",
        "flickr_curling_brooms", "flickr_ski_goggles", "flickr_ski_misc", "flickr_slalom_obstacles", "flickr_snowboards"
    ])


    # Create the initial config file
    config_instance = config.Config()
    config_instance.backbone_size = BACKBONE_SIZE
    config_instance.dataset_name = DATASET_NAME
    config_instance.max_clicks = MAX_CLICKS
    config_instance.min_iou = MIN_IOU
    config_instance.device = DEVICE

    # This is ALWAYS false in the main table
    config_instance.additional_bbox = False
    config_instance.use_bbox_in_mask = False

    config_instance.reset()

    nocs, frs = [], []

    for class_name in SPORTS_TOOLS_CLASSES:
        print("=" * 60)
        print("Sports Tools Class: ", class_name)
        config_instance = config_altering_function(config_instance)

        # Ensure all manual changes to the configuration
        config_instance.learning_rate = LEARNING_RATE
        config_instance.erosion_iters = EROSION_ITERS
        config_instance.delta_conf = DELTA_CONF

        config_instance.sports_tools_class = class_name
        current_evaluator = evaluator.Evaluator(config_instance)
        noc, fr = current_evaluator.adaptive_test()
        nocs.append(noc); frs.append(fr)

    # The printed string are formatted in a way that comports well
    # with copying them into a LaTeX tabular.
    row_string = ""
    for noc, fr in zip(nocs, frs):
        row_string = row_string + "{: .3f}".format(noc) + " & "
        row_string = row_string + "{: .2f}".format(fr) + " & "

    row_string = row_string + str(np.array(nocs).mean()) + " & "
    row_string = row_string + str(np.array(frs).mean()) + " & "

    print(row_string)



if __name__ == "__main__":
    # Remember to set the hyperparameters in the run_sports_tools section.
    run_sports_tools()

