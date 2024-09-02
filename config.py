
class Config:
    def __init__(self):
        self.sam_input_image_size = 1024
        self.image_level_adaption = True
        self.click_level_adaption = True
        self.reset_decoder_after_each_image = True
        self.use_dense_mask_for_img_adapt = True

        self.filter_result_mask = 1 # 0 : No filtering | 1 : Using erosion | 2 : Using confidence
        self.delta_conf = 0.4 # Only meaningful for confidence filtering (self.filter_result_mask = 2)
        self.erosion_iters = 5 # Only meaningful for erosion filtering (self.filter_result_mask = 1)

        self.replay_size = 5 # If negative -> no replay
        self.learning_rate = 1e-6 # In case of doubt: A solid value was 1e-6

        self.hq_token_use = True

        # The possible values for sports_tools_class are "flickr_bobsleighs", "flickr_curling_stone",
        # "flickr_ski_helmets", "flickr_snow_kites",  "yt_sports", "flickr_curling_brooms",
        # "flickr_ski_goggles", "flickr_ski_misc", "flickr_slalom_obstacles", "flickr_snowboards"
        self.sports_tools_class = "flickr_ski_misc"

        self.backbone_size = "b" # "b" for the regular SAM and "b_hq" in order to use HQ-SAM

        self.dataset_name = "sports_tools"
        self.min_iou = 0.85
        self.max_clicks = 20
        self.device = "cuda:1"

        self.reset()


    def reset(self):
        self.dataset_path = "./sports_tools"

        if self.backbone_size == "b":
            self.checkpoint = "./weights/sam_vit_b_01ec64.pth"
        elif self.backbone_size == "b_hq":
            self.checkpoint = "./weights/sam_hq_vit_b.pth"
        else:
            raise NotImplementedError

