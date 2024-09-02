from torch.utils import data
from PIL import Image
import numpy as np
from pathlib import Path

CLASS_NAMES = [
    "flickr_bobsleighs", "flickr_curling_stone", "flickr_ski_helmets", "flickr_snow_kites",  "yt_sports",
    "flickr_curling_brooms", "flickr_ski_goggles", "flickr_ski_misc", "flickr_slalom_obstacles", "flickr_snowboards"
]

class SportsTools(data.Dataset):
    def __init__(self, dataset_path, class_name):
        super(SportsTools, self).__init__()
        assert class_name in CLASS_NAMES
        self.class_name = class_name
        if class_name == "yt_sports":
            self.dataset_path = Path(dataset_path) / "yt_sports/skijump/annotated_frames/0"
        else:
            self.dataset_path = Path(dataset_path) / class_name

        self.mask_files = list(sorted(self.dataset_path.glob("mask_*.png")))
        self.dataset_samples = []

        for midx, mfile in enumerate(self.mask_files):
            mask = np.array(Image.open(mfile))
            item_idxs = np.unique(mask)[1:]
            for iidx in item_idxs:
                self.dataset_samples.append((midx, iidx))



    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, item):
        # 1. Obtain the mask and create an int32-0/1-mask
        midx, iidx = self.dataset_samples[item]
        mask_path = self.mask_files[midx]
        mask = np.array(Image.open(str(mask_path)))
        mask = np.where(mask == iidx, 1, 0)
        mask = mask.astype(np.int32)

        # 2. Load the image
        image_path = str(mask_path)[:-4] + ".jpg" # Change the extension
        image_path = image_path.split("/")
        image_path[-1] = image_path[-1][5:]
        image_path = "/".join(image_path)
        image = np.array(Image.open(str(image_path))) / 255.0

        if len(image.shape) == 2:
            image = np.stack([
                image, image, image
            ], axis=2)
        image = image.astype(np.float32)


        return image, mask





