import io
import math
import random
import time
from pathlib import Path
from posixpath import expanduser
import os
import blobfile as bf
import numpy as np
import webdataset as wds
from braceexpand import braceexpand
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import random
import copy

def load_data(
    distr_backend,
    data_dir,
    batch_size,
    image_size,
    dataset_length,
    random_crop=False,
    random_flip=False,
    use_webdataset=False,
    num_workers=0,
    inpainting=False,
    set_data=True,
    set_size=15,
):
    rank = distr_backend.get_rank()
    num_shards = distr_backend.get_world_size()
    if use_webdataset:
        wds_urls = parse_data_dir(data_dir)
        print(f"wds_urls: {wds_urls}")
        ds = load_webdataset(distr_backend, resolution=image_size,
                             file_paths=wds_urls, batch_size=batch_size, random_crop=random_crop, random_flip=random_flip)
        return ds
    elif set_data:
        """
        Make dataset that gets image along with it's set excluding that image.
        Each set is placed in a different directory
        Steps:
        1. Make a program to get directory as key and then the files within and vice versa x
        2. Make ImageSetDataset which takes both dictionaries
        3. Computes the number of images to use based on batch with fewest set pics then randomly select
        """
        data_dir = expanduser(data_dir)
        set_dict = {}
        file_dict = {}
        for folder in os.listdir(data_dir):
            set_path = os.path.join(data_dir, folder)
            if not os.path.isdir(set_path):
                continue
            set_dict[folder] = _list_image_files_recursively(set_path)
            for file in set_dict[folder]:
                file_dict[file] = folder
        # print(file_dict)
        ds = ImageSetDataset(
            image_size,
            set_dict,
            file_dict,
            classes=None,
            shard=rank,
            num_shards=num_shards,
            random_crop=random_crop,
            random_flip=random_flip,
            inpainting=inpainting,
            set_size=set_size
        )
        print(f"Loaded {len(ds)} images on {distr_backend.get_world_size()} gpus")
        return ds
    else:
        # Change to painter dataset+single img. google cross attention
        data_dir = expanduser(data_dir)
        all_paths = _list_image_files_recursively(data_dir)
        print(f"Found {len(all_paths)} images in {data_dir}")
        ds = ImageDataset(
            image_size,
            all_paths,
            classes=None,
            shard=rank,
            num_shards=num_shards,
            random_crop=random_crop,
            random_flip=random_flip,
            inpainting=inpainting,
        )
        print(f"Loaded {len(ds)} images on {distr_backend.get_world_size()} gpus")
        return ds


def _list_image_files_recursively(data_dir, get_text_file=True):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        entry = entry.split(".")
        ext = entry[-1].strip()
        filename = entry[0]
        if ext and ext.lower() in ["jpg", "jpeg", "png", "gif", "webp"]:
            text_path = bf.join(data_dir, filename+'.txt')
            if not get_text_file:
                results.append(full_path)
            else:
                if bf.exists(text_path):
                    results.append((full_path, text_path))
                else:
                    results.append((full_path, ""))
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path, get_text_file=get_text_file))
    return results

class ImageSetDataset(Dataset):
    def __init__(
        self,
        resolution,
        set_dict,
        file_dict,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        inpainting=False,
        set_size=15
    ):
        super().__init__()
        self.resolution = resolution
        self.local_files = list(file_dict.keys())[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.inpainting = inpainting
        self.set_size = set_size
        self.set_dict = set_dict
        self.file_dict = file_dict
    def __len__(self):
        return len(self.local_files)
    def get_img(self, path):
        with bf.BlobFile(path[0], "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)
        if path[1] == '':
            text = ''
        else:
            with bf.BlobFile(path[1], "r") as f:
                text = f.read().strip().split("\n")
                text = random.choice(text)
        forbidden_words = ["left", "right", "up", "down"] # dont flip if directional
        if not any(word in text for word in forbidden_words):
            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        return text, np.transpose(arr, [2, 0, 1])
    def get_set_imgs(self, set_name, path):
        set_datas = []
        set_file_paths = copy.copy(self.set_dict[set_name])
        try:
            set_file_paths.remove(path)
        except:
            None
        random.shuffle(set_file_paths)
        set_file_paths = set_file_paths[:self.set_size]
        for set_file_path in set_file_paths:
            set_datas.append(self.get_img(set_file_path)[1])
        return np.array(set_datas)
    def __getitem__(self, idx):
        path = self.local_files[idx]
        set_name = self.file_dict[path]
        
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        
        text, arr = self.get_img(path)
        
        
        return text, self.get_set_imgs(set_name, path), arr

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        file_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        inpainting=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_files = file_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.inpainting = inpainting

    def __len__(self):
        return len(self.local_files)
    def __getitem__(self, idx):
        path = self.local_files[idx]
        with bf.BlobFile(path[0], "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        if self.inpainting:
            imgs = np.array(pil_image)
            w = pil_image.shape[1] // 2
            masked_image, pil_image = imgs[:, :w], imgs[:, w:2*w]
            pil_image = Image.PIL.fromarray(pil_image)
        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if path[1] == '':
            text = ''
        else:
            with bf.BlobFile(path[1], "r") as f:
                text = f.read().strip().split("\n")
                text = random.choice(text)

        forbidden_words = ["left", "right", "up", "down"] # dont flip if directional
        if not any(word in text for word in forbidden_words):
            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        return text, np.transpose(arr, [2, 0, 1])


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(
        min_smaller_dim_size, max_smaller_dim_size + 1)

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def clean_caption(caption):
    caption = caption.decode("utf-8")
    caption = caption.replace("\n", " ").replace(
        "\t", " ").replace("\r", " ").replace("  ", " ")
    caption = caption.strip()
    return caption


def load_webdataset(
    distr_backend,
    resolution,
    file_paths,
    batch_size,
    random_crop=False,
    random_flip=False,
):
    def filter_by_item(item):
        if mycap not in item: return False
        if myimg not in item: return False
        return True

    def pil_transform_to_np(arr):
        if random_crop:
            arr = random_crop_arr(arr, resolution)
        else:
            arr = center_crop_arr(arr, resolution)
        if random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        arr = arr.astype(np.float32) / 127.5 - 1
        return np.transpose(arr, [2, 0, 1])

    def bytes_to_pil_image(item): 
        pil_image = Image.open(io.BytesIO(item)).convert("RGB")
        pil_image.load()
        return pil_image

    myimg, mycap = "jpg", "txt"
    image_text_mapping = {
        mycap: clean_caption,
        myimg: bytes_to_pil_image
    }
    image_mapping = {myimg: pil_transform_to_np}
    dataset = wds.WebDataset(urls=file_paths, handler=wds.warn_and_continue, nodesplitter=wds.split_by_worker)
    filtered_dataset = dataset.select(filter_by_item)
    dataset = filtered_dataset.map_dict(**image_text_mapping).map_dict(**image_mapping).to_tuple(mycap, myimg).batched(batch_size, partial=True)
    return dataset

def parse_data_dir(data_dir):
    if Path(data_dir).is_dir():
        wds_uris = [str(p) for p in Path(data_dir).glob(
            "**/*") if ".tar" in str(p).lower()]  # .name
        assert len(
            wds_uris) > 0, 'The directory ({}) does not contain any WebDataset/.tar files.'.format(data_dir)
        print('Found {} WebDataset .tar(.gz) file(s) under given path {}!'.format(
            len(wds_uris), data_dir))
    elif ('http://' in data_dir.lower()) | ('https://' in data_dir.lower()):
        wds_uris = f"pipe:curl -L -s {data_dir} || true"
        print('Found {} http(s) link under given path!'.format(
            len(wds_uris), data_dir))
    elif 'gs://' in data_dir.lower():
        wds_uris = f"pipe:gsutil cat {data_dir} || true"
        print('Found {} GCS link under given path!'.format(
            len(wds_uris), data_dir))
    elif '.tar' in data_dir:
        wds_uris = data_dir
        print('Found WebDataset .tar(.gz) file under given path {}!'.format(data_dir))
    else:
        raise Exception(
            'No folder, no .tar(.gz) and no url pointing to tar files provided under {}.'.format(data_dir))
    return wds_uris
