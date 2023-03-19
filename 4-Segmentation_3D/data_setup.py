
from pathlib import Path

import torch
import torchio as tio
import numpy as np
import imgaug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

def change_img_to_label_path(path):
    """
    Replace data with mask to get the masks
    """
    parts = list(path.parts)
    parts[parts.index("imagesTr")] = "labelsTr"
    return Path(*parts)


def create_datasets(path):

    subjects_paths = list(path.glob("la_*"))
    subjects = []

    for subject_path in subjects_paths:
        label_path = change_img_to_label_path(subject_path)
        subject = tio.Subject({"CT":tio.ScalarImage(subject_path), "Label":tio.LabelMap(label_path)})
        subjects.append(subject)

    for subject in subjects:
        assert subject["CT"].orientation == ("R", "A", "S")

    process = tio.Compose([
                tio.CropOrPad((256, 256, 140)),
                tio.RescaleIntensity((-1, 1))
                ])


    augmentation = tio.RandomAffine(scales=(0.9, 1.1), degrees=(-10, 10))


    val_transform = process
    train_transform = tio.Compose([process, augmentation])

    train_dataset = tio.SubjectsDataset(subjects[:16], transform=train_transform)
    val_dataset = tio.SubjectsDataset(subjects[16:], transform=val_transform)
    
    return train_dataset, val_dataset
