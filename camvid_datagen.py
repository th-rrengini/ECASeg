from matplotlib import pyplot as plt
import os
import numpy as np
import re
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import torch

image_transform = T.Compose([
    T.Resize((384, 512)),
    T.ToTensor()
    ])

color_dict = {0: (0.275, 0.510, 0.706), #sky
              1: (0.275, 0.275, 0.275), #building
              2: (0.6, 0.6, 0.6), #column_pole
              3: (0.502, 0.251, 0.502), #road
              4: (0.957, 0.137, 0.910), #sidewalk
              5: (0.420, 0.557, 0.137), #tree
              6: (0.980, 0.667, 0.118), #trafficLight
              7: (0.745, 0.6, 0.6), #fence
              8: (0.0, 0.0, 0.557), #car
              9: (0.863, 0.078, 0.235), #pedestrian
              10: (0.467, 0.043, 0.125), #bicyclist
              11:  (0, 0, 0)} #void


object_class = ["Sky", #Sky
                "Building", #Archway, Bridge, Building, Tunnel, Wall
                "Column_Pole", #Column_Pole, Traffic Cone
                "Road", #Road, LaneMkgsDriv, LaneMkgsNonDriv
                "Sidewalk",	#Sidewalk, ParkingBlock, RoadShoulder
                "Tree",	#Tree, VegetationMisc
                "TrafficLight", #TrafficLight, Misc_Text, SignSymbol
                "Fence", #Fence
                "Car",	#Car, OtherMoving, SUVPickupTruck, Train, Truck_Bus
                "Pedestrian", #Animal, CartLuggagePram, Child, Pedestrain
                "Bicyclist", #Bicyclist, MotorcycleScooter
                "Void"]	

def one_hot_encoder(bitmask, class_colors):

    height, width = bitmask.shape

    one_hot_vector_list = []

    for key in class_colors:
        object_mask = np.zeros((height, width))
        object_loc = bitmask == key
        object_mask[object_loc] = 1 # set the location where where key is to 1
        one_hot_vector_list.append(object_mask)

    return np.dstack(one_hot_vector_list) # dstack the lists to create a 20 dimensional array each array for a class

    
class CamVid(Dataset):
    def __init__(self,image_folder,mask_folder,color_dict, img_transform):
        self.image_folder = image_folder
        self.image_files = os.listdir(image_folder)
        self.mask_folder = mask_folder
        self.color_dict = color_dict
        self.image_transform = img_transform

    def __len__(self):
        return(len(self.image_files))
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_folder,self.image_files[index])
        mask_path = os.path.join(self.mask_folder, re.sub('.jpg', '.png', self.image_files[index]))
        image = Image.open(img_path)
        if self.image_transform:
            image = self.image_transform(image)
        mask = Image.open(mask_path)
        resize_transform = T.Resize((384, 512),interpolation=Image.NEAREST)
        resized_mask = resize_transform(mask)
        resized_mask = np.array(resized_mask)
        one_hot_bitmask = one_hot_encoder(resized_mask, self.color_dict)
        one_hot_bitmask = torch.Tensor(one_hot_bitmask)
        one_hot_bitmask = one_hot_bitmask.permute(2,0,1)

        return image, one_hot_bitmask