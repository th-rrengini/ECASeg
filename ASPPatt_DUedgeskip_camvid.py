# Imports
import torch
from torchvision import transforms as T
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import gc
import numpy as np 
from camvid_edge_model import UNET_edge
from camvid_datagen import CamVid
import torch.nn as nn
from torchinfo import summary
from torch.nn.functional import relu
import time
from metrics import scores, evaluate_seg
import torch.nn.functional as F
from torch.nn.functional import softmax, relu, sigmoid
import os
from plot_metrics import plot_curves_seg, save_scores

#*********************************************************************************************************************************************************************************

# Image transform: Resize and convert to tensor
image_transform = T.Compose([
    T.Resize((384, 512)),
    T.ToTensor()
    ])

# Color dictionary to colorize masks
color_dict_camvid = {0: (0.275, 0.510, 0.706), #sky
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

# List of object classes in order of their IDs as present in masks
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

# Function to colourize the bitmask
def colorize_bitmask(bitmask, class_colors):
   
    height, width = bitmask.shape
    colorized_image = np.zeros((height, width, 3)) # create a 0 array that is of the shape of the bitmask

    for key, color in class_colors.items():
        mask = (bitmask == key)
        colorized_image[mask]= color # assign the color based on the key

    return colorized_image # return colorized image

# Function to convert bitmasks into one-hot-encoded numpy arrays
def one_hot_encoder(bitmask, class_colors):

    height, width = bitmask.shape

    one_hot_vector_list = []

    for key in class_colors:
        object_mask = np.zeros((height, width))
        object_loc = bitmask == key
        object_mask[object_loc] = 1 # set the location where where key is to 1
        one_hot_vector_list.append(object_mask)

    return np.dstack(one_hot_vector_list) # dstack the lists to create a 12 dimensional array 

#*********************************************************************************************************************************************************************************

# Initialize the edge model with pre-trained weights and freeze
edge_model = UNET_edge() 
edge_model_path = 'path_to_pretrained_edge_model_ckpt'
checkpoint = torch.load(edge_model_path)
edge_model.load_state_dict(checkpoint['model_state_dict'])
edge_model.eval()

#*********************************************************************************************************************************************************************************

# Creating model blocks
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.c1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.c2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, x):
        x = self.bn(self.c1(x))
        x = relu(x)
        x = relu(self.c2(x))
        return x
    

class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv2d(ch_in, ch_out,
                                         kernel_size=3,stride=1,
                                         padding=1, bias=False),
                                nn.BatchNorm2d(ch_out)
        )
        
    def forward(self, x):
        x = relu(self.up(x))
        return x
    
#Attention mechanism
class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(f_int))
        
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(f_int))
        
        self.psi_xg = nn.Sequential(nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(1))
        
        
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = relu(g1+x1)
        psi = sigmoid(self.psi_xg(psi))
        
        return psi*x

# ASPP Module with dilation rates 6, 12, 18
class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(512)

        self.conv_3x3_1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(512)

        self.conv_3x3_2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(512)

        self.conv_3x3_3 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(512)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(512)

        self.conv_1x1_3 = nn.Conv2d((5*512), 1024, kernel_size=1) 
        self.bn_conv_1x1_3 = nn.BatchNorm2d(1024)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2] 
        feature_map_w = feature_map.size()[3] 

        out_1x1 = relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) 
        out_3x3_1 = relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) 
        out_3x3_2 = relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) 
        out_3x3_3 = relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) 

        out_img = self.avg_pool(feature_map) 
        out_img = relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) 
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") 

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) 
        out = relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) 

        return out

#*********************************************************************************************************************************************************************************

# Defining the model
class ECASeg(nn.Module):
    def __init__(self, edge_model, n_classes):
        super(ECASeg, self).__init__()
        self.edge_model = edge_model
        self.target_layers =  ['pool1','pool2','pool3', 'pool4', 'conv1', 'conv2', 'conv3', 'conv4'] 
        
        # Freeze the edge UNet
        for param in self.edge_model.parameters():
            param.requires_grad = False

        self.n_classes = n_classes
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = ConvBlock(ch_in=3, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)
        
        self.aspp = ASPP()
        self.up5 = UpConvBlock(ch_in=1024, ch_out=512)
        self.att5 = AttentionBlock(f_g=512, f_l=512, f_int=256)
        self.upconv5 = ConvBlock(ch_in=1536, ch_out=512)


        self.up4 = UpConvBlock(ch_in=512, ch_out=256)
        self.att4 = AttentionBlock(f_g=256, f_l=256, f_int=128)
        self.upconv4 = ConvBlock(ch_in=768, ch_out=256)

        
        self.up3 = UpConvBlock(ch_in=256, ch_out=128)
        self.dec3 = ConvBlock(ch_in = 384, ch_out=128)

        self.up2 = UpConvBlock(ch_in=128, ch_out=64)
        self.dec2 = ConvBlock(ch_in = 192, ch_out=64)

        self.LE13 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.LE23 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.LE33 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.LE43 = nn.Conv2d(1024, 512, kernel_size=1, padding=0)

        self.output_layer = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

        for name, module in self.edge_model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(lambda module, input, output, name=name: self.hook(module, input, output, name))


    def hook(self, module, input, output, name):
        self.intermediate_outputs[name] = output

    
    def forward(self, imgs):
        self.intermediate_outputs = {}
        _ = self.edge_model(imgs)

        skip1 = self.conv1(imgs)
        e = self.maxpool(skip1)
        x = torch.cat([e, self.intermediate_outputs['pool1']], dim=1)
        e = relu(self.LE13(x))

        skip2 = self.conv2(e)
        e = self.maxpool(skip2)
        x = torch.cat([e, self.intermediate_outputs['pool2']], dim=1)
        e = relu(self.LE23(x))

        skip3 = self.conv3(e)
        e = self.maxpool(skip3)
        x = torch.cat([e, self.intermediate_outputs['pool3']], dim=1)
        e = relu(self.LE33(x))

        skip4 = self.conv4(e)
        e = self.maxpool(skip4)
        x = torch.cat([e, self.intermediate_outputs['pool4']], dim=1)
        e = relu(self.LE43(x))

        e = self.conv5(e)
        e = self.aspp(e)

        # decoder + concat
        d5 = self.up5(e)
        x4 = self.att5(g=d5, x=skip4)
        d5 = torch.cat((x4, d5, self.intermediate_outputs['conv4']), dim=1)
        d5 = self.upconv5(d5)
        
        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=skip3)
        d4 = torch.cat((x3, d4, self.intermediate_outputs['conv3']), dim=1)
        d4 = self.upconv4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat((skip2, d3, self.intermediate_outputs['conv2']), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((skip1, d2, self.intermediate_outputs['conv1']), dim=1)
        d2 = self.dec2(d2)
        out = self.output_layer(d2)
        
        return out

#*********************************************************************************************************************************************************************************

# Setup model and print summary
model = ECASeg(edge_model, n_classes=12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
edge_model.to(device)
model.to(device)
try:
    input_size = (1, 3, 384, 512)
    summary(model, input_size=input_size, device='cuda')
except Exception as e1:
    print(f"Unexpected error during model summary: {e1}")

folder_path = 'path_to_folder'
checkpoint_path = folder_path+'checkpoints/'

# Hyperparameters
patience = 15
num_epochs = 150
batch_size = 4
max_lr = 0.001
weight_decay = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

# Setup train and validation dataloaders
train_data = CamVid('path_to_train_imgs',
                    'path_to_train_masks',
                    color_dict=color_dict_camvid, img_transform=image_transform)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle = True, drop_last = True)



val_data = CamVid('path_to_val_imgs',
                  'path_to_val_masks',
                  color_dict=color_dict_camvid, img_transform=image_transform)

val_loader = torch.utils.data.DataLoader(val_data,batch_size=batch_size,shuffle = True, drop_last = True)

#*********************************************************************************************************************************************************************************

print(f"Training started for {num_epochs} epochs", flush = True)
print('\n', flush=True)
train_time = time.time()
min_loss = 2000
no_improve = 0


train_losses_list = []
train_accs_list = []
train_mious_list =[]
val_losses_list = []
val_accs_list = []
val_mious_list = []


for epoch in range(num_epochs):
    since = time.time()
    gc.collect()
    torch.cuda.empty_cache()
    train_loss = 0
    val_loss = 0
    train_labels = []
    train_preds = []
    val_labels = []
    val_preds = []

    # Train Phase
    for imgs,masks in train_loader:
        model.train()
        imgs = imgs.to(device)
        masks = masks.to(device)
        output = model(imgs)
        del imgs
        torch.cuda.empty_cache()
        loss = loss_fn(output, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        output = softmax(output,dim=1)
        predicted_classes = torch.argmax(output, dim=1)
        actual_classes = torch.argmax(masks.to('cpu'), dim=1)
        train_labels.extend(actual_classes.cpu().numpy())
        train_preds.extend(predicted_classes.cpu().numpy())
    train_scores = scores(train_labels, train_preds, n_class=12, ignore_class=11)
    torch.cuda.empty_cache()

    # Validation Phase
    for imgs,masks in val_loader:
        model.eval()
        with torch.no_grad():
            imgs = imgs.to(device)
            masks = masks.to(device)
            output = model(imgs)
            del imgs
            torch.cuda.empty_cache()
            loss = loss_fn(output, masks)
            val_loss += loss.item()
            output = softmax(output,dim=1)
            predicted_classes = torch.argmax(output, dim=1)
            actual_classes = torch.argmax(masks.to('cpu'), dim=1)
            val_labels.extend(actual_classes.cpu().numpy())
            val_preds.extend(predicted_classes.cpu().numpy())
    val_scores = scores(val_labels, val_preds, n_class=12, ignore_class=11)
    torch.cuda.empty_cache()


    if (epoch+1) % 5 == 0 and (epoch+1)>=75:
            print('Saving model...', flush=True)
            checkpoint_name = f'checkpoint_epoch_{epoch+1}.pt'
            checkpoint_filepath = checkpoint_path+checkpoint_name
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss, 'val_loss': val_loss
                        }, checkpoint_filepath)  
    

    if min_loss>(val_loss/len(val_loader)) and np.abs(((min_loss*100)-(val_loss/len(val_loader)))/min_loss) > 10:
        print('Loss Decreasing... {:.3f} >> {:.3f} '.format(min_loss, (val_loss/len(val_loader))), flush=True)
        min_loss = val_loss/len(val_loader)
        if no_improve > 0:
            no_improve = 0
            print('Saving best weights at present.', flush=True)
            checkpoint_name = f'restore_best_weights.pt'
            checkpoint_filepath = checkpoint_path+checkpoint_name
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss, 'val_loss': val_loss
                        }, checkpoint_filepath)                   
    else:
        no_improve += 1
        min_loss = val_loss/len(val_loader)
        print(f'Loss did not decrease for {no_improve} epochs.', flush=True)
        if no_improve == patience:
            print(f'Early stopping at epoch {epoch+1}.', flush=True)
            break
    
    # Accumulate metrics
    train_losses_list.append(train_loss/len(train_loader))
    train_accs_list.append(train_scores["Pixel Accuracy"]*100)
    train_mious_list.append(train_scores["Mean IoU"]*100)
    val_losses_list.append(val_loss/len(val_loader))
    val_accs_list.append(val_scores["Pixel Accuracy"]*100)
    val_mious_list.append(val_scores["Mean IoU"]*100)

    save_scores(train_scores, folder_path+'train_scores.json')
    save_scores(val_scores, folder_path+'val_scores.json')

    # Print train and validation stats
    print(f'Epoch {epoch + 1}/{num_epochs} --> Train Stats - Loss : {train_loss/len(train_loader)} Scores: {train_scores}', flush=True)
    print(f'Epoch {epoch + 1}/{num_epochs} --> Val Stats - Loss : {val_loss/len(val_loader)} Scores : {val_scores}', flush=True)
    print(f'Time: {(time.time()-since)/60}', flush=True)
    print('\n', flush=True)
   

print(f'Total time taken: {(time.time()-train_time)/60}')
print('Training ended!')


# Save final version of model
try:
    torch.save(model, folder_path+'model.pth')
    print(f"Model successfully saved to {folder_path+'model.pth'}")

except RuntimeError as e:
    print(f"RuntimeError during model save: {e}")
except IOError as e:
    print(f"IOError during model save: {e}")
except Exception as e:
    print(f"Unexpected error during model save: {e}")

# Train and val loss plot, acc plot, miou plot after training is completed
plot_curves_seg(num_epochs, train_losses_list, val_losses_list, train_accs_list, val_accs_list, train_mious_list, val_mious_list, filename = folder_path+'metrics_plot.png')

#*********************************************************************************************************************************************************************************

# Running evaluation on test set
# Setup test dataloader
test_data = CamVid('path_to_test_imgs',
                   'path_to_val_masks',
                    color_dict=color_dict_camvid, img_transform=image_transform)

test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle = True, drop_last = True)


# Load model from a specific checkpoint
model_checkpoint_path = 'path_to_ckpt'
if os.path.isfile(model_checkpoint_path):
    model = ECASeg(edge_model, n_classes=12)
    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Test Stats for model version at {model_checkpoint_path}", flush = True)
    evaluate_seg(model, test_loader)
    
    # Visualization of model predictions on an unseen images
    torch.cuda.empty_cache()
    for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            preds = softmax(preds,dim=1)
            preds = torch.argmax(preds, dim = 1)
            masks = torch.argmax(masks, dim = 1)

            imgs = imgs[:5,:,:,:]
            masks = masks[:5,:,:]
            preds = preds[:5,:,:]

            fig, axes = plt.subplots(5, 3, figsize=(25, 20))
            for i, (image, gt_mask, pred_mask) in enumerate(zip(imgs, masks, preds)):
                image = image.permute(1,2,0).cpu().numpy()
                gt_mask = gt_mask.cpu().numpy()
                pred_mask = pred_mask.cpu().numpy()
                gt_mask = colorize_bitmask(gt_mask,color_dict_camvid)
                pred_mask = colorize_bitmask(pred_mask,color_dict_camvid)
                ax = axes[i]
                ax[0].imshow(image)
                ax[1].imshow(gt_mask)
                ax[2].imshow(pred_mask)
                ax[0].axis('off')
                ax[1].axis('off')
                ax[2].axis('off')

            axes[0][0].set_title('Images', fontsize=16)
            axes[0][1].set_title('Ground Truth Masks', fontsize=16)
            axes[0][2].set_title('Predictions', fontsize=16)

            plt.tight_layout()
            plt.savefig(folder_path+'ecaseg_test_preds.png')
            plt.close()
            break

#*********************************************************************************************************************************************************************************
