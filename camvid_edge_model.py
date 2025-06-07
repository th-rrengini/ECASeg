# Imports
import torch
from torchvision import transforms as T
from torch.nn.functional import sigmoid, relu
import torch.nn as nn
import gc
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from metrics import compute_Binary_miou, compute_recall, compute_BCEloss 
import numpy as np 
from torchinfo import summary
import time
import os
import cv2
import torchmetrics
from plot_metrics import plot_curves_edge, visualize_edge, evaluate_edge
print("imports done")

#*********************************************************************************************************************************************************************************
# CamVid dataset for generating pair of image and corresponding sobel edge mask
class CamVid(Dataset):
    def __init__(self,image_folder, img_transform):
        self.image_folder = image_folder
        self.image_files = os.listdir(image_folder)
        self.image_transform = img_transform

    def __len__(self):
        return(len(self.image_files))
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_folder,self.image_files[index])
        image = Image.open(img_path)
        edge = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Sobel operator in X direction
        sobel_x = cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=3)
        # Sobel operator in Y direction
        sobel_y = cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=3)
        # Combine the Sobel X and Y images
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
        sobel_combined = np.uint8(sobel_combined)
        sobel_combined = Image.fromarray(sobel_combined)
        if self.image_transform:
            image = self.image_transform(image)
            edge = self.image_transform(sobel_combined)
        edge[(edge>0.025)] = 1.0
        edge[(edge<=0.025)] = 0.0
        return image, edge

#*********************************************************************************************************************************************************************************
# Model blocks
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
    
# Attention mechanism    
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
    
#*********************************************************************************************************************************************************************************

# Edge model architecture
class UNET_edge(nn.Module):
    def __init__(self):
        super().__init__() 
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=3, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)
        
        self.up5 = UpConvBlock(ch_in=1024, ch_out=512)
        self.att5 = AttentionBlock(f_g=512, f_l=512, f_int=256)
        self.upconv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.up4 = UpConvBlock(ch_in=512, ch_out=256)
        self.att4 = AttentionBlock(f_g=256, f_l=256, f_int=128)
        self.upconv4 = ConvBlock(ch_in=512, ch_out=256)
        
        self.up3 = UpConvBlock(ch_in=256, ch_out=128)
        self.dec3 = ConvBlock(ch_in = 256, ch_out=128)
        
        self.up2 = UpConvBlock(ch_in=128, ch_out=64)
        self.dec2 = ConvBlock(ch_in = 128, ch_out=64)
        
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        
    def forward(self, img):
        # encoder
        skip1 = self.conv1(img)
        
        e = self.pool1(skip1)
        skip2 = self.conv2(e)
        
        e = self.pool2(skip2)
        skip3 = self.conv3(e)
        
        e = self.pool3(skip3)
        skip4 = self.conv4(e)
        
        e = self.pool4(skip4)
        e = self.conv5(e)
        
        # decoder + concat
        d5 = self.up5(e)
        x4 = self.att5(g=d5, x=skip4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.upconv5(d5)
        
        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=skip3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.upconv4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat((skip2, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((skip1, d2), dim=1)
        d2 = self.dec2(d2)
        
        out = self.output_layer(d2)
        
        return out
#*********************************************************************************************************************************************************************************

if __name__ == '__main__':
    # Image transform: Resize and convert to tensor
    image_transform = T.Compose([
        T.Resize((384, 512)),
        T.ToTensor()
        ])

    folder_path = 'path_to_folder'
    checkpoint_path = folder_path+'checkpoints/'

    # Setup edge model and print summary
    model = UNET_edge()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_size = (1, 3, 384, 512)  
    summary(model, input_size=input_size, device=device)

    # Hyperparameters
    num_epochs = 100
    patience = 8
    batch_size = 8
    max_lr = 0.001
    weight_decay = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    miou_metric = torchmetrics.classification.BinaryJaccardIndex()

    # Setup train and validation dataloaders
    train_data = CamVid('path_to_train_imgs',
                        img_transform=image_transform)

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle = True)


    val_data = CamVid('path_to_val_imgs',
                    img_transform=image_transform)

    val_loader = torch.utils.data.DataLoader(val_data,batch_size=batch_size,shuffle = True)

#*********************************************************************************************************************************************************************************

    # Train Phase
    print(f"Training started for {num_epochs} epochs", flush = True)
    print('\n', flush=True)
    train_time = time.time()
    min_loss = 2000
    no_improve = 0

    train_losses_list = []
    train_accs_list = []
    train_mious_list =[]
    train_recall_list = []
    val_losses_list = []
    val_accs_list = []
    val_mious_list = []
    val_recall_list = []

    for epoch in range(num_epochs):
        since = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        train_loss = 0
        train_acc = 0
        train_recall = 0
        train_miou = 0
        val_loss = 0
        val_acc = 0
        val_miou = 0
        val_recall = 0
    
    for imgs, edges in train_loader:
        model.train()
        imgs = imgs.to(device)
        edges = edges.to(device)
        loss, output = compute_BCEloss(model, imgs, edges, loss_fn)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        output = sigmoid(output)
        accuracy = (output.round() == edges).float().mean().to('cpu')
        recall = compute_recall(output, edges)
        miou = compute_Binary_miou(output.round(), edges, miou_metric)
        train_acc+=accuracy
        train_miou+=miou
        train_recall+=recall

    # Validation Phase
    for imgs, edges in val_loader:
        model.eval()
        with torch.no_grad():
            imgs = imgs.to(device)
            edges = edges.to(device)
            loss, output = compute_BCEloss(model, imgs, edges, loss_fn)
            val_loss += loss.item()
            output = sigmoid(output)
            accuracy = (output.round() == edges).float().mean().to('cpu')
            recall = compute_recall(output, edges)
            miou = compute_Binary_miou(output.round(),edges, miou_metric)
            val_acc+=accuracy
            val_miou+=miou
            val_recall+=recall

        if (epoch+1) % 5 == 0 and (epoch+1)>=50:        #save checkpoint for every 5 epochs after 50th epoch
            print('Saving model...', flush=True)
            checkpoint_name = f'checkpoint_epoch_{epoch+1}.pt'
            checkpoint_filepath = checkpoint_path+checkpoint_name
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss, 'val_loss': val_loss,
                        'train_acc': train_acc, 'val_acc': val_acc,
                        'train_miou': train_miou, 'val_miou': val_miou,
                        'train_recall': train_recall, 'val_recall': val_recall
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
                            'train_loss': train_loss, 'val_loss': val_loss,
                            'train_acc': train_acc, 'val_acc': val_acc,
                            'train_miou': train_miou, 'val_miou': val_miou,
                            'train_recall': train_recall, 'val_recall': val_recall
                            }, checkpoint_filepath)    
                            
        else:  
            no_improve += 1
            min_loss = val_loss/len(val_loader)
            print(f'Loss did not decrease for {no_improve} epoch(s).', flush=True)
            if no_improve == patience:
                print(f'Early stopping at epoch {epoch+1}.', flush=True)
                break

    # Accumulate Metrics
    train_losses_list.append(train_loss/len(train_loader))
    train_accs_list.append(train_acc*100/len(train_loader))
    train_mious_list.append(train_miou*100/len(train_loader))
    train_recall_list.append(train_recall*100/len(train_loader))
    val_losses_list.append(val_loss/len(val_loader))
    val_accs_list.append(val_acc*100/len(val_loader))
    val_mious_list.append(val_miou*100/len(val_loader))
    val_recall_list.append(val_recall*100/len(val_loader))

    #Print train, validation stats
    print(f'Epoch {epoch + 1}/{num_epochs} --> Train Stats - Loss : {train_loss/len(train_loader)} Accuracy : {train_acc*100/len(train_loader)} Recall : {train_recall*100/len(train_loader)} mIoU : {train_miou/len(train_loader)}', flush=True)
    print(f'Epoch {epoch + 1}/{num_epochs} --> Val Stats - Loss : {val_loss/len(val_loader)} Accuracy : {val_acc*100/len(val_loader)} Recall : {val_recall*100/len(val_loader)} mIoU : {val_miou/len(val_loader)}', flush=True)
    print(f'Time: {(time.time()-since)/60}', flush=True)
    print('\n', flush=True)

    print(f'Total time taken: {(time.time()-train_time)/60}')
    print('Training ended!')
    torch.save(model, folder_path+'edge_model.pth') #final checkpoint

    # Train and val loss plot, acc plot, miou plot, recall plot
    plot_curves_edge(num_epochs, train_losses_list, val_losses_list, train_accs_list, val_accs_list, train_mious_list, val_mious_list, train_recall_list, val_recall_list, filename = folder_path+'edge_metrics_final_plot.png')

#*********************************************************************************************************************************************************************************

    # Model evaluation on test set
    #Setup test dataloader
    test_data = CamVid('path_to_test_imgs',
                        img_transform=image_transform)

    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle = True)

    evaluate_edge(model, test_loader)

    # Visualization of prediction on an unseen image
    visualize_edge(model, test_loader, filename=folder_path+'edge_final_test_preds.png')


    # Run test for specific checkpoint
    model_checkpoint_path = f'path_to_ckpt'
    if os.path.isfile(model_checkpoint_path):
        model = UNET_edge()
        checkpoint = torch.load(model_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        print(f"Test Stats for model {model_checkpoint_path}", flush = True)
        evaluate_edge(model, test_loader)

        #V isualize predictions for that checkpoint
        for imgs,edges in test_loader:
            fig, axes = plt.subplots(8, 2, figsize=(12, 40))
            for i, (image, edge) in enumerate(zip(imgs, edges)):
                image = image.permute(1,2,0).cpu().numpy()
                edge = edge.squeeze(0).cpu().numpy()
                ax = axes[i]
                ax[0].imshow(image)
                ax[1].imshow(edge, cmap='gray')

                ax[0].axis('off')
                ax[1].axis('off')

            axes[0][0].set_title('Images', fontsize=16)
            axes[0][1].set_title('Edges', fontsize=16)

            plt.tight_layout()
            plt.savefig("sobeledge_prediction.png")
            plt.close()
            break

#*********************************************************************************************************************************************************************************
