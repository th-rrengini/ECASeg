import matplotlib.pyplot as plt
import torch
from metrics import compute_loss, scores, compute_recall, compute_Binary_miou, compute_BCEloss
from torch.nn.functional import softmax, sigmoid
import time
import numpy as np 
# from skimage.metrics import structural_similarity as ssim
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# function to colourize the bitmask
def colorize_bitmask(bitmask, class_colors):
   
    height, width = bitmask.shape
    colorized_image = np.zeros((height, width, 3)) # create a 0 array that is of the shape of the bitmask

    for key, color in class_colors.items():
        mask = (bitmask == key)
        colorized_image[mask]= color # assign the color based on the key

    return colorized_image # return colorized image


### Plot model loss and metrics 
def plot_curves_seg(num_epochs, train_losses, val_losses, train_accs, val_accs, train_mious, val_mious, filename):
    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(18,5))

    #plot losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.xticks(range(0, num_epochs+1, 5)) 
    plt.xticks(rotation=45)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Cross-Entropy Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'b', label='Training accuracy')
    plt.plot(epochs, val_accs, 'r', label='Validation accuracy')
    plt.xticks(range(0, num_epochs+1, 5)) 
    plt.xticks(rotation=45)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot mious
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_mious, 'b', label='Training miou')
    plt.plot(epochs, val_mious, 'r', label='Validation miou')
    plt.xticks(range(0, num_epochs+1, 5)) 
    plt.xticks(rotation=45)
    plt.title('Training and Validation mIoUs')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

### Evaluate segmentation model on unseen test set
def evaluate_seg(model, dataloader):
    eval_time = time.time()
    test_labels = []
    test_preds = []
    test_loss = 0
    model.eval()  # Set model to evaluation mode
    for imgs,masks in dataloader:
        with torch.no_grad():
            imgs = imgs.to(device)
            masks = masks.to(device)
            loss, output = compute_loss(model, imgs, masks, loss_fn = torch.nn.CrossEntropyLoss().to(device))
        test_loss += loss.item()
        output = softmax(output,dim=1)
        predicted_classes = torch.argmax(output, dim=1)
        actual_classes = torch.argmax(masks.to('cpu'), dim=1)
        test_labels.extend(actual_classes.cpu().numpy())
        test_preds.extend(predicted_classes.cpu().numpy())
        test_scores = scores(test_labels, test_preds, n_class=12, ignore_class=11)

    print("*" * 100 + "\n")
    print(f'Test Statistics - Loss : {test_loss/len(dataloader)} Scores : {test_scores}', flush=True)
    print(f'Time taken for evaluation: {(time.time()-eval_time)/60}', flush=True)
    print('\n', flush=True)


### Visualize segmentation model predictions on 5 random samples from the unseen test set
def visualize_predictions_camvid(model, dataloader, filename):
    for imgs, masks in dataloader:
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
        plt.savefig(filename)
        plt.close()
        break

### Plot model performance loss and metrics 
def plot_curves_edge(num_epochs, train_losses, val_losses, train_accs, val_accs, train_mious, val_mious, train_recall, val_recall, filename):
    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(14,14))

    #plot losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.xticks(range(0, num_epochs+1, 5)) 
    plt.xticks(rotation=45)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r', label='Validation Accuracy')
    plt.xticks(range(0, num_epochs+1, 5)) 
    plt.xticks(rotation=45)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot mious
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_mious, 'b', label='Training mIoU')
    plt.plot(epochs, val_mious, 'r', label='Validation mIoU')
    plt.xticks(range(0, num_epochs+1, 5)) 
    plt.xticks(rotation=45)
    plt.title('Training and Validation mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()

    # Plot recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_recall, 'b', label='Training Recall')
    plt.plot(epochs, val_recall, 'r', label='Validation Recall')
    plt.xticks(range(0, num_epochs+1, 5)) 
    plt.xticks(rotation=45)
    plt.title('Training and Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



### Visualize edge model predictions on 5 random samples from unseen test set 
def visualize_edge(model, dataloader, filename):
    model.eval()
    for imgs, masks in dataloader:
        with torch.no_grad():
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            preds = sigmoid(preds)
            preds = preds.round().cpu()

            imgs = imgs[:5,:,:,:]
            masks = masks[:5,:,:]
            preds = preds[:5,:,:]

        fig, axes = plt.subplots(5, 3, figsize=(25, 20))
        for i, (image, gt_mask, pred) in enumerate(zip(imgs, masks, preds)):
            image = image.permute(1,2,0).cpu().numpy()
            gt_mask = gt_mask.squeeze(0).cpu().numpy()
            pred = (pred.squeeze(0).numpy())
            ax = axes[i]
            ax[0].imshow(image)
            ax[1].imshow(gt_mask, cmap='gray')
            ax[2].imshow(pred, cmap='gray')
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')

        axes[0][0].set_title('Images', fontsize=16)
        axes[0][1].set_title('Ground Truth Edges', fontsize=16)
        axes[0][2].set_title('Predictions', fontsize=16)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        break


## Evaluate edge model on test set
def evaluate_edge(model, dataloader):
    eval_time = time.time()
    test_loss = 0
    test_acc = 0
    test_miou = 0
    test_recall = 0
    model.eval()  # Set model to evaluation mode
    for imgs,masks in dataloader:
        with torch.no_grad():
            imgs = imgs.to(device)
            masks = masks.to(device)
            loss, output = compute_BCEloss(model, imgs, masks, loss_fn = torch.nn.BCEWithLogitsLoss().to(device))
        test_loss += loss.item()
        output = sigmoid(output)
        accuracy = (output.round() == masks).float().mean()
        recall = compute_recall(output, masks)
        miou = compute_Binary_miou(output, masks)
        test_acc+=accuracy
        test_miou+=miou
        test_recall+=recall
    print("*" * 100 + "\n")
    print(f'Test Statistics - Loss : {test_loss/len(dataloader)} Accuracy : {test_acc*100/len(dataloader)} mIoU : {test_miou/len(dataloader)} Recall : {test_recall/len(dataloader)}', flush=True)
    print(f'Time taken for evaluation: {(time.time()-eval_time)/60}', flush=True)
    print('\n', flush=True)



def save_scores(scores, filepath):
        with open(filepath, 'a') as f:
            json.dump(scores, f)
            f.write('\n') 
