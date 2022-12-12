import os
import sys

from tqdm import tqdm
import torch
import numpy as np

from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)
from monai.data import decollate_batch


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def validation(epoch_iterator_val,model,global_step):
    model.eval()
    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (torch.from_numpy(batch["image"]).cuda(), torch.from_numpy(batch["label"]).cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs.type(torch.float32), (96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, val_loader, model, dice_val_best=0, global_step_best=0,
            device=DEFAULT_DEVICE, output_dir='/scratch/ejg8qa/RESULTS', eval_num=1,
            max_iterations=5000):
    
    model.train()
    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    
    epoch_loss = 0
    step = 0
    epoch_loss_values = []
    metric_values = []
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1

        x, y = (torch.from_numpy(batch["image"]).to(device), torch.from_numpy(batch["label"]).to(device))
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            #import pdb;pdb.set_trace()
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        # track loss metric
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" 
                                    % (global_step, max_iterations, loss))

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
            val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            # track val metric
            dice_val = validation(epoch_iterator_val,model,global_step)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step

                #save best metric model
                torch.save(
                    model.state_dict(), os.path.join(output_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val)
                )
            model.train()
        global_step += 1
    return global_step, dice_val_best, global_step_best
