
  import os
  from tqdm import tqdm
  from monai.inferers import sliding_window_inference
  from monai.metrics import DiceMetric
  from monai.networks.nets import SwinUNETR
  import torch



  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #for 2D single channel input with 
  #size (96,96), 2-channel output and gradient checkpointing.
  model = SwinUNETR(img_size=(96, 96),in_channels=3, out_channels=2, 
                    use_checkpoint=True, spatial_dims=2).to(device)


  def validation(epoch_iterator_val):
      model.eval()
      with torch.no_grad():
          for step, batch in enumerate(epoch_iterator_val):
              val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
              with torch.cuda.amp.autocast():
                  val_outputs = sliding_window_inference(val_inputs, (96, 96), 4, model)
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



   def train(global_step, train_loader, dice_val_best, global_step_best):
     model.train()
     epoch_loss = 0
     step = 0
     epoch_iterator = tqdm(
          train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
     for step, batch in enumerate(epoch_iterator):
         step += 1
         x, y = (batch["image"].to(device), batch["label"].to(device))
         with torch.cuda.amp.autocast():
             logit_map = model(x)
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
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step

                #save best metric model
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
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
         global_step += 1
      return global_step, dice_val_best, global_step_best
