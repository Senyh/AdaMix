import torch
import torch.nn as nn
import numpy as np
import unfoldNd


class AdaMix(nn.Module):
    def __init__(self, total_steps=0, num_classes=4, image_size=256, patch_size=8, topk=16, age=1, growing_factor=1.01, mode='hard', p=0.5, self_paced=True, device='cuda', inverse=False, age_increase='gaussian'):
        super(AdaMix, self).__init__()
        # Probability
        self.p = p
        # Device
        self.device = device
        # Patch Mix
        self.h, self.w = image_size // patch_size, image_size // patch_size
        self.s = self.h
        self.unfolds  = torch.nn.Unfold(kernel_size=(self.h, self.w), stride=self.s).to(device)
        self.folds = torch.nn.Fold(output_size=(image_size, image_size), kernel_size=(self.h, self.w), stride=self.s).to(device)
        # Self-Paced Learning
        self.age = age
        self.growing_factor = growing_factor
        self.mode = mode
        self.num_classes = num_classes
        self.topk = topk
        self.total_steps = total_steps
        self.self_paced = self_paced
        self.inverse = inverse
        self.age_incease = age_increase

    @torch.no_grad()
    def forward(self, oimage, aimage, olabel, alabel, oconf, aconf, prediction, cur_step):
        
        if torch.rand(1) < self.p:
            if self.self_paced:
                self.super_loss = self.dice_loss(prediction, olabel)
                splc = self.spl_curriculum(self.super_loss)
                sp_mask, sp_weight = splc['mask'], splc['weight']
            else:
                batch_size = oimage.shape[0]
                if self.inverse:
                    sp_mask = torch.ones(batch_size).bool().tolist()  # [(True) * batch_size]
                else:
                    sp_mask = torch.zeros(batch_size).bool().tolist()  # [(False) * batch_size]
                
                sp_weight = torch.ones(batch_size).tolist()

            oconf_map = oconf.clone().unsqueeze(1)
            aconf_map = aconf.clone().unsqueeze(1)
            B, C = oimage.shape[:2]
            oconf_unfolds = self.unfolds(oconf_map)  # B x C*kernel_size[0]*kernel_size[1] x L
            oconf_unfolds = oconf_unfolds.view(B, 1, self.h, self.w, -1)  # B x C x h x w x L
            oconf_unfolds_mean = torch.mean(oconf_unfolds, dim=(1, 2, 3))  # B x L

            aconf_unfolds = self.unfolds(aconf_map)  # B x C*kernel_size[0]*kernel_size[1] x L
            aconf_unfolds = aconf_unfolds.view(B, 1, self.h, self.w, -1)  # B x C x h x w x L
            aconf_unfolds_mean = torch.mean(aconf_unfolds, dim=(1, 2, 3))  # B x L

            oimage_unfolds = self.unfolds(oimage).view(B, C, self.h, self.w, -1)  # B x C x h x w x L
            olabel_unfolds = self.unfolds(olabel.unsqueeze(1).float()).view(B, 1, self.h, self.w, -1)  # B x C x h x w x L
            aimage_unfolds = self.unfolds(aimage).view(B, C, self.h, self.w, -1)  # B x C x h x w x L
            alabel_unfolds = self.unfolds(alabel.unsqueeze(1).float()).view(B, 1, self.h, self.w, -1) # B x C x h x w x L

            for i in range(B):
                topk = min(self.topk, abs(int(self.topk * sp_weight[i])))
                _, oconf_unfolds_max_index = torch.sort(oconf_unfolds_mean[i], dim=0, descending=sp_mask[i])  # B x L B x L
                _, aconf_unfolds_max_index = torch.sort(aconf_unfolds_mean[i], dim=0, descending=not sp_mask[i])  # B x L B x L
                oimage_unfolds[i, :, :, :, oconf_unfolds_max_index[:topk]] = aimage_unfolds[i, :, :, :, aconf_unfolds_max_index[:topk]]
                olabel_unfolds[i, :, :, :, oconf_unfolds_max_index[:topk]] = alabel_unfolds[i, :, :, :, aconf_unfolds_max_index[:topk]]
                oconf_unfolds[i, :, :, :, oconf_unfolds_max_index[:topk]] = aconf_unfolds[i, :, :, :, aconf_unfolds_max_index[:topk]]

            oimage = self.folds(oimage_unfolds.view(B, C*self.h*self.w, -1)).clone()
            olabel = self.folds(olabel_unfolds.view(B, 1*self.h*self.w, -1)).clone().squeeze(1).long()
            oconf = self.folds(oconf_unfolds.view(B, 1*self.h*self.w, -1)).clone().squeeze(1)
        self.increase_age(cur_step=cur_step, total_steps=self.total_steps, age_incease=self.age_incease)
        
        return oimage, olabel, oconf
    
    def dice_loss(self, prediction, target):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
        smooth = 1e-5
        prediction = torch.softmax(prediction, dim=1)
        batchsize = target.size(0)
        # Calculate the Dice Similarity Coefficient for each class
        intersection = torch.sum(prediction * target, dim=(2, 3))
        union = torch.sum(prediction + target, dim=(2, 3))
        dice = ((2 * intersection) / (union + smooth)).mean(1)
        dice_loss = 1. - dice
        return dice_loss
    
    def increase_age(self, cur_step, total_steps, age_incease):
        with torch.no_grad():
            if age_incease == 'gaussian':
                self.age = self.sigmoid_rampup(cur_step, total_steps)
            elif age_incease == 'linear':
                self.age = self.linear_rampup(cur_step, total_steps)
            elif age_incease == 'cosine':
                self.age = self.cosine_rampup(cur_step, total_steps)
            elif age_incease == 'step':
                self.age = self.step_rampup(cur_step, total_steps)
            else:
                self.age = 0.5

    def spl_curriculum(self, super_loss):
        m = super_loss < self.age
        v = m.clone().float()
        v = 1. - (super_loss / (self.age + 1e-5))
        return {'mask':m.tolist(), 'weight': v.tolist()}
    
    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        
    def linear_rampup(self, current, rampup_length):
        """Linear rampup"""
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        else:
            return current / rampup_length
        
    def cosine_rampdown(self, current, rampdown_length):
        """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
        # assert 0 <= current <= rampdown_length
        return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
    
    def cosine_rampup(self, current, rampdown_length):
        """Cosine rampup from https://arxiv.org/abs/1608.03983"""
        # assert 0 <= current <= rampdown_length
        return float(.5 * (np.cos(np.pi * (1 - current / rampdown_length)) + 1))
    
    def step_rampup(self, current, rampup_length, segment_size=1000, initial_value=0.0, target_value=1.0):
        """
        Segmented step ramp-up function to gradually increase a value from initial_value to target_value.
        
        Args:
            current_step (int): Current step.
            total_steps (int): Total number of steps to reach target_value.
            segment_size (int): Number of steps per segment.
            initial_value (float): Initial value (default is 0.0).
            target_value (float): Target value (default is 1.0).
        
        Returns:
            float: Value between initial_value and target_value.
        """
        if rampup_length == 0 or segment_size == 0:
            return target_value
        
        # Calculate the number of segments
        num_segments = rampup_length // segment_size
        if rampup_length % segment_size != 0:
            num_segments += 1
        
        # Calculate the increment per segment
        increment = (target_value - initial_value) / num_segments
        
        # Determine the current segment
        current_segment = current // segment_size
        
        # Calculate the current value
        value = initial_value + increment * current_segment
        
        # Ensure the value does not exceed the target_value
        return min(target_value, value)
            


class AdaMix3D(nn.Module):
    def __init__(self, total_steps=0, num_classes=4, image_size=[96, 96, 96], patch_size=8, topk=16, age=1, growing_factor=1.01, mode='hard', p=0.5, self_paced=True, device='cuda', inverse=False):
        super(AdaMix3D, self).__init__()
        # Probability
        self.p = p
        # Device
        self.device = device
        # Patch Mix
        self.d, self.h, self.w = image_size[0] // patch_size, image_size[1] // patch_size, image_size[2] // patch_size
        self.sd, self.sh, self.sw = image_size[0] // patch_size, image_size[1] // patch_size, image_size[2] // patch_size
        self.unfolds  = unfoldNd.UnfoldNd(kernel_size=(self.d, self.h, self.w), stride=(self.d, self.h, self.w)).to(device)
        self.folds = unfoldNd.FoldNd(output_size=(image_size[0], image_size[1], image_size[2]), kernel_size=(self.d, self.h, self.w), stride=(self.sd, self.sh, self.sw)).to(device)
        # Self-Paced Learning
        self.age = age
        self.growing_factor = growing_factor
        self.mode = mode
        self.num_classes = num_classes
        self.topk = topk
        self.total_steps = total_steps
        self.self_paced = self_paced
        self.inverse = inverse

    @torch.no_grad()
    def forward(self, oimage, aimage, olabel, alabel, oconf, aconf, prediction, cur_step):
        
        if torch.rand(1) < self.p:
            if self.self_paced:
                self.super_loss = self.dice_loss(prediction, olabel)
                splc = self.spl_curriculum(self.super_loss)
                sp_mask, sp_weight = splc['mask'], splc['weight']
            else:
                batch_size = oimage.shape[0]
                if self.inverse:
                    sp_mask = torch.ones(batch_size).bool().tolist()  # [(True) * batch_size]
                else:
                    sp_mask = torch.zeros(batch_size).bool().tolist()  # [(False) * batch_size]
                
                sp_weight = torch.ones(batch_size).tolist()

            oconf_map = oconf.clone().unsqueeze(1)
            aconf_map = aconf.clone().unsqueeze(1)
            B, C = oimage.shape[:2]
            oconf_unfolds = self.unfolds(oconf_map)  # B x C*kernel_size[0]*kernel_size[1] x L
            oconf_unfolds = oconf_unfolds.view(B, 1, self.d, self.h, self.w, -1)  # B x C x d x h x w x L
            oconf_unfolds_mean = torch.mean(oconf_unfolds, dim=(1, 2, 3, 4))  # B x L

            aconf_unfolds = self.unfolds(aconf_map)  # B x C*kernel_size[0]*kernel_size[1] x L
            aconf_unfolds = aconf_unfolds.view(B, 1, self.d, self.h, self.w, -1)  # B x C x h x w x L
            aconf_unfolds_mean = torch.mean(aconf_unfolds, dim=(1, 2, 3, 4))  # B x L

            oimage_unfolds = self.unfolds(oimage).view(B, C, self.d, self.h, self.w, -1)  # B x C x h x w x L
            olabel_unfolds = self.unfolds(olabel.unsqueeze(1).float()).view(B, 1, self.d, self.h, self.w, -1)  # B x C x h x w x L
            aimage_unfolds = self.unfolds(aimage).view(B, C, self.d, self.h, self.w, -1)  # B x C x h x w x L
            alabel_unfolds = self.unfolds(alabel.unsqueeze(1).float()).view(B, 1, self.d, self.h, self.w, -1) # B x C x h x w x L

            for i in range(B):
                topk = min(self.topk, abs(int(self.topk * sp_weight[i])))
                _, oconf_unfolds_max_index = torch.sort(oconf_unfolds_mean[i], dim=0, descending=sp_mask[i])  # B x L B x L
                _, aconf_unfolds_max_index = torch.sort(aconf_unfolds_mean[i], dim=0, descending=not sp_mask[i])  # B x L B x L
                oimage_unfolds[i, :, :, :, :, oconf_unfolds_max_index[:topk]] = aimage_unfolds[i, :, :, :, :, aconf_unfolds_max_index[:topk]]
                olabel_unfolds[i, :, :, :, :, oconf_unfolds_max_index[:topk]] = alabel_unfolds[i, :, :, :, :, aconf_unfolds_max_index[:topk]]
                oconf_unfolds[i, :, :, :, :, oconf_unfolds_max_index[:topk]] = aconf_unfolds[i, :, :, :, :, aconf_unfolds_max_index[:topk]]

            oimage = self.folds(oimage_unfolds.view(B, C*self.d*self.h*self.w, -1)).detach()
            olabel = self.folds(olabel_unfolds.view(B, 1*self.d*self.h*self.w, -1)).detach().squeeze(1).long()
            oconf = self.folds(oconf_unfolds.view(B, 1*self.d*self.h*self.w, -1)).detach().squeeze(1)
        self.increase_age(cur_step=cur_step, total_steps=self.total_steps)
        
        return oimage, olabel, oconf
    
    def dice_loss(self, prediction, target):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).contiguous()
        smooth = 1e-5
        prediction = torch.softmax(prediction, dim=1)
        batchsize = target.size(0)
        # Calculate the Dice Similarity Coefficient for each class
        intersection = torch.sum(prediction * target, dim=(2, 3, 4))
        union = torch.sum(prediction + target, dim=(2, 3, 4))
        dice = ((2 * intersection) / (union + smooth)).mean(1)
        dice_loss = 1. - dice
        return dice_loss

    def increase_age(self, cur_step, total_steps):
        with torch.no_grad():
            self.age = self.sigmoid_rampup(cur_step, total_steps)

    def spl_curriculum(self, super_loss):
        m = super_loss < self.age
        v = m.clone().float()
        v = 1. - (super_loss / (self.age + 1e-5))
        return {'mask':m.tolist(), 'weight': v.tolist()}
    
    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))