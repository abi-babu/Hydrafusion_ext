import torch
import torch.nn as nn
import torch.nn.functional as F
from .branch import RadarBranch, CameraBranch, LidarBranch, DualCameraFusionBranch, CameraLidarFusionBranch, RadarLidarFusionBranch, ResNetTail
from .stem import RadarStem, CameraStem, LidarStem
from .fusion import FusionBlock
from torchvision.models.resnet import BasicBlock
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from typing import List



'''This file defines our HydraNet-based sensor fusion architecture.'''
class HydraFusion(nn.Module):

    def __init__(self, config):
        super(HydraFusion, self).__init__()
        self.config = config
        self.dropout = config.dropout
        self.activation = F.relu if config.activation == 'relu' else F.leaky_relu
        self.initialize_transforms()
        self.initialize_stems()
        self.initialize_branches()
        self.fusion_block = FusionBlock(config, fusion_type=1, weights=self.num_branches*[1], iou_thr=0.4, skip_box_thr=0.01, sigma=0.5, alpha=1)
  

    '''initializes the normalization/resizing transforms applied to input images.'''
    def initialize_transforms(self):
        if self.config.use_custom_transforms:
            self.image_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[88.12744903564453,90.560546875,90.5104751586914], image_std=[66.74466705322266,74.3885726928711,75.6873779296875])
            self.radar_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[15.557413101196289,15.557413101196289,15.557413101196289], image_std=[18.468725204467773,18.468725204467773,18.468725204467773])
            self.lidar_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[2.1713976860046387,2.1713976860046387,2.1713976860046387], image_std=[20.980266571044922,20.980266571044922,20.980266571044922])
            self.fwd_lidar_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[0.0005842918762937188,0.0005842918762937188,0.0005842918762937188], image_std=[0.10359727591276169,0.10359727591276169,0.10359727591276169])
        else:
            self.transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]) #from ImageNet
            self.image_transform = self.transform
            self.radar_transform = self.transform
            self.lidar_transform = self.transform
            self.fwd_lidar_transform = self.transform


    '''initializes the stem modules as the first blocks of resnet-18.'''
    def initialize_stems(self):
        if self.config.enable_radar:
            self.radar_stem = RadarStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)  # TODO:define these config values in config.py
        if self.config.enable_camera:  
            self.camera_stem = CameraStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)
        if self.config.enable_lidar:
            self.lidar_stem = LidarStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)
        if self.config.enable_cam_lidar_fusion:
            self.fwd_lidar_stem = LidarStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)


    '''initializes the branch modules as the remaining blocks of resnet-18 and the RPN.'''
    def initialize_branches(self):
        self.num_branches = 0
        if self.config.enable_radar:
            self.radar_branch = RadarBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.radar_transform).to(self.config.device)
            self.num_branches += 1
        if self.config.enable_camera:    
            self.l_cam_branch = CameraBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.image_transform).to(self.config.device)
            self.r_cam_branch = CameraBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.image_transform).to(self.config.device)
            self.num_branches += 2
        if self.config.enable_lidar:
            self.lidar_branch = LidarBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.lidar_transform).to(self.config.device)
            self.num_branches += 1
        if self.config.enable_cam_fusion:
            self.cam_fusion_branch = DualCameraFusionBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.image_transform).to(self.config.device)
            self.num_branches += 1
        if self.config.enable_cam_lidar_fusion:
            self.lidar_cam_fusion_branch = CameraLidarFusionBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.image_transform).to(self.config.device)
            self.num_branches += 1
        if self.config.enable_radar_lidar_fusion:
            self.radar_lidar_fusion_branch = RadarLidarFusionBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.radar_transform).to(self.config.device)
            self.num_branches += 1


    '''
    <sensor>_x is in the input image/sensor data from each modality for a single frame. 
    radar_y, cam_y contains the target bounding boxes for training BEV and FWD respectively.
    Currently. all enabled branches are executed for every input.
    '''
    def forward(self, leftcamera_x=None, rightcamera_x=None, radar_x=None, bev_lidar_x=None,
                l_lidar_x=None, r_lidar_x=None, radar_y=None, cam_y=None):
    # NOTE: For modality-split training we intentionally *do not* raise here so
    # a camera-only client can call forward(...) with radar_x=None etc.
        branch_selection = []
        output_losses, output_detections = {}, {}

    # if targets are None, fix_targets will return None
        radar_y = self.fix_targets(radar_y)
        cam_y = self.fix_targets(cam_y)

    # initialize branch outputs to None so later checks don't crash
        radar_output = None
        l_camera_output = None
        r_camera_output = None
        bev_lidar_output = None
        r_lidar_output = None

    # RADAR branch (only when radar_x AND radar_y present or when inference and radar_x present)
        if self.config.enable_radar and radar_x is not None:
        # Only call transform if we have data to transform
            if radar_y is not None:
                radar_x, radar_y = self.radar_transform(radar_x, radar_y)
                self.check_for_degenerate_bboxes(radar_y)
            else:
            # try transform with radar_x alone if transform supports it; otherwise skip
                try:
                    radar_x, radar_y = self.radar_transform(radar_x, radar_y)
                except Exception:
                    radar_y = None
        # only proceed if radar_x was transformed into a valid object with .tensors
            if radar_x is not None:
                branch_selection.append(0)
                radar_output = F.dropout(self.radar_stem(radar_x.tensors), self.dropout, training=self.training)

    # CAMERA branch (only when camera input present)
        if self.config.enable_camera and (rightcamera_x is not None or leftcamera_x is not None):
        # If a cam target exists, prefer transforming with cam_y to get proper target format
            if rightcamera_x is not None:
                rightcamera_x, cam_y = self.image_transform(rightcamera_x, cam_y)
            if leftcamera_x is not None:
                leftcamera_x, _ = self.image_transform(leftcamera_x)
        # check for valid targets before using branches (only if cam_y exists)
            if cam_y is not None:
                self.check_for_degenerate_bboxes(cam_y)
        # make sure transformed inputs exist
            if leftcamera_x is not None:
                branch_selection.append(1)
                l_camera_output = F.dropout(self.camera_stem(leftcamera_x.tensors), self.dropout, training=self.training)
            if rightcamera_x is not None:
                branch_selection.append(2)
                r_camera_output = F.dropout(self.camera_stem(rightcamera_x.tensors), self.dropout, training=self.training)

    # LIDAR branch (only when lidar input present)
        if self.config.enable_lidar and bev_lidar_x is not None:
            try:
                bev_lidar_x, _ = self.lidar_transform(bev_lidar_x)
            except Exception:
                pass
            if bev_lidar_x is not None:
                branch_selection.append(3)
                bev_lidar_output = F.dropout(self.lidar_stem(bev_lidar_x.tensors), self.dropout, training=self.training)

    # CAM+LIDAR fusion variant (only if r_lidar_x exists or camera not yet run but inputs exist)
        if self.config.enable_cam_lidar_fusion:
         # ensure camera stems are present (if not already run)
            if 1 not in branch_selection and (rightcamera_x is not None or leftcamera_x is not None):
                if rightcamera_x is not None:
                    rightcamera_x, cam_y = self.image_transform(rightcamera_x, cam_y)
                if leftcamera_x is not None:
                    leftcamera_x, _ = self.image_transform(leftcamera_x)
                if leftcamera_x is not None:
                    l_camera_output = F.dropout(self.camera_stem(leftcamera_x.tensors), self.dropout, training=self.training)
                if rightcamera_x is not None:
                    r_camera_output = F.dropout(self.camera_stem(rightcamera_x.tensors), self.dropout, training=self.training)

            if r_lidar_x is not None:
                try:
                    r_lidar_x, _ = self.fwd_lidar_transform(r_lidar_x)
                except Exception:
                    pass
                if r_lidar_x is not None:
                    branch_selection.append(5)
                    r_lidar_output = F.dropout(self.fwd_lidar_stem(r_lidar_x.tensors), self.dropout, training=self.training)

    # CAM fusion-only gate (if enabled, add fused camera branch output if inputs present)
        if self.config.enable_cam_fusion:
            if 1 not in branch_selection and (rightcamera_x is not None or leftcamera_x is not None):
                if rightcamera_x is not None:
                    rightcamera_x, cam_y = self.image_transform(rightcamera_x, cam_y)
                if leftcamera_x is not None:
                    leftcamera_x, _ = self.image_transform(leftcamera_x)
                if leftcamera_x is not None and rightcamera_x is not None:
                    l_camera_output = F.dropout(self.camera_stem(leftcamera_x.tensors), self.dropout, training=self.training)
                    r_camera_output = F.dropout(self.camera_stem(rightcamera_x.tensors), self.dropout, training=self.training)
                    branch_selection.append(4)

    # RADAR + LIDAR fusion gate
        if self.config.enable_radar_lidar_fusion:
         # run radar if not already
            if 0 not in branch_selection and radar_x is not None:
                if radar_y is not None:
                    radar_x, radar_y = self.radar_transform(radar_x, radar_y)
                    self.check_for_degenerate_bboxes(radar_y)
                else:
                    try:
                        radar_x, radar_y = self.radar_transform(radar_x, radar_y)
                    except Exception:
                        radar_y = None
                if radar_x is not None:
                    radar_output = F.dropout(self.radar_stem(radar_x.tensors), self.dropout, training=self.training)
                    branch_selection.append(0)
        # run lidar if not already
            if 3 not in branch_selection and bev_lidar_x is not None:
                try:
                    bev_lidar_x, _ = self.lidar_transform(bev_lidar_x)
                except Exception:
                    pass
                if bev_lidar_x is not None:
                    bev_lidar_output = F.dropout(self.lidar_stem(bev_lidar_x.tensors), self.dropout, training=self.training)
                    branch_selection.append(3)
            if 0 in branch_selection or 3 in branch_selection:
            # only add fusion branch index if at least one of radar/lidar ran
                branch_selection.append(6)

    # collect outputs from activated branches
        for branch_index in branch_selection:
            if branch_index == 0 and radar_output is not None:
                output_losses['radar'], output_detections['radar'] = self.radar_branch(radar_output, radar_x, radar_y)
            elif branch_index == 1 and l_camera_output is not None:
                output_losses['camera_left'], output_detections['camera_left'] = self.l_cam_branch(l_camera_output, leftcamera_x, cam_y)
            elif branch_index == 2 and r_camera_output is not None:
                output_losses['camera_right'], output_detections['camera_right'] = self.r_cam_branch(r_camera_output, rightcamera_x, cam_y)
            elif branch_index == 3 and bev_lidar_output is not None:
                output_losses['lidar'], output_detections['lidar'] = self.lidar_branch(bev_lidar_output, bev_lidar_x, radar_y)
            elif branch_index == 4 and l_camera_output is not None and r_camera_output is not None:
                output_losses['camera_both'], output_detections['camera_both'] = self.cam_fusion_branch(l_camera_output, r_camera_output, rightcamera_x, cam_y)
            elif branch_index == 5 and l_camera_output is not None and r_lidar_output is not None:
                output_losses['camera_lidar'], output_detections['camera_lidar'] = self.lidar_cam_fusion_branch(l_camera_output, r_camera_output, r_lidar_output, rightcamera_x, cam_y)
            elif branch_index == 6 and radar_output is not None and bev_lidar_output is not None:
                output_losses['radar_lidar'], output_detections['radar_lidar'] = self.radar_lidar_fusion_branch(radar_output, bev_lidar_output, radar_x, radar_y)

    # If inference and create_gate_dataset set -> return stems for gating dataset building
        if (not self.training) and self.config.create_gate_dataset:
        # return whichever stem outputs are available
            stems = {}
            if radar_output is not None: stems['radar'] = radar_output.cpu()
            if l_camera_output is not None: stems['camera_left'] = l_camera_output.cpu()
            if r_camera_output is not None: stems['camera_right'] = r_camera_output.cpu()
            if bev_lidar_output is not None: stems['lidar'] = bev_lidar_output.cpu()
            if 'r_lidar_output' in locals() and r_lidar_output is not None:
                stems['r_lidar'] = r_lidar_output.cpu()
            return output_losses, output_detections, stems

    # If not training -> run fusion block if we have any detection outputs
        if (not self.training):
            if output_losses or output_detections:
                final_loss, final_detections = self.fusion_block(output_losses, output_detections, self.config.fusion_sweep)
                return final_loss, final_detections
            else:
                # nothing to fuse: return empty structures
                return {}, {}

    # During training, return per-branch outputs (as original)
        return output_losses, output_detections


    def fix_targets(self, targets):
        """ Convert targets to proper format. Be tolerant to None. """
        if targets is None:
            return None
    # sometimes targets is already processed or empty list
        if not isinstance(targets, (list, tuple)):
            return targets
        for t in targets:
         # guard in case a malformed target slipped in
            if 'labels' in t and t['labels'] is not None:
                t['labels'] = t['labels'].long().squeeze(0).to(self.config.device)
            if 'boxes' in t and t['boxes'] is not None:
                t['boxes'] = t['boxes'].squeeze(0).to(self.config.device)
        return targets


    def check_for_degenerate_bboxes(self, targets):
        """ Raise if any invalid bboxes are found. Tolerant to None. """
        if targets is None:
            return
        for target_idx, target in enumerate(targets):
            boxes = target.get("boxes", None)
            if boxes is None:
                continue
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

