import os
import pickle
import torch
import numpy as np
from PIL import Image
from zod import ZodSequences,EgoRoadAnnotation
from zod.constants import Camera, Lidar, AnnotationProject, Radar
import glob
from zod.utils.polygon_transformations import polygons_to_binary_mask
# Dataset setup
dataset_root = os.path.expanduser("~/zod_mini")
version = "mini"
zod_sequences = ZodSequences(dataset_root=dataset_root, version=version)

# Retrieve all sequence IDs and filter only those from "000000"
sequence_ids = [seq_id for seq_id in zod_sequences.get_all_ids() if "000000" in seq_id] #just change the seq id for diff sequences

def sanitize_path(filepath):
    drive, path = os.path.splitdrive(filepath)
    cleaned_path = path.replace(":", "_")
    return os.path.join(drive, cleaned_path)

hydrafusion_inputs = []
processed_timestamps = set()
stored_timestamps = set()

for seq_id in sequence_ids:
    try:
        print(f"\nProcessing Sequence ID: {seq_id}")
        seq = zod_sequences[seq_id]
        frames = seq.info.get_camera_lidar_map()

        # Load the one radar file
        radar_pc = None
        radar_dir = os.path.join(dataset_root, "sequences", seq_id, "radar_front")
        radar_files = glob.glob(os.path.join(radar_dir, "*.npy"))
        if radar_files:
            radar_path = sanitize_path(radar_files[0])
            try:
                radar_data = np.load(radar_path, allow_pickle=True)
                radar_pc = np.array(radar_data.tolist())
                print(f"Loaded radar data from: {radar_path}, shape: {radar_pc.shape}")
            except Exception as e:
                print(f"Error loading radar data: {e}")
        else:
            print(f"No radar file found in {radar_dir}")

        for camera_frame, lidar_frame in frames:
            frame_timestamp = camera_frame.time
            if frame_timestamp in stored_timestamps or frame_timestamp in processed_timestamps:
                print(f"Skipping duplicate frame at time {frame_timestamp}")
                continue

            stored_timestamps.add(frame_timestamp)
            processed_timestamps.add(frame_timestamp)

            try:
                input_dict = {}

                # IMAGE
                image_path = sanitize_path(camera_frame.filepath)
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                with Image.open(image_path) as img:
                    original_width, original_height = img.size
                    img = img.convert("RGB").resize((1024, 512))
                    image = np.array(img, dtype=np.uint8)

                scale_x = 1024 / original_width
                scale_y = 512 / original_height
                input_dict["file_name"] = image_path
                metadata = seq.metadata
                print(metadata)
                annotations = seq.get_annotation(AnnotationProject.OBJECT_DETECTION) #Annotation file has different classes
                vehicle_annotations = [anno for anno in annotations]
                bbox_2d_list = []
                bbox_3d_list = []
                radar_y_list = []
                try:
                    ego_motion = seq.ego_motion
                    frame_time_unix = frame_timestamp.timestamp()
                    ego_times = seq.ego_motion.timestamps
                    idx = np.argmin(np.abs(ego_times - frame_time_unix))
                    position = ego_motion.poses[idx, :3, 3]
                    input_dict["position_xyz"] = torch.tensor(position, dtype=torch.float32)
                    velocity = ego_motion.velocities[idx]
                    speed = np.linalg.norm(velocity)
                    input_dict["velocity_vector"] = torch.tensor(velocity, dtype=torch.float32)
                    input_dict["speed"] = torch.tensor(speed, dtype=torch.float32)
                    acceleration = ego_motion.accelerations[idx]
                    input_dict["acceleration_vector"] = torch.tensor(acceleration, dtype=torch.float32)
                    rotation_matrix = ego_motion.poses[idx, :3, :3]
                    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    input_dict["yaw_angle"] = torch.tensor(yaw, dtype=torch.float32)
                except Exception as motion_error:
                    print(f"Error extracting ego-motion data: {motion_error}")
                for anno in vehicle_annotations:
                    box2d = anno.box2d
                    box3d = anno.box3d
                    if box2d is None or box3d is None:
                        continue

                    # 2D
                    x1, y1, x2, y2 = box2d.xyxy
                    bbox_2d_list.append(torch.tensor([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=torch.float32))
                    bbox_3d_list.append(torch.tensor(box3d.center, dtype=torch.float32))

                    # Radar_y (BEV 3D ground truth format): [x, y, w, l, yaw, class_id]
                    x, y, _ = box3d.center
                    w, l, _ = box3d.size
                    yaw = box3d.orientation  # assume float angle in radians
                    class_id = 1  # 'vehicle' class ID
                    radar_y_list.append([x, y, w, l, yaw, class_id])

                input_dict["camera"] = torch.tensor(image, dtype=torch.float32)
                input_dict["bbox_2d"] = torch.stack(bbox_2d_list) if bbox_2d_list else None
                input_dict["bbox_3d"] = torch.stack(bbox_3d_list) if bbox_3d_list else None
                input_dict["labels"] = torch.ones(len(bbox_2d_list), dtype=torch.int64) if bbox_2d_list else None

                # LIDAR
                lidar_path = sanitize_path(lidar_frame.filepath)
                if not os.path.exists(lidar_path):
                    print(f"LiDAR file not found: {lidar_path}")
                    continue

                pc = np.load(lidar_path, allow_pickle=True)
                pc = np.array(pc.tolist())
                input_dict["lidar_xyz"] = torch.tensor(pc[:, :3], dtype=torch.float32)
                input_dict["lidar_intensity"] = torch.tensor(pc[:, 4], dtype=torch.float32)

                # RADAR
                if radar_pc is not None:
                    input_dict["radar_xyz"] = torch.tensor(radar_pc[:, :3], dtype=torch.float32)
                    if radar_y_list:
                        input_dict["radar_y"] = torch.tensor(radar_y_list, dtype=torch.float32)
                    try:
                        radar_calib = seq.calibration.get_extrinsics(Radar.FRONT)
                        input_dict["radar_extrinsics"] = torch.tensor(radar_calib.transform, dtype=torch.float32)
                    except Exception as radar_calib_error:
                        print(f"Radar calibration missing: {radar_calib_error}")

                # CALIBRATION
                calib = seq.calibration
                cam_calib = calib.cameras[Camera.FRONT]
                lidar_calib = calib.lidars[Lidar.VELODYNE]

                input_dict.update({
                    "camera_intrinsics": torch.tensor(cam_calib.intrinsics, dtype=torch.float32),
                    "camera_extrinsics": torch.tensor(cam_calib.extrinsics.transform, dtype=torch.float32),
                    "lidar_extrinsics": torch.tensor(lidar_calib.extrinsics.transform, dtype=torch.float32),
                })

                hydrafusion_inputs.append(input_dict)

            except Exception as frame_error:
                print(f"Error processing frame {camera_frame.time}: {frame_error}")
                continue

    except Exception as seq_error:
        print(f"General error for Sequence ID {seq_id}: {seq_error}")
        continue

# Save
output_file = "test_seq0.pkl" #Final pkl File
with open(output_file, "wb") as f:
    pickle.dump(hydrafusion_inputs, f)

print(f"\nSuccessfully processed {len(hydrafusion_inputs)} frames from '000002'. Data saved to '{output_file}'")
