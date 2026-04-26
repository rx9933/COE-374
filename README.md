Data Processing
1. get sample throw and put in VideoCameraProcessing/throws/
2. run VideoCameraProcessing/image_video_processing_scripts/raw_to_png.py
3. run VideoCameraProcessing/image_video_processing_scripts/zsh_script_ignore.py
4. Go through video and get interval that throw occurs
5. run VideoCameraProcessing/image_video_processing_scripts/png_folder_to_mp4.py

Intrinsics
1. Download data and put into Video_Camera_Processing/intrinsics_calibration_camera_0 and Video_Camera_Processing/intrinsics_calibration_camera_1
2. Run Video_Camera_Processing/CameraIntrinsics.py

Extrinsics
1. Get single photo of extrinsics for each camera and put in VideoCameraProcessing/extrinsics_calibration/
2. Run  VideoCameraProcessing/CameraExtrinsics.py

Running the full pipeline
1. Change parameters in video_to_position.py to the correct video directories and run code
