import os
import shutil

def organize_by_extension(directory):
    # Make sure directory exists
    if not os.path.isdir(directory):
        print("Invalid directory")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Get file extension
        _, ext = os.path.splitext(filename)

        if ext == "":
            folder_name = "no_extension"
        else:
            folder_name = ext[1:].lower()  # remove dot and lowercase

        target_folder = os.path.join(directory, folder_name)

        # Create folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        # Move file
        shutil.move(file_path, os.path.join(target_folder, filename))

    print("Files organized successfully")

if __name__ == "__main__":
    path = "Video_Camera_Processing/throws/third_throw1"
    organize_by_extension(path)