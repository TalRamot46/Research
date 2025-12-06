# split videos into frames for the pendulum dataset creation
def split_video_to_frames():
    import cv2
    import os

    video_path = "pendulum_yolo/pendulum_raw_data/pendulum_video.mp4"
    output_dir = "pendulum_yolo/pendulum_raw_data/clean_images"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    save_every_n_frames = 6  # e.g., save every 6th frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % save_every_n_frames == 0:
            out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
        frame_idx += 1

    cap.release()
    print("Done.")

# in clean_images folder, there are files with ending of .xml.txt,
# I need to rename them to .txt only
def rename_label_files():
    import os
    import glob

    label_dir = "pendulum_yolo/pendulum_raw_data/clean_images"
    xml_txt_files = glob.glob(os.path.join(label_dir, "*.xml.txt"))

    for xml_txt_file in xml_txt_files:
        base = os.path.splitext(os.path.splitext(xml_txt_file)[0])[0]  # remove .xml.txt
        new_txt_file = base + ".txt"
        os.rename(xml_txt_file, new_txt_file)

    print("Renaming done.")

# create folder structure and split dataset into train and val sets
def create_folders_and_split_dataset():
    import os, shutil, random, glob

    base = "pendulum_yolo/pendulum_dataset"
    img_dir = os.path.join(base, "images")
    lbl_dir = os.path.join(base, "labels")
    os.makedirs(os.path.join(img_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(img_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(lbl_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(lbl_dir, "val"), exist_ok=True)

    all_images = glob.glob("pendulum_yolo/pendulum_raw_data/clean_images/*.jpg")
    random.shuffle(all_images)

    split_idx = int(0.8 * len(all_images))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    def move_pairs(image_list, split_name):
        for img_path in image_list:
            name = os.path.basename(img_path)
            txt_path = img_path.replace(".jpg", ".txt")
            shutil.copy(img_path, os.path.join(img_dir, split_name, name))
            shutil.copy(txt_path, os.path.join(lbl_dir, split_name, name.replace(".jpg", ".txt")))

    move_pairs(train_images, "train")
    move_pairs(val_images, "val")
    print("Done.")

def main():
    # split_video_to_frames()
    #rename_label_files()
    create_folders_and_split_dataset()

if __name__ == "__main__":
    main()
