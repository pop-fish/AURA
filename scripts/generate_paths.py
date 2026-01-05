import os

def generate_meta_info(root_folder, output_file):
    """
    扫描指定的 'train' 目录，并生成一个包含正确相对路径的 meta_info 文件。
    """
    print(f"正在扫描目录: {root_folder}")
    
    # 检查根目录是否存在
    if not os.path.isdir(root_folder):
        print(f"错误：目录 '{root_folder}' 不存在。请检查路径。")
        return

    count = 0
    with open(output_file, 'w') as f:
        # 遍历根目录下的所有分类文件夹 (nXXXXXXXX)
        for category_folder in sorted(os.listdir(root_folder)):
            category_path = os.path.join(root_folder, category_folder)
            
            # 确保它是一个文件夹
            if os.path.isdir(category_path):
                # 遍历分类文件夹下的所有图片
                for image_name in sorted(os.listdir(category_path)):
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # 写入正确的相对路径，例如: n01440764/n01440764_10026.JPEG
                        relative_path = os.path.join(category_folder, image_name)
                        f.write(f"{relative_path}\n")
                        count += 1

    print(f"完成！总共找到 {count} 张图片。")
    print(f"新的 meta info 文件已保存至: {output_file}")

if __name__ == '__main__':
    # --- 请根据您的实际情况修改下面这两个路径 ---

    # 1. 指向您的 ImageNet 'train' 目录的路径 (该目录下应包含 nXXXXXXXX 文件夹)
    imagenet_train_path = '/dataset/ImageNet/ILSVRC2012_TRAIN/train' 
    
    # 2. 您希望将新生成的 meta_info.txt 文件保存到哪里
    #    建议放在项目文件夹下，并命名为新的文件名以作区分
    output_meta_file = '/root/project/UPSR/basicsr/data/meta_info/meta_info_ImageNet_GT_correct.txt'
    
    # --- 修改结束 ---

    generate_meta_info(imagenet_train_path, output_meta_file)