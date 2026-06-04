import os
import shutil

# 基础目录
base_dir = '/home/zhaosong/PointCloud/flemme-main/data/MedPointS/completion'

# 需要处理的 fold 列表
folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

for fold in folds:
    fold_path = os.path.join(base_dir, fold)
    
    if not os.path.exists(fold_path):
        print(f"Folder {fold_path} does not exist, skipping.")
        continue
    
    for class_dir in os.listdir(fold_path):
        class_path = os.path.join(fold_path, class_dir)
        
        if os.path.isdir(class_path) and class_dir not in ['partial', 'target']:
            partial_src = os.path.join(class_path, 'partial')
            target_src = os.path.join(class_path, 'target')
            
            # 处理 partial：移动内容到 /fold1/partial/adrenalgland/
            if os.path.exists(partial_src):
                partial_dst_dir = os.path.join(fold_path, 'partial', class_dir)
                os.makedirs(partial_dst_dir, exist_ok=True)
                for item in os.listdir(partial_src):
                    src_item = os.path.join(partial_src, item)
                    dst_item = os.path.join(partial_dst_dir, item)
                    if os.path.isdir(src_item):
                        shutil.move(src_item, dst_item)
                    else:
                        shutil.move(src_item, dst_item)
                # 删除空的 partial_src 文件夹
                try:
                    os.rmdir(partial_src)
                except OSError:
                    pass
                print(f"Moved contents of {partial_src} to {partial_dst_dir}")
            
            # 处理 target：类似
            if os.path.exists(target_src):
                target_dst_dir = os.path.join(fold_path, 'target', class_dir)
                os.makedirs(target_dst_dir, exist_ok=True)
                for item in os.listdir(target_src):
                    src_item = os.path.join(target_src, item)
                    dst_item = os.path.join(target_dst_dir, item)
                    if os.path.isdir(src_item):
                        shutil.move(src_item, dst_item)
                    else:
                        shutil.move(src_item, dst_item)
                # 删除空的 target_src 文件夹
                try:
                    os.rmdir(target_src)
                except OSError:
                    pass
                print(f"Moved contents of {target_src} to {target_dst_dir}")
            
            # 删除空的类别文件夹
            try:
                if not os.listdir(class_path):
                    os.rmdir(class_path)
                    print(f"Removed empty directory: {class_path}")
            except OSError as e:
                print(f"Error removing {class_path}: {e}")

print("Reorganization completed!")