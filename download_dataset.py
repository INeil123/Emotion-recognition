import kagglehub
import os

def download_dataset():
    # 创建models目录（如果不存在）
    if not os.path.exists('models'):
        os.makedirs('models')
    
    print("开始下载FER2013数据集...")
    # 下载数据集
    path = kagglehub.dataset_download("msambare/fer2013")
    print("数据集下载完成！")
    print("数据集路径:", path)
    
    # 打印下载目录内容
    print("下载目录内容:")
    for file in os.listdir(path):
        print(file)
    
    # 将fer2013.csv复制到当前目录
    csv_path = os.path.join(path, 'fer2013.csv')
    if os.path.exists(csv_path):
        import shutil
        shutil.copy2(csv_path, 'fer2013.csv')
        print("fer2013.csv已复制到当前目录")
    else:
        print("警告：未找到fer2013.csv文件！")

if __name__ == "__main__":
    download_dataset() 