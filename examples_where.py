import os

def list_files_in_directory(directory):
    try:
        # 列出指定目录下的所有文件和文件夹
        files = os.listdir(directory)
        for file in files:
            # 判断是否为文件
            # if os.path.isfile(os.path.join(directory, file)):
            print(file)
    except FileNotFoundError:
        print(f"目录 {directory} 不存在")
    except Exception as e:
        print(f"发生错误: {e}")

# 示例: 列出当前目录下的所有文件
list_files_in_directory(".")
