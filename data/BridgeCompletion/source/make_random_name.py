import glob
import rstr
import os
import os.path


def main():
    path = "./BridgeCompletion/source/make_random_name/"
    before_path = glob.glob(path + "complete/*")
    files = (before_path)
    print(files)
    for file in files:
        dir, ext = os.path.splitext(file)
        dir_path = os.path.dirname(file)

        path_name = rstr.xeger(r'^[0-9]{2}[0-9a-zA-Z0-9]{10}')
        rename_str = dir_path + '/' + path_name + ext

        os.rename(file, rename_str)
        os.makedirs(path + "partial/" + path_name)

if __name__ == "__main__":
    main()