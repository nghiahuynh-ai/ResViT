import os

def find_files_by_ext(dir_path, extension=['.jpg', '.png'], acc=[]):
    if not os.path.isdir(dir_path):
        return
    files_and_dirs = os.listdir(dir_path)
    for file_or_dir in files_and_dirs:
        full_path = os.path.join(dir_path, file_or_dir)
        if os.path.isdir(full_path):
            acc = find_files_by_ext(full_path, extension, acc)
        else:
            for ext in extension:
                if full_path.endswith(ext):
                    acc.append(full_path)
    return acc