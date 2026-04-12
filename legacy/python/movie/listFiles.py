import os


def get_files_recursive(folder):
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                file_list.append((file_path, file_size))
            except OSError as e:
                print(f"Error getting size of {file_path}: {e}")
    return file_list


def main():
    # folder = input("Enter the path to the folder: ")
    files = get_files_recursive("C:/Users/smt")
    sorted_files = sorted(files, key=lambda x: x[1], reverse=True)

    print(f"{'File Path':<100} {'Size (bytes)':>15}")
    print("=" * 115)
    for file_path, file_size in sorted_files[:100]:
        print(f"{file_path:<100} {file_size:>15}")


if __name__ == "__main__":
    main()
