import wget
import shutil
import os


def download(url, name):
    print(f"Downloading {url}\n")
    zip_name = f"{name}.zip"
    wget.download(url, zip_name)

    print(f"\nExtracting {zip_name}\n")

    shutil.unpack_archive(zip_name, name)
    os.remove(zip_name)
    print(f"\nDone unpacking {name}")


download("https://deepquest.sfo2.digitaloceanspaces.com/deepstack/shared-files/sharedfiles.zip", "sharedfiles")
download("https://deepquest.sfo2.digitaloceanspaces.com/deepstack/shared-files/interpreter.zip", "interpreter")
download("https://deepquest.sfo2.digitaloceanspaces.com/deepstack/shared-files/redis.zip", "redis")
download("https://deepquest.sfo2.digitaloceanspaces.com/deepstack/windows_packages_cpu.zip",
         "windows_packages_cpu")
download("https://deepquest.sfo2.digitaloceanspaces.com/deepstack/windows_packages_gpu.zip",
         "windows_packages_gpu")
download("https://deepquest.sfo2.digitaloceanspaces.com/deepstack/shared-files/windows_setup.zip", "windows_setup")
