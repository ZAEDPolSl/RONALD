import os
import glob


def ends_with(string, extension_list):
    """check whether the strings ends with any of the extensions in the list"""
    for extension in extension_list:
        if string.endswith(extension):
            return True
    return False


def listdirs(rootdir):
    """list all files in a dir recursively"""
    dirs = []
    for path in glob.glob(f"{rootdir}/*/**/", recursive=True):
        dirs.append(path)
    dirs.append(rootdir)
    return dirs


def valid_dicom_file(path):
    """Check if on 0x80 offset of the file is b'DICM' filed - this tells if it is DICOM file"""
    try:
        with open(path, "rb") as file:
            file.seek(128)
            valid_dicom = file.read(4) == b'DICM'
    except:
        if not os.path.isdir(path):
            print(f"DICOM validation failed at {path}")
        valid_dicom = False
    return valid_dicom