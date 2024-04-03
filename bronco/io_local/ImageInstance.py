import os
import uuid
import time
import random
import SimpleITK as sitk
from .utils import valid_dicom_file, listdirs, ends_with

sitk.ProcessObject_SetGlobalWarningDisplay(False)

SUPPORTED_EXTENSIONS = [".nrrd", ".nii.gz", ".nifti"]


class ImageInstance:
    def __init__(self, show_exceptions=True):
        self.image_reader = None
        self.mode = None
        self.image = None
        self.new_series_uid = ""
        self.show_exceptions = show_exceptions

    def init_reader(self, input_path, series_uid=None, read_meta=True):
        if os.path.isdir(input_path):
            # if no series uid is given read the first series from the folder
            if series_uid is None:
                series_uid = self.get_first_series(input_path)

            # get series file names
            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_path, series_uid)
            if self._validate_dicom_series(series_file_names):
                # initialize the reader
                self.image_reader = self._get_reader(mode='series')
                self.image_reader.SetFileNames(series_file_names)
            else:
                if self.show_exceptions:
                    raise ValueError(f"Folder {os.path.basename(input_path)} does not contain a valid DICOM series!")
                else:
                    return False

        elif ends_with(input_path, SUPPORTED_EXTENSIONS):
            self.image_reader = self._get_reader(mode='file')
            self.image_reader.SetFileName(input_path)
            if read_meta:
                self.image_reader.LoadPrivateTagsOn()
                self.image_reader.ReadImageInformation()

        else:
            if self.show_exceptions:
                raise AttributeError(f"Input data of format {os.path.basename(input_path)} not supported!")
            else:
                return False
        return True

    def read(self, input_path, series_uid=None):
        """
        Read DICOM, nrrd or nifti.

        Parameters
        ----------
        input_path : path to the image location
        series_uid : Series Instance UID, if not given and DICOM is being read, system will pick DICOM series with
                    the most numbers of slices.

        Returns
        -------
        input_image : sitk.Image

        """
        if self.image_reader is None:
            if self.init_reader(input_path, series_uid):
                input_image = self.image_reader.Execute()
            else:
                raise ValueError(f"Could not read the DICOM {os.path.basename(input_path)}!")
        else:
            input_image = self.image_reader.Execute()
        return input_image

    def get_series_instance_uid(self):
        return self.image_reader.GetMetaData(0, "0020|000e")

    def get_new_series_instance_uid(self):
        return self.new_series_uid

    def get_study_instance_uid(self):
        return self.image_reader.GetMetaData(0, "0020|000d")

    def get_slice_thickness(self):
        return self.image_reader.GetMetaData(0, "0018|0050")

    def get_convolution_kernel(self):
        return self.image_reader.GetMetaData(0, "0018|1210").replace("\\", "_").rstrip().lstrip()

    def get_filter_type(self):
        return self.image_reader.GetMetaData(0, "0018|1160").replace("\\", "_").rstrip().lstrip()

    def get_manufacturer(self):
        return self.image_reader.GetMetaData(0, "0008|0070").rstrip().lstrip()

    def get_accession_number(self):
        return self.image_reader.GetMetaData(0, "0008|0050")

    def get_contrast_agent(self):
        try:
            return self.image_reader.GetMetaData(0, "0018|0010")
        except:
            return ""

    def write(self, image, output_path, description='Segmentation', relate_dirs_to_uids=False, forced_mode=None,
              series_number=None):
        """
        Write DICOM or nrrd.

        Parameters
        ----------
        image : ndarray/sitk.Image, image content to write, it should be related to the one which was read.
        output_path : str, path to save DICOM or nrrd.
        description : str, if DICOM this string is the description,
        relate_dirs_to_uids : bool, if True, under the output path there are created folders study_uid/series_uid,
        forced_mode : str, 'series' - forced DICOM write or 'file' - forced nrrd wrtie

        Notes
        -------
        Saved image should be related to previously read image because a DICOM header information is going to be copied.
        """
        if type(image) is not sitk.Image:
            image = sitk.GetImageFromArray(image)
        self._handle_mode(output_path, forced_mode=forced_mode)
        if self.mode == 'series':
            self._write_dicom(image, output_path, description, relate_dirs_to_uids, series_number=series_number)
        elif self.mode == 'file':
            self._write_nrrd(image, output_path, relate_dirs_to_uids)

    def _get_reader(self, mode='series'):
        """
        Get Series Reader for DICOM or File Reader for NRRD, NIFTI, PNG itp.
        Parameters
        ----------
        mode : str, 'series' - dicom image, 'file' - others

        Returns
        -------
        reader : sitk.ImageSeriesReader or sitk.ImageFileReader
        """
        if mode == 'series':
            reader = sitk.ImageSeriesReader()
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            self.mode = mode
        elif mode == 'file':
            reader = sitk.ImageFileReader()
            self.mode = mode
        else:
            raise AttributeError(f"Value of mode={mode} don't recognised, accepting ['series', 'file']!")
        return reader

    def _write_dicom(self, image: sitk.Image, output_path, description='Segmentation', relate_dirs_to_uids=False,
                     dir_relation_scheme='simplified', series_number=None, unique_patient_id=None, facility='unknown'):
        # prepare path
        if relate_dirs_to_uids:
            _output_path = self.create_relation_path(output_path, dir_relation_scheme,
                                                     unique_patient_id, facility)
        else:
            _output_path = output_path
        os.makedirs(_output_path, exist_ok=True)
        # initialize reader
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        # get modification time
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        tags_to_add = []

        if self.image_reader is not None:
            # Copy relevant tags from the original meta-data dictionary (private tags are also
            # accessible).
            tags_to_copy = ["0010|0010",  # Patient Name
                            "0010|0020",  # Patient ID
                            "0010|0030",  # Patient Birth Date
                            "0020|000d",  # Study Instance UID, for machine consumption
                            "0020|0010",  # Study ID, for human consumption
                            "0008|0020",  # Study Date
                            "0008|0030",  # Study Time
                            "0008|0050",  # Accession Number
                            "0008|0060",  # Modality
                            "0008|1030",  # Study Description
                            "0008|0070",  # Manufacturer
                            "0018|1210",  # Convolution Kernel
                            "0008|1090",  # Manufacturer's model name
                            "0010|0040",  # Patient's Sex
                            "0010|1010",  # Patient's Age
                            "0018|0060",  # KVP
                            "0018|1160",  # Filter type
                            "0018|5100",  # Patient position
                            "0018|9324",  # Estimated dose saving
                            "0018|9345",  # CTDIvol
                            "0018|0010",  # Contrast Agent
                            ]

            tags_to_copy_per_slice = [
                # "0028|0030",  # Pixel Spacing
                "0020|0032",  # Image Position (Patient)
                "0020|0037",  # Image Orientation (Patient)
                "0028|1050",  # Window Center
                "0028|1051",  # Window Width
            ]
            original_image = self.image_reader.Execute()
            spacing = original_image.GetSpacing()
        else:
            tags_to_copy = []
            tags_to_copy_per_slice = []
            new_study_uid = "1.2.826.0.1.3680043.10.877." + modification_date + ".2" + modification_time + ".2" \
                             + str(random.randint(100000, 999999))
            tags_to_add = [
                ("0008|0020", modification_date),  # Study Date
                ("0008|0030", modification_time),  # Study Time
                ("0010|0020", str(int(uuid.uuid4()))),  # Patient's ID
                ("0020|0010", str(uuid.uuid4())),  # Study ID, for human consumption
                ("0020|0011", str(1)),  # Series Number
                ("0020|000d", new_study_uid),  # Study Instance UID
                ("0010|0010", "ANON NO REF SAVE"),  # Patient's Name
                ("0008|0050", str(random.randint(10000, 99999)) + "/POLSL/" + modification_date +
                 "/" + modification_time)  # Accession Number
            ]
            spacing = image.GetSpacing()

        # Copy some of the tags and add the relevant tags indicating the change.
        # For the series instance UID (0020|000e), each of the components is a number, cannot start
        # with zero, and separated by a '.' We create a unique series ID using the date and time.
        # tags of interest:
        new_series_uid = "1.2.826.0.1.3680043.10.877." + modification_date + ".1" + modification_time + ".1" \
                         + str(random.randint(100000, 999999))
        self.new_series_uid = new_series_uid
        direction = image.GetDirection()
        series_tag_values = [(k, self.image_reader.GetMetaData(0, k)) for k in tags_to_copy if
                             self.image_reader.HasMetaDataKey(0, k)] + \
                            [("0008|0031", modification_time),  # Series Time
                             ("0008|0021", modification_date),  # Series Date
                             ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
                             ("0020|000e", new_series_uid),  # Series Instance UID
                             ("0020|0037", '\\'.join(
                                 map(str, (direction[0], direction[3], direction[6],
                                           direction[1], direction[4], direction[7])))
                              ),  # Image Orientation (Patient)
                             ("0008|103e", description)  # Series Description
                             ] + tags_to_add

        for instance_number, i in zip(range(image.GetDepth()), range(image.GetDepth() - 1, -1, -1)):
            image_slice = image[:, :, i]

            # Tags shared by the series.
            for tag, value in series_tag_values:
                image_slice.SetMetaData(tag, value)
            # Slice specific tags.
            image_slice.SetMetaData("0008|0012", modification_date)  # Instance Creation Date
            image_slice.SetMetaData("0008|0013", modification_time)  # Instance Creation Time

            for key in tags_to_copy_per_slice:
                try:
                    value = self.image_reader.GetMetaData(i, key)
                    image_slice.SetMetaData(key, value)
                except:
                    pass

            image_slice.SetMetaData("0020|0013", str(instance_number))  # Instance Number

            if series_number is not None:
                image_slice.SetMetaData("0020|0011", str(series_number))  # Series Number

            image_slice.SetSpacing(spacing)

            # Write to the output directory and add the extension dcm, to force writing in DICOM format.
            writer.SetFileName(os.path.join(_output_path, str(instance_number) + '.dcm'))
            writer.Execute(image_slice)

    def _write_nrrd(self, image, output_path, relate_dirs_to_uids=True):
        if relate_dirs_to_uids:
            study_uid = self.image_reader.GetMetaData(0, "0020|000d")
            series_uid = self.image_reader.GetMetaData(0, "0020|000e")
            _output_path = os.path.join(output_path, study_uid)
            os.makedirs(_output_path, exist_ok=True)
        else:
            _output_path = output_path
        if not _output_path.endswith('.nrrd'):
            _output_path = _output_path + ".nrrd"
        # get information
        if self.image_reader is not None:
            image_original = self.image_reader.Execute()
            image.CopyInformation(image_original)
        # save image
        _output_path = os.path.join(_output_path)
        sitk.WriteImage(image, _output_path)

    def _handle_mode(self, output_path, forced_mode):
        if forced_mode is not None:
            self.mode = forced_mode
        if ends_with(output_path, SUPPORTED_EXTENSIONS):
            self.mode = 'file'

    def create_relation_path(self, output_path, dir_relation_scheme='simplified', unique_patient_id=None,
                             facility='unknown'):
        if dir_relation_scheme == 'simplified':
            study_uid = self.image_reader.GetMetaData(0, "0020|000d")
            series_uid = self.image_reader.GetMetaData(0, "0020|000e")
            _output_path = os.path.join(output_path, study_uid, series_uid)
            os.makedirs(_output_path, exist_ok=True)
        elif dir_relation_scheme == 'covrad':
            study_uid = self.image_reader.GetMetaData(0, "0020|000d")
            series_uid = self.image_reader.GetMetaData(0, "0020|000e")
            study_date = self.image_reader.GetMetaData(0, "0008|0020")
            manufacturer = self.image_reader.GetMetaData(0, "0008|0070").replace("\\", ".").strip()
            if manufacturer == "Hitachi Medical Corporation":
                manufacturer = "Hitachi"
            try:
                filter_type = str(self.image_reader.GetMetaData(0, "0018|1160")).replace("\\", ".").strip()
            except:
                filter_type = 'unknown'
            try:
                convolutional_kernel = str(self.image_reader.GetMetaData(0, "0018|1210")).replace("\\", ".").strip()
            except:
                convolutional_kernel = 'unknown'
            facility = facility
            study = "$_$".join([study_date, study_uid])
            series = "$_$".join([manufacturer, filter_type, convolutional_kernel, series_uid])
            if unique_patient_id is None:
                _output_path = os.path.join(output_path, facility, study, series)
            else:
                patient = unique_patient_id
                _output_path = os.path.join(output_path, facility, patient, study, series)
            os.makedirs(_output_path, exist_ok=True)
        else:
            _output_path = output_path
            print("Warning: Exception during relation path creation, returning the original output path!")
        return _output_path

    @staticmethod
    def find_proper_series(input_path):
        series_filenames = dict()

        for nested_dir in listdirs(input_path):

            series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(nested_dir))

            for series_ID in series_IDs:
                series_filenames[series_ID] = (
                    nested_dir, len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(nested_dir, series_ID)))

        proper_series_ID = max(series_filenames.keys(), key=(lambda key: series_filenames[key][1]))
        path_to_proper_series = series_filenames[proper_series_ID][0]

        return proper_series_ID, path_to_proper_series

    @staticmethod
    def get_first_series(input_path):
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(input_path)
        return series_IDs[0]

    @staticmethod
    def _validate_dicom_series(series_file_names):
        """Some DICOM series happens to be study description and are not meant to be read by this class."""
        if len(list(series_file_names)) < 10:
            shapes = []
            for s in list(series_file_names):
                image = sitk.ReadImage(s)
                shapes.append(image.GetSize())
            shape_consistency_check = [shapes[0] == shape
                                       for shape in shapes]
            if not all(shape_consistency_check):
                return False
        is_first_a_valid_dicom_file = valid_dicom_file(series_file_names[0])
        return is_first_a_valid_dicom_file
