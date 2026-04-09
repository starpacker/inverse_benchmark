import numpy as np

import os

class DaxReader:
    """
    Simplified Dax Reader for STORM movies.
    Based on storm_analysis.sa_library.datareader.DaxReader
    """
    def __init__(self, filename):
        self.filename = filename
        self.image_height = None
        self.image_width = None
        self.number_frames = None
        self.bigendian = 0
        
        # Try to read .inf file
        dirname = os.path.dirname(filename)
        basename = os.path.splitext(os.path.basename(filename))[0]
        inf_filename = os.path.join(dirname, basename + ".inf")
        
        if os.path.exists(inf_filename):
            with open(inf_filename, 'r') as fp:
                for line in fp:
                    if "frame dimensions" in line:
                        dim_str = line.split("=")[1].strip()
                        w, h = dim_str.split("x")
                        self.image_width = int(w)
                        self.image_height = int(h)
                    elif "number of frames" in line:
                        self.number_frames = int(line.split("=")[1].strip())
                    elif "big endian" in line:
                        self.bigendian = 1
        
        if self.image_height is None:
            # Fallback based on file size assuming 256x256
            filesize = os.path.getsize(filename)
            if filesize % (256*256*2) == 0:
                self.image_height = 256
                self.image_width = 256
                self.number_frames = filesize // (256*256*2)
            else:
                raise ValueError("Could not determine DAX dimensions from .inf or file size.")

        self.fileptr = open(filename, "rb")

    def loadAFrame(self, frame_number):
        if frame_number >= self.number_frames:
            raise ValueError("Frame number out of range")
            
        self.fileptr.seek(frame_number * self.image_height * self.image_width * 2)
        image_data = np.fromfile(self.fileptr, dtype='uint16', count = self.image_height * self.image_width)
        image_data = np.reshape(image_data, [self.image_height, self.image_width])
        if self.bigendian:
            image_data.byteswap(True)
        return image_data.astype(np.float32)

    def close(self):
        self.fileptr.close()

def load_and_preprocess_data(file_path, frame_idx, offset, gain):
    """
    Loads a specific frame from a DAX file and applies preprocessing 
    (gain/offset correction).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    reader = DaxReader(file_path)
    try:
        raw_image = reader.loadAFrame(frame_idx)
    finally:
        reader.close()
        
    # Preprocessing
    image = (raw_image - offset) / gain
    image[image < 0] = 0
    
    return image
