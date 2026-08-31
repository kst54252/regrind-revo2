import h5py
import numpy as np


def read_h5_to_dict(h5_file_path):
    def h5_to_dict(group):
        result = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                result[key] = h5_to_dict(item)
            else:
                # Handle scalar datasets properly
                if item.shape == ():  # scalar dataset
                    data = item[()]
                else:  # array dataset
                    data = item[:]

                # Convert bytes back to strings if needed
                if isinstance(data, bytes):
                    # Handle raw bytes (scalar strings)
                    result[key] = data.decode('utf-8')
                elif isinstance(data, np.ndarray) and data.dtype.kind == 'S':
                    if data.ndim == 0:  # scalar
                        result[key] = data.item().decode('utf-8')
                    else:  # array
                        result[key] = np.array([x.decode('utf-8') for x in data.flat]).reshape(data.shape)
                else:
                    result[key] = data
        return result

    with h5py.File(h5_file_path, 'r') as f:
        return h5_to_dict(f)
