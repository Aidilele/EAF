import h5py
import numpy as np
import yaml


def read_hdf5_files(task):
    ob_values = {}
    with h5py.File("./dataset/datasets/" + task + ".hdf5", "r") as f:
        for key in f.keys():
            value = f[key]
            try:
                for sub_key in value.keys():
                    ob_values[sub_key] = np.array(value[sub_key])
            except:
                ob_values[key] = np.array(value)
    return ob_values


def load_yaml(path: str):
    with open(path, encoding='utf-8') as file:
        try:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)  # noqa: S506
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{path} error: {exc}') from exc

    return kwargs
