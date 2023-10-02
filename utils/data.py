import os
import polars as ps

def read_folder(folderpath='./input'):
    files = {}
    for directory, _, filenames in os.walk(folderpath):
        for filename in filenames:
            files[filename.split("_")[0]] = os.path.join(directory, filename)

    return files

def read_file(files: dict[str, str], code: str, input_cols: list[str], output_cols: list[str]):
    df = ps.read_csv(files[code])
    return df.select(input_cols), df.select(output_cols)

def get_as_df(data):
    return ps.DataFrame(data)
