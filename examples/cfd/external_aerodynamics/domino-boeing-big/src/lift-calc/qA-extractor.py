import numpy as np
import pandas as pd
from pathlib import Path
import glob

base_dir = Path("/lustre/users/nashton/cadence/HiLiftAeroML/")

geo_folders = sorted(base_dir.glob("geo_*"))

results = []
missing_folders = []

for geo_folder in geo_folders:
    print('In folder ', geo_folder)
    try:
        geo_tag = geo_folder.name
        ref_files = list(geo_folder.glob("ref_values_*.csv"))
        ref_file = ref_files[0]
        df = pd.read_csv(ref_file)
        area_ref = df['areaRef'].iloc[0]
        q_ref = df['qRef'].iloc[0]

        results.append({'geometry_tag': geo_tag,
                        'areaRef': area_ref,
                        'qRef': q_ref
                        })
    except:
        print('Missing reference file for', geo_folder)
        missing_folders.append(geo_folder)

print('MISSING FOLDERS')
for folder in missing_folders:
    print(f"{folder}")

output_df = pd.DataFrame(results)
output_df.to_csv('qA.csv', index=False)