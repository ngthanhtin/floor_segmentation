import pandas as pd
import json
import glob

root_path = '/home/tin_nguyen/Downloads/dfsfdsfsdf/data_floor_segmentation/20_5_2020_4_undistortion_labeled/'
json_files = glob.glob(root_path + '/*.json')
csv_file_name = "abc_4.csv"

df = pd.DataFrame(columns=["imagePath","Pixels","Category"])

for json_path in json_files:
    # Read one json file
    with open(json_path) as f:
        data = json.load(f)

        image_path = root_path + "/" + data['imagePath']

        for i in range(len(data['shapes'])):
            label = data['shapes'][i]['label']
            points = data['shapes'][i]['points']

            new_points = []
            for i, p in enumerate(points):
                new_points.append(int(p[0]))
                new_points.append(int(p[1]))
            
            new_row = {'imagePath': image_path, 'Pixels':new_points, 'Category':label}
            df = df.append(new_row, ignore_index=True)

df.to_csv(csv_file_name)