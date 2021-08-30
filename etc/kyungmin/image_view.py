import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

#%config InlineBackend.figure_format='retina'

def show_imgs(data = None, path: str = "train/images/",
              row: int = 5, col: int = 5, 
              width: int = 10, height: int = 15, 
              start_position: int = 0,
              img_type: str = "normal", img_extension = ".jpg"):

  fig, axes = plt.subplots(row, col, figsize = (width, height))
  if path[-1] != "/":
    path = path + "/"
  
  for i in range(row):
    for j in range(col):
      idx   = start_position + i * col + j

      img_path = path + 'images/' + data.iloc[idx]['ImageID']
      print(img_path)
      #title = data["gender"].iloc[idx] + "/" + str(data["age"].iloc[idx])
      title = "inference"

      image = Image.open(img_path)
      if i == 0 and j == 0:
        print("Image size:", image.size)

      axes[i][j].imshow(image)
      axes[i][j].set_title(title)
      axes[i][j].axis('off')

  plt.show()

data = pd.read_csv("./eval/12submission.csv")
#print(data.iloc[0]['ImageID'])
print(data.value_counts(data['ans'].values, sort=True))
show_imgs(data=data, path="eval/", img_type="normal")