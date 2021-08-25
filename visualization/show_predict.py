import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def show_imgs(data = None, path: str = "train/images/",
              row: int = 5, col: int = 5, 
              width: int = 10, height: int = 15, 
              start_position: int = 0,
              img_type: str = "normal", img_extension = ".jpg", n = 0, classes = 0):

  
  fig, axes = plt.subplots(row, col, figsize = (width, height))
  
  
  if path[-1] != "/":
    path = path + "/"
  
  for i in range(row):
    for j in range(col):
      idx   = start_position + i * col + j
    
      while data.ans[n] != classes:
          n += 1
      file_name = data.ImageID[n]
      suptitle = data.Gender[n] +' '+ data.Age[n] +' '+ data.Mask[n]
      #print(file_name, n)
      n+=1
      img_path = path + 'images/' + file_name
      title = "inference"

      image = Image.open(img_path)
      if i == 0 and j == 0:
        print("Image size:", image.size)

      axes[i][j].imshow(image)
      axes[i][j].set_title(title)
      axes[i][j].axis('off')


  plt.suptitle(suptitle)
  plt.show()
  return n

if __name__ == "__main__":
  class_num = 11
  show_imgs_num = 2
  
  data = pd.read_csv('/opt/ml/input/data/eval/submission.csv')

  data['Gender'] = 'female'

  data.loc[data.ans % 6 == 0, 'Gender'] = 'male'
  data.loc[data.ans % 6 == 1, 'Gender'] = 'male'
  data.loc[data.ans % 6 == 2, 'Gender'] = 'male'

  data['Mask'] = 'Wear'

  data.loc[data.ans > 5, 'Mask'] = 'Incorrect'
  data.loc[data.ans > 11, 'Mask'] = 'Not Wear'

  data['Age'] = ''
  data.loc[data.ans % 3 == 0, 'Age'] = '~29'
  data.loc[data.ans % 3 == 1, 'Age'] = '30~59'
  data.loc[data.ans % 3 == 2, 'Age'] = '60~'

  print(data.groupby(['Mask', 'Gender', 'Age'])['ans'].count())

  k = 0
  for i in range(show_imgs_num):
      k = show_imgs(data=data, path="/opt/ml/input/data/eval", img_type="normal", n=k, classes=class_num)
