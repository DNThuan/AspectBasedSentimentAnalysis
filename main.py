from utils import preprocessing
import json
from sklearn.preprocessing import MultiLabelBinarizer

aspect_path = "./Label/aspect.json"
SA_path = "./Label/SA.json"

def read_label(path):
  with open(path) as f:
    data = json.load(f)
  f.close()
  return data
  
aspect_labels = read_label(aspect_path)
AS_labels = read_label(SA_path)

transform_label_aspect = MultiLabelBinarizer().fit([aspect_labels])
list_label_aspect = transform_label_aspect.classes_

transform_label_SA = MultiLabelBinarizer().fit([AS_labels])


y_label = ['{HOTEL#DESIGN&FEATURES, positive}, {HOTEL#GENERAL, negative}',
            '{LOCATION#GENERAL, positive}']

pre_label = preprocessing.Preprocessing_Label(y_label,list_label_aspect,transform_label_aspect,aspect=True).make_label_dataframe()

print(pre_label)
