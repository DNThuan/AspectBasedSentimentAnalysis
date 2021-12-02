from utils import preprocessing


y_label = ['{HOTEL#DESIGN&FEATURES, positive}, {HOTEL#GENERAL, negative}',
            '{LOCATION#GENERAL, positive}']

pre_label = preprocessing.Preprocessing_Label(y_label,aspect=True).make_label_dataframe()


print(pre_label)



