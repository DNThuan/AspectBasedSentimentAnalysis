import emoji
import re
import numpy as np
from sklearn.pipeline import make_pipeline
from pyvi import ViTokenizer, ViPosTagger


def lower_text(texts):
    texts_result = texts.copy()
    for i in range(len(texts_result)):
        texts_result[i]=texts_result[i].lower()
    return texts_result

def delete_emoji(texts):
    texts_result = texts.copy()
    texts_result =  np.array([emoji.get_emoji_regexp().sub('', text) for text in texts_result])
    return texts_result

def replace_symbol(texts):
  texts_result = []
  for text in texts:
    distance_pattern = "([0-9.,]{1,9}?.km)|([0-9.,]{1,9}?.cây số)|([0-9.,]{1,9}?.cây)|([0-9.,]{1,9}?.mét)|([0-9.,]{1,3}?.m)"
    new_text = re.sub(distance_pattern, 'khoang_cach', text)
    money_pattern = "(\d{1,3}k.{0})|([0-9.]{1,9}?.vnd)|([0-9.]{1,9}?.việt nam đồng)|([0-9.]{1,9}?.đồng)"
    new_text = re.sub(money_pattern, 'gia_tien', new_text)
    texts_result.append(new_text)
  return texts_result

def delete_special_character(texts):
  texts_result = []
  for text in texts:
    special_character_pattern = "[+=<>@#$%^&~]"
    new_text = re.sub(special_character_pattern, '', text)
    words = new_text.split()
    new_text = ' '.join(words)
    texts_result.append(new_text)
  return texts_result

def normalize_elongate_words(texts):
  texts_result = []
  for text in texts:
    elongate_pattern = r"(\w)\1*"
    new_text = re.sub(elongate_pattern, r'\1', text)
    texts_result.append(new_text)
  return texts_result


def replace_negative_words(texts):
  texts_result = []
  for text in texts:
    hotel_pattern = r"\bksạn\b|\bk sạn\b|\bks\b|\bKS\b|\bKs\b"
    new = re.sub(hotel_pattern, 'khách sạn', text)
    new = re.sub(r"\bnc\b", 'nước', new)
    new = re.sub(r"\bnvs\b|\bnhà vs\b", 'nhà vệ sinh', new)
    new = re.sub(r"\bnv\b", 'nhân viên', new)
    new = re.sub(r"\bvs\b", 'vệ sinh', new)
    negative_pattern = r"\bkh\b|\bko\b|\bkhg\b|\bkhong\b|\bk\b|\bhông\b|\bhem\b|\bk0\b"
    new = re.sub(negative_pattern, 'không', new)
    new = re.sub(r"\bdc\b|\bdk\b", 'được', new)
    new = re.sub(r" 1 ", " một ", new)
    texts_result.append(new)
  return texts_result


class Preprocessing_dataset(object):
    def __init__(self,texts ="",
                      lower_text=True, 
                      delete_emoji =True,
                      replace_symbol=True,
                      delete_special_character = True,
                      replace_negative_words=True,
                      normalize_elongate_words=True):
        self.texts = texts
        self.lower_text = lower_text
        self.delete_emoji = delete_emoji
        self.replace_symbol = replace_symbol
        self.delete_special_character = delete_special_character
        self.replace_negative_words = replace_negative_words
        self.normalize_elongate_words = normalize_elongate_words

    def process(self):
        result = self.texts
        
        if self.lower_text:
          result = lower_text(result)

        if self.delete_emoji:
          result = delete_emoji(result)

        if self.replace_symbol:
          result = replace_symbol(result)

        if self.delete_special_character:
          result = delete_special_character(result)

        if self.replace_negative_words:
          result = replace_negative_words(result)

        if self.normalize_elongate_words:
          result = normalize_elongate_words(result)

        return result


