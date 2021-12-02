
from . import preprocess_review as pre_review
from . import preprocess_label as pre_label

class Preprocessing_Review(object):
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
          result = pre_review.lower_text(result)
        if self.delete_emoji:
          result = pre_review.delete_emoji(result)
        if self.replace_symbol:
          result = pre_review.replace_symbol(result)
        if self.delete_special_character:
          result = pre_review.delete_special_character(result)
        if self.replace_negative_words:
          result = pre_review.replace_negative_words(result)
        if self.normalize_elongate_words:
          result = pre_review.normalize_elongate_words(result)
        return result


class Preprocessing_Label(object):
    def __init__(self, label = "",
                      aspect = False,
                      positive = False,
                      negative = False,
                      neutral = False):
      self.label = label
      self.aspect = aspect
      self.positive = positive
      self.negative = negative
      self.neutral = neutral
    
    def make_label_dataframe(self):
 
      if self.aspect:
        df = pre_label.get_aspect_data_frame(self.label)
      elif self.positive:
        df = pre_label.get_positive_data_frame(self.label)
      elif self.negative:
        df = pre_label.get_negative_data_frame(self.label)
      else:
        df = pre_label.get_neutral_data_frame(self.label)

      return df
