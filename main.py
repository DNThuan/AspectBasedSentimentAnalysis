from utils import preprocessing


texts = ["Kháchhhh Sạn đẹp, dc cái giá rẻ chỉ hon 300k một đêm @@@",
        "Khách sạn gần trung tâm, cách chợ 500m"]

pre_review = preprocessing.Preprocessing_Review(texts).process()


print(texts)
print(pre_review)



