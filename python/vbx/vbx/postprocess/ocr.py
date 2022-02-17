import numpy as np


lpr_chinese_characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "Anhui", "Beijing", "Chongqing", "Fujian",
         "Gansu", "Guangdong", "Guangxi", "Guizhou",
         "Hainan", "Hebei", "Heilongjiang", "Henan",
         "HongKong", "Hubei", "Hunan", "InnerMongolia",
         "Jiangsu", "Jiangxi", "Jilin", "Liaoning",
         "Macau", "Ningxia", "Qinghai", "Shaanxi",
         "Shandong", "Shanghai", "Shanxi", "Sichuan",
         "Tianjin", "Tibet", "Xinjiang", "Yunnan",
         "Zhejiang", "Police",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
         "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
         "U", "V", "W", "X", "Y", "Z"]

lpr_characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
         "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
         "U", "V", "W", "X", "Y", "Z"]


def ctc_greedy_decode(arr, merge_repeated=False, characters=None):
    blank_index = arr.shape[-1] - 1
    indices = [np.argmax(x) for x in arr if np.max(x) > 0]

    unique = []
    prev = None
    for idx in indices:
        if idx != blank_index:
            if merge_repeated:
                if not prev or prev != idx:
                    prev = idx
                    unique.append(idx)
            else:
                unique.append(idx)
        else:
            prev = None

    if characters == None:
        return unique
    else:
        return [characters[idx] for idx in unique]
