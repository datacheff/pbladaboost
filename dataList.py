def chars() :
    chars = ['话',
             '知',
             '吉',
             '加',
             '挂',
             '抗',
             '拥',
             '扬',
             '脂',
             '胖',
             '崩',
             '胁',
             '涉',
             '河',
             '涵',
             '江',
             '般',
             '侵',
             '餐',
             '反'
             ]
    #print(len(chars))
    return chars

def buguns() :
    buguns = [[1, 2, 3],
              [1, 4],
              [1, 5],
              [1, 6],
              [7, 8, 8],
              [7, 9, 10],
              [7, 11],
              [7, 12],
              [13, 14, 15],
              [13, 16],
              [13, 13, 17],
              [13, 18],
              [19, 20],
              [19, 21],
              [19, 22],
              [19, 23],
              [24, 25, 26],
              [24, 27, 28, 29],
              [24, 30, 31, 32],
              [24, 33],
              ]
    return buguns

def haar_sizes() :
    haar_sizes = [
        [30, 30],
        [30, 70],
        [50, 50],
        [40, 70],
        [40, 40],
        [35, 70],
        [40, 90],
        [40, 60],
        [30, 30],
        [50, 50],
        [40, 80],
        [40, 70],
        [30, 70],
        [30, 30],
        [30, 30],
        [40, 80],
        [30, 80],
        [20, 70],
        [40, 80],
        [40, 80],
        [30, 80],
        [60, 90],
        [40, 50],
        [50, 40],
        [40, 70],
        [30, 30],
        [30, 80],
        [30, 30],
        [45, 30],
        [40, 55],
        [70, 30],
        [50, 50],
        [30, 60]
    ]
        # [26, 26],
        # [26, 62],
        # [39, 44],
        # [44, 64],
        # [41, 40],
        # [38, 59],
        # [33, 68],
        # [45, 71],
        # [41, 34],
        # [52, 50],
        # [46, 75],
        # [48, 70],
        # [40, 74],
        # [31, 34],
        # [36, 53],
        # [51, 86],
        # [49, 34],
        # [65, 68],
        # [24, 64],
        # [41, 89],
        # [44, 76],
        # [64, 71],
        # [50, 52],
        # [50, 46],
        # [49, 82],
        # [34, 40],
        # [34, 80],
        # [34, 40],
        # [43, 31],
        # [37, 57],
        # [60, 47],
        # [48, 54],
        # [41, 61],

    return haar_sizes

'''
char_dic = {}
for i in range(20) :
    char_dic[chars[i]] = buguns[i]
#print(char_dic)

#char_dic.keys()
lt = []

# 부건 하나 하나 호출
for i in char_dic.keys() :
     for j in char_dic[i] :
        #print(j)
        lt.append(j)


# 2번(火) 부건이 있는 한자만 호출

ans1 = []

for i in char_dic.keys() :
     for j in char_dic[i] :
        if j == 2 :
            ans1.append(i)
ans1

'''