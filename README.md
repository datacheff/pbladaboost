# PBL project: Chinese character recognition With Adaboost Algorithm
김예빈(2017036480)
양효용(2017057410)
송유진(2016033063)
최윤정(2016033327)


****

### 사용한 데이터 셋
##### http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
에서 **CASIA-HWDB1.1** 데이터 셋을 사용하였습니다. 
gnt 확장자로 되어 있는 train dataset과 test dataset을 모두 다운 받았습니다.

### 1) 데이터셋 포맷 변환
##### 기본 파일: utils.py

##### 첫 번째 파일: 1gntchange.py
여기에서는 gnt 확장자로 되어있는 파일을 HDF5 binary data format으로 바꿨습니다.

##### 두 번째 파일: 2tosubset.py
여기에서는 HDF5 data set을 200개 문자 클래스로 변환하였습니다.

### 2) adaboost 생성
##### 세 번째 파일: whynot.py
테스트 데이터는 저장된 라벨이 순서대로 저장되어 있지 않고 랜덤하게 저장되어 있어
이에 대해서 tstlist.txt 파일로 라벨을 저장하였습니다.
200개의 한자 라벨에서 1이 몇번째 있는지를 받아와 몇번째 한자인지를 숫자로 저장하였습니다.

**200개의 한자 라벨은 다음과 같습니다.**
'谈','般','盏','坤','膀','脂','型','骏','童','挟','损','恋','婴','读','账','服','任','茸','张','亢','耀','涉',
'个','随','挂','抗','贞','瞥','瘤','作','河','欲','侵','吸','眺','线','捂','倾','牌','筒','渊','拥','话','赞',
'知','除','巩','惫','揭','扬','驼','绿','渔','榆','辊','应','儡','假','崩','抬','是','讲','刷','鸿','契','寒',
'录','教','也','艾','囤','秦','峨','括','诲','滴','凶','须','孽','巾','沉','餐','暂','蒙','攘','键','厄','的',
'芭','岳','惜','椰','足','伴','离','笼','临','胁','泉','晚','迟','汞','级','跳','轴','偶','啸','移','贾','老',
'节','蜗','堑','帕','肖','伟','渝','撮','臀','吉','汉','反','双','坏','翔','胖','绪','固','舀','再','咏','堂',
'尔','沟','符','涵','水','误','岿','所','摄','广','结','学','苫','臭','恬','诱','递','烷','硼','茁','标','越',
'吏','笑','馒','耗','氟','加','砧','稻','晃','臂','其','配','城','筑','痹','揖','江','连','卡','狠','瓤','乳',
'赵','仿','睹','相','好','屿','争','袭','王','吃','疏','粕','涟','垣','逢','锤','覆','薯','贴','冷','霸','聂',
'糕','占'


##### 네 번째 파일: makedataset.py
hdf5에 넣어져 있는 ['trn/ x'],['tst/x'] 데이터 셋을 이차원 행렬로 변환시켰습니다.
['trn/y'],['tst/y']에 저장된 한자 라벨을 리스트로 변환하였습니다.


