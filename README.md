# PBL project: Chinese character recognition With Adaboost Algorithm
김예빈(2017036480)
송유진(2016033063)
양효용(2017057410)
최윤정(2016033327)


****
### 연구의 필요성
![스크린샷(152)](https://user-images.githubusercontent.com/48639285/79532993-c1f75500-80b1-11ea-970d-c1b4191b997b.png)
![스크린샷(153)](https://user-images.githubusercontent.com/48639285/79532998-c3c11880-80b1-11ea-83f1-95ed46cec5e8.png)
![스크린샷(154)](https://user-images.githubusercontent.com/48639285/79533000-c4f24580-80b1-11ea-9ef6-545923b538e7.png)
![스크린샷(155)](https://user-images.githubusercontent.com/48639285/79533001-c4f24580-80b1-11ea-8105-2fb766dc7bfc.png)
![스크린샷(156)](https://user-images.githubusercontent.com/48639285/79533002-c58adc00-80b1-11ea-8817-62d4d6725efe.png)


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


##### 다섯 번째 파일: adaboost.py
-decision stump를 사용해 weak learner를 생성하였습니다.
의사결정 스텀프 생성 함수를 stumpClassify()만들어 입력가능한 모든 값을 buildstump()에서 반복하고 데이터 집합세어 가장 좋은 의사결정 스텀프를 찾아 이 데이터 가중치 벡터 D에 보관하였습니다.


##### 여섯 번째 파일: feature_bugan.py

논문들에서 어떤 weak classifier를 사용했는지 정리해본 결과 SingleStump weak classifier 와 Haar-LikeStump weak classifier와 최종 분류기를 추출해내는 알고리즘에서 차이가 있는 방법들이 있었습니다.
먼저 SingleStump weak classifier 는 피처를 이미지 픽셀값들로 넣어서 오차율이 50% 이상인 -1을 곱하고 오차율이 50퍼센트 이하인 경우 가중치에 1을 곱해줘서 훈련 설정 오류율을 50% 이하로 줄이는 방법입니다. 식은 아래와 같았습니다.
![adahx](https://user-images.githubusercontent.com/48639285/80779442-9e023c00-8ba6-11ea-80c6-a46bc5f4f92d.png)

그리고 haar-likestump weak classifier는

![adaintel](https://user-images.githubusercontent.com/48639285/80779425-8dea5c80-8ba6-11ea-96cc-b70618c072ee.png)
와 같이 좌표와 픽셀값을 추출하여 적분을 통해 이미지의 직사각형 피처에 대해서 훈련하는 약분류기였습니다.

그리고 멀티클래스 분류를 활용한 요소 분류기 (MQDF)의 weak classifier는
![멀티클래스 분류를 활용한 요소분류기](https://user-images.githubusercontent.com/48639285/80782700-2fc37680-8bb2-11ea-8762-825e04b7ae00.PNG) 

또한 [유사한 필기 한자 구별을 위한 다중 인스턴스 학습 기반 방법.docx](https://github.com/datacheff/pbladaboost/files/4562515/default.docx)논문에서도 이러한 방법을 사용한다는 것으로 정리해볼 수 있었습니다.


그래서 저희는 부건에 해당하는 haar feature selection을 위해서 기존의 직사각형이 아닌 13가지의 부건결합 모형에 따른
haarlike feature selection 함수를 제작하고 있습니다.
13가지 부건결합 모형은 다음과 같습니다.
![부건13](https://user-images.githubusercontent.com/48639285/80779672-54662100-8ba7-11ea-93a8-902a57b04167.PNG)
