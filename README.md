# PBL project: Chinese character recognition With Adaboost Algorithm

김예빈(2017036480)
송유진(2016033063)
양효용(2017057410)
최윤정(2016033327)

최종 코드는 zip으로 올렸습니다.
위에는 일부 코드입니다.

데이터가 있는 구글 드라이브 입니다.
https://drive.google.com/drive/folders/1JfKpBbYu8yE0TJ9evEnC4h_wQhy9r9V1?usp=sharing

감사합니다.

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


그래서 저희는 부건에 해당하는 haar feature selection을 위해서 기존의 직사각형이 아닌 13가지의 부건결합 모형에 따른
haarlike feature selection 함수를 제작하고 있습니다.
13가지 부건결합 모형은 다음과 같습니다.
![부건13](https://user-images.githubusercontent.com/48639285/80779672-54662100-8ba7-11ea-93a8-902a57b04167.PNG)
