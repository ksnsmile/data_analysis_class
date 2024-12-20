# 데이터 기반 제조 수업(by 이주연교수님)

## 내용 : 소스코드,강의 pdf, 데이터

### EDA(Exploratory Data Analysis)
- 분석 작업 초기에 데이터의 상태를 파악하기 위해 하는 것 
- 데이터를 시각화하고 통계적 요약을 통해 이해하며, 패턴이나 특이점을 발견하기 위한 과정 
 - 상관계수 파악, 산점도 그리기, type 보기 등

### Data preprocessing

- 데이터 표준화와 정규화 하는 것 
- 표준화 : 데이터를 평균이 0이고 표준편차가 1로 변환하는 과정
  - 일반적으로 z score 값이 절대값이 2이상 이면 이상치라고 본다.
- 정규화 : 데이터 범위를 0에서 1구간으로 만드는것 
- 정규화 스케일링 할 때 학습 데이터와 테스트 데이터가 따로 있을때는 먼저 학습데이터에서 min max를 정하고 그것을 기준으로 테스트에서도 바꾸고 적용하기
즉 fit을 하는게 아니라 transfrom만 하면 된다.


## 라이브러리, 모듈, 패키지

### 라이브러리
 - 정의: 라이브러리는 특정 기능이나 작업을 수행하기 위해 여러 모듈과 패키지를 포함하는 코드의 모음입니다. 일반적으로 특정 도메인이나 작업에 대해 광범위한 기능을 제공합니다.

- 예시: requests 라이브러리는 HTTP 요청을 쉽게 처리할 수 있는 모듈들을 포함하고 있습니다. 사용자는 이 라이브러리를 통해 API와 통신할 수 있습니다:

### 패키지 

- 정의: 패키지는 여러 모듈을 포함하는 디렉토리입니다. 패키지를 사용하면 관련된 모듈들을 그룹화하여 더 쉽게 관리하고 사용할 수 있습니다.

- 구성: 패키지 디렉토리 내에는 __init__.py라는 파일이 있어야 하며, 이 파일은 해당 디렉토리가 패키지임을 나타냅니다. 이 파일이 없으면 Python은 해당 디렉토리를 패키지로 인식하지 않습니다.

- 예시: numpy는 여러 모듈을 포함하는 패키지입니다. 패키지를 임포트하는 방법은 다음과 같습니다:

### 모듈

- 정의: 모듈은 하나의 파일로 구성된 코드의 집합입니다. 일반적으로 Python 파일(예: example.py)이 하나의 모듈로 간주됩니다.

- 용도: 특정 기능이나 클래스를 정의하고, 이들을 다른 코드에서 쉽게 가져와 사용할 수 있도록 합니다.

- 예시: math 모듈은 수학 관련 함수들을 포함하고 있으며, 다음과 같이 임포트할 수 있습니다

### 결론 : 라이브러리 > 패키지 > 모듈


### Cloud computing service model

IaaS(Infrastructure as a service): 인프라 수준의 서비스로 서버와 스토리지 같은 리소스를 제공합니다.

PaaS(Platform as a service): 개발자에게 플랫폼 환경을 제공해 애플리케이션 개발과 배포를 지원합니다.

SaaS( Software as a service): 완전히 운영되는 소프트웨어를 제공하며, 사용자는 애플리케이션을 웹을 통해 사용합니다.
