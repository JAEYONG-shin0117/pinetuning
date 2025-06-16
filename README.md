# 자연어처리 2025-1 지정주제 기말 프로젝트: GPT-2 구축

팀원
2022113591 신재용
2020112036 김상현
2022113596 오유나


## GPT-2 기본 구현

#### 다음 각 모듈의 코드를 완성.
* `modules/attention.py`
* `modules/gpt2_layer.py`
* `models/gpt2.py`
* `classifier.py`
* `optimizer.py`
* `paraphrase_detection.py`: 
* `sonnet_generation.py`: 

## 파인튜닝 진행

#### 다음 파인튜닝의 방법들을 추가
* LoRA
* Adapter
* 데이터 증강
* Pre-LayerNorm
* Beam search

#### 실행방법 

## 환경 설정

#### GitHub에서 Source code 내려 받기:
* GitHub의 프로젝트 리포지토리를 클론
* Colab에 파일을 열어 다음을 실행

```
!git clone https://github.com/JAEYONG-shin0117/osss.git
```
#### 디렉토리 이동

cd osss

#### 필수 패키치 설치

!pip -r requirements.txt

#### 감정분류, 패러파이즈 탐지, 소넷 생성 실행

!python classifer.py --use_gpu

!python paraphrase_dection.py

!python sonnet_generation.py


* conda env create -f env.yml
* conda activate nlp_final  


