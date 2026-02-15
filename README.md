# LifeClover — 장례·사후 정보 제공 RAG 챗봇

> 전체 프로젝트 레포지토리: [LifeClover](https://github.com/SKN19-3rd-4th-Project/Lifeclover-merged)<br>
> 본 문서는 프로젝트 전체가 아닌, **정보 챗봇 RAG 파이프라인**에 대한 본인 기여 부분을 정리한 것입니다.


## 프로젝트 개요

**LifeClover**는 장례 절차, 시설 정보, 지자체 지원금, 유산 상속, 디지털 유산 등 사후 행정 정보를 자연어로 질의하면 정확한 답변을 제공하는 RAG(Retrieval-Augmented Generation) 기반 챗봇입니다.


### 담당 범위

| 단계 | 파일 | 설명 |
|------|------|------|
| 데이터 파싱 실험 | `processing_and_upsert.ipynb` | pdfplumber 표추출 → EasyOCR → Clova OCR → pdfplumber bbox 전환 과정 |
| 조례 PDF 파싱·적재 | `insert_db_funeral_ordinance.ipynb` | 최종 pdfplumber bbox 로직으로 조례 데이터 추출 및 Pinecone 적재 |
| 시설 데이터 전처리·적재 | `insert_db_funeral_facilities.ipynb` | 구조적 데이터를 자연어로 변환 후 Pinecone 적재 |
| 검색 로직 | `search_info.py` | 지역 매칭, 멀티 지역 검색, 메타데이터 필터 분기 |
| 에이전트 프롬프트 설계 | `info_agent.py` | 도구 사용 규칙, 지역 처리 분기, 출력 포맷 가이드라인 |


### 기술 스택

| 분야                | 사용 도구 |
|---------------------|-----------|
| **Language**        | [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)](https://www.python.org/) |
| **Collaboration Tool** | [![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/) [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/) [![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)](https://www.notion.so/) [![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/) |
| **LLM Model**       | [![GPT-4o](https://img.shields.io/badge/GPT--4o%20-412991?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/) |
| **Embedding Model** | [![text-embedding-3-small](https://img.shields.io/badge/text--embedding--3--small-00A67D?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/embeddings) |
| **Vector DB**       | [![Pinecone](https://img.shields.io/badge/Pinecone-0075A8?style=for-the-badge&logo=pinecone&logoColor=white)](https://www.pinecone.io/) |
| **Orchestration / RAG** | [![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/) [![LangGraph](https://img.shields.io/badge/LangGraph-000000?style=for-the-badge)](https://langchain-ai.github.io/langgraph/) |
| **Development Env** | [![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)](https://code.visualstudio.com/) [![Conda](https://img.shields.io/badge/Conda-3EB049?style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com/)
<br>

---

## 1. 조례 PDF 파싱 — 다단 레이아웃 문제 해결

### 문제

수집한 전국 공영장례 조례집 및 화장장려금 조례집 PDF의 텍스트가 2~3단 컬럼으로 배치되어 있으나, **표(table) 구조가 없어** 단순 텍스트 추출 시 행·열 구분이 무너지는 문제가 있었습니다.

초기에는 텍스트를 직접 드래그하여 수작업으로 추출했으나, 전국 단위로 데이터가 확대되면서 자동화가 필요해졌습니다.

### 시행착오 과정 (`processing_and_upsert.ipynb`)
**1차: Pdfplumber의 표 추출 시도**  
Pdfplumber의 extract_tables() 등을 먼저 시도했으나, PDF에 표 구분 선이 없어 적용을 할 수 없었습니다. 이후 OCR을 시도하는 방향으로 전환하였습니다. 

**2차: EasyOCR 기반 접근**  
PDF를 이미지로 변환한 후 EasyOCR로 텍스트를 인식하고, 바운딩 박스의 x좌표 중앙값이 유사한 항목끼리 묶어 컬럼을 재구성하는 방식을 시도했습니다. DPI를 200, 300, 400으로 바꿔가며 테스트했으나 한글 인식률이 낮아 "을"→"올", "를"→"흘" 등의 오인식이 반복되었고, 데이터 신뢰도를 확보하기 어려웠습니다.

**3차: Naver Clova OCR 시도**  
한글 인식률 개선을 위해 Clova OCR API를 테스트했으나, API 비용 문제로 전체 데이터에 적용하기에는 적합하지 않다고 판단해 중단했습니다.

**4차 (최종): pdfplumber bbox 기반 접근**  
OCR 자체를 우회하는 방향으로 전환했습니다. pdfplumber의 `extract_words()`가 반환하는 좌표 정보(x1, x2, y1, y2)를 직접 활용하여, PDF 내부에 이미 존재하는 텍스트 위치 데이터로 컬럼을 구분하는 로직을 구현했습니다.

### 최종 파싱 로직의 핵심 설계 (`insert_db_funeral_ordinance.ipynb`)

좌표 기반으로 텍스트를 분류하는 규칙을 설계했습니다:

- **같은 줄 판정**: 이전 텍스트와 y좌표 차이가 `LINE_GAP` 이하이고, x 간격이 `MIN_COLUMN_GAP` 미만이면 같은 줄로 취급
- **새 컬럼 판정**: x 간격이 `MIN_COLUMN_GAP` 이상이거나, x 시작점이 `SOME_INT`(페이지 너비의 약 1/4) 이상이면 별도 컬럼으로 분류
- **컬럼 내 텍스트 수집**: 새 컬럼이 발견되면, x 중앙값(`cx`)이 `CENTER_X_GAP` 이내인 미사용 텍스트를 모두 해당 컬럼에 편입


### 판단 근거

| 방법 | 장점 | 한계 |
|------|------|------|
| EasyOCR | 무료, GPU 가속 가능 | 한글 인식률 낮음 |
| Clova OCR | 한글 인식률 높음 | API 비용 |
| **pdfplumber bbox** | **OCR 불필요, 좌표 정확, 무료** | 이미지 기반 PDF에는 적용 불가 |

조례집 PDF는 텍스트가 내장된 문서였기 때문에, OCR을 거치지 않고 좌표를 직접 활용하는 것이 정확도와 비용 모두에서 최선이었습니다.

---

## 2. 시설 데이터 전처리 — 구조적 데이터의 자연어 변환

### 배경과 설계 판단

장례 시설 데이터(장례식장, 봉안당, 묘지, 화장시설, 자연장지)는 이름, 주소, 전화번호, 주차 가능 여부 등 **정형 데이터**로, RDB로도 충분히 관리 가능한 데이터입니다.

그러나 프로젝트에서는 조례, 디지털 유산, 유산 상속 등 다른 비정형 데이터와 **Pinecone 하나의 인덱스에서 통합 관리**하는 구조를 택했습니다. 이때 정형 데이터를 그대로 임베딩하면 벡터 검색의 이점(의미 기반 매칭)을 살릴 수 없으므로, **임베딩에 맥락을 부여하기 위해** 자연어 텍스트로 변환하는 `generate_text_logic()`을 설계했습니다.

### `generate_text_logic()` — 시설 유형별 자연어 생성

**장례식장 전용 로직:**  
장례식장은 다른 시설과 달리 주차대수, 빈소 수, 편의시설(식당/매점/유족대기실/장애인편의시설), 공사설 구분, 운영 형태 등의 컬럼이 존재합니다. 이 정보를 조건에 따라 자연어 문장으로 서술합니다.

```
경기도 수원시 팔달구 ...에 위치한 ○○병원 장례식장입니다. 시설 유형은 장례식장입니다.
주차장이 있고, 주차 가능 대수는 200대이고, 빈소수는 5개이고, 식당은 있고, 매점은 있고,
유족대기실은 있고, 장애인편의시설은 없고 사설 위탁 장례식장입니다.
```

**기타 시설 로직 — 원본에 없는 맥락 생성:**  
봉안당, 묘지 등의 원본 데이터에는 종교 구분이나 공설/사설 구분 컬럼이 없었습니다(장례식장은 공사설 컬럼이 존재하여 그대로 활용). 이를 해결하기 위해:

1. 시설명에서 LLM을 활용하여 종교 관련 키워드 후보를 추출
2. 추출된 키워드를 수작업으로 검수하여, 확신할 수 있는 것만 키워드 리스트로 확정
3. 시설명에 해당 키워드가 포함되면 "천주교(가톨릭)와 관련이 있는 시설입니다" 등의 맥락을 자연어에 추가

```python
catholic_keywords = ['천주교', '가톨릭', '성당', '성요셉', '성모', '수녀원', ...]
buddhist_keywords = ['불교', '조계종', '태고종', '연화', '극락', ...]
```

이를 통해 사용자가 "천주교 납골당"처럼 질의할 때 의미 기반 검색이 가능해졌습니다.

### `extract_facilities_region_list()` — 검색 필터용 지역 리스트 추출

적재 시점에 시설 유형별로 존재하는 지역값을 JSON으로 추출하여 `facilities_region_list.json`을 생성했습니다. 이 파일은 이후 `search_info.py`에서 사용자 입력 지역을 매칭할 때 참조 리스트로 사용됩니다.

---

## 3. 검색 로직 설계 (`search_info.py`)

### 지역 매칭: `find_matching_regions()`

사용자가 입력하는 지역명은 형식이 다양합니다 ("서울 강남", "강남구", "서울특별시 강남구" 등). 이를 Pinecone 메타데이터 필터에 사용할 수 있는 정확한 지역명으로 변환하기 위해 2단계 매칭 로직을 구현했습니다.

1. **양방향 문자열 포함 체크**: 사용자 입력이 지역 리스트의 항목에 포함되거나, 반대로 지역 리스트 항목이 사용자 입력에 포함되는지 확인
2. **difflib 유사도 매칭**: 포함 관계가 없으면 `get_close_matches()`로 유사도 0.6 이상인 항목을 반환

이 함수는 조례 검색과 시설 검색 모두에서 공통으로 사용됩니다.

### 시설 검색: `search_funeral_facilities()`

장례 시설은 한 지역뿐 아니라 "서울 근처", "수도권" 등 넓은 범위에서 찾는 경우를 고려하여 **멀티 지역 검색**을 지원합니다.

설계 포인트:

- **k값 동적 조정**: `k = 10 // len(regions)` — 지역이 많아지면 지역당 반환 수를 줄여 응답 시간을 제어
- **지역·시설유형 조합 분기**: 지역이 1개일 때는 시설유형별 반복 검색, 지역이 복수일 때는 지역별 반복 검색으로 분기
- **중복 제거**: `page_content` 기준 set으로 중복 결과 필터링

### 조례 검색: 단일 지역 필수 분기

조례/지원금 검색(`search_public_funeral_ordinance`, `search_cremation_subsidy_ordinance`)은 **고인의 주민등록상 거주지** 기준이므로, 시설 검색과 달리 단일 지역만 받도록 설계했습니다. 매칭 결과가 1개면 `$eq`, 복수면 `$in` 필터로 분기합니다.

---

## 4. 에이전트 프롬프트 설계 (`info_agent.py`)

코드 자체는 LLM에 도구를 바인딩하고 호출하는 단순한 구조이지만, **시스템 프롬프트가 도구 사용 품질을 좌우**하는 핵심 요소였습니다.

도구의 docstring만으로는 LLM이 지역 처리 규칙(시설은 복수 지역 허용, 조례는 단일 지역 필수)이나 출력 포맷을 올바르게 따르지 않는 문제가 있었습니다. 이를 해결하기 위해 테스트를 반복하며 다음과 같은 규칙을 프롬프트에 명시했습니다:

- **시설 검색**: `regions` 인자에 리스트로 최대 3개 입력, 여러 지역·여러 시설은 병렬 호출
- **조례 검색**: 지역이 모호하면 되묻기, 단일 지역 문자열만 `region` 인자에 입력
- **출력 포맷**: 시설 정보는 번호+굵은 시설명+줄바꿈 상세정보, 지원금 정보는 불릿 포인트

---

## 5. 전체 아키텍처 요약

```
[사용자 질의]
     │
     ▼
[info_agent.py]  ── 시스템 프롬프트로 도구 선택 규칙 제어
     │
     ├── search_funeral_facilities()     ── 멀티 지역 + 시설유형 필터
     ├── search_public_funeral_ordinance()  ── 단일 지역 필터
     ├── search_cremation_subsidy_ordinance()
     ├── search_digital_legacy()
     └── search_legacy()
          │
          ▼
     [Pinecone]  ── 네임스페이스별 분리 (funeral_facilities / ordinance / digital_legacy / legacy)
          │
          ▼
     [LLM 응답 생성]
```

### 데이터 적재 흐름

```
[원본 데이터]
     │
     ├── 조례 PDF ──→ pdfplumber bbox 파싱 ──→ Pinecone (ordinance)
     │                (insert_db_funeral_ordinance.ipynb)
     │
     └── 시설 CSV ──→ generate_text_logic() 자연어 변환 ──→ Pinecone (funeral_facilities)
                      (insert_db_funeral_facilities.ipynb)
```

---

## 프로젝트 구조 (전체)

```
Lifeclover/
├── chatbot/
│   ├── ...                                   # 대화 흐름 제어 (타 팀원)
│   └── chatbot_modules/
│       ├── info_agent.py                     # 정보 에이전트 (본인 담당)
│       ├── search_info.py                    # 검색 로직 (본인 담당)
│       └── ...
│
├── scripts/
│   ├── insert_db_funeral_facilities.ipynb    # 시설 데이터 적재 (본인 담당)
│   ├── insert_db_funeral_ordinance.ipynb     # 조례 데이터 적재 (본인 담당)
│   └── processing_and_upsert.ipynb           # 파싱 실험 과정 (본인 담당)
│
├── data/
│   ├── raw/                                  # 원본 PDF·CSV
│   ├── processed/                            # 전처리 결과물
│   │   ├── facilities_region_list.json
│   │   └── ordinance_region_list.json
│   └── ...
│
├── config/                                   # Django 프로젝트 설정 (타 팀원)
├── nginx/                                    # Nginx 설정 파일 (타 팀원)
├── sql/                                      # DB 초기 설정 스크립트(타 팀원)
├── static/                                   # 정적 리소스 (타 팀원)
├── templates/                                # Django HTML 템플릿 (타 팀원)
└── web/                                      # Django 앱 (타 팀원)

```

---

## 회고

조례 파싱 작업에서 가장 오래 붙잡았던 건 데이터 추출 품질이었습니다. "LLM이 어느 정도 이해할 수 있으면 되지 않나?"라는 생각도 했지만, 뒤죽박죽된 데이터를 보면서 스스로 납득할 수 없어서 EasyOCR, Clova OCR, pdfplumber bbox까지 파고들었습니다. 

VectorDB에 정형 데이터를 넣는 결정, 자연어 맥락 생성, 멀티 지역 검색 설계 등 각 단계에서 "왜 이 방법인가"라는 근거를 세우려고 했습니다.

도구의 docstring만으로는 LLM이 의도대로 동작하지 않는 경우가 많았고, 시스템 프롬프트에 명시적 규칙을 추가하는 것이 실질적인 품질 개선으로 이어졌습니다.

또 하나 느낀 점은, OpenAI 임베딩 모델을 별 고민 없이 선택했다가 요청량이 많아지면서 시간 지연 문제가 생겼던 경험입니다. 설계 단계에서 병목이 될 수 있는 부분을 미리 검토하는 것, 그리고 문제가 생겼을 때 라이브러리 문서를 꼼꼼히 확인하는 습관의 중요성을 배웠습니다.
