# 🚀 PersonalizedCareerGuide

<details>
  <summary>📌 README (KOR) - 클릭하여 보기</summary>

## 📌 Service Overview

### 🔥 서비스 개발 배경
> **📢 바쁘다 바빠 취준생을 위한 AI Agent!**  

고학년이 되면서 취업 준비에 뛰어든 친구들이 많아졌는데요. 이에 따라 취업 준비에 실질적인 도움이 되는 서비스를 고민하게 되었습니다.  
친구들에게 **LLM을 활용한 취업 준비의 불편함**을 조사한 결과, 다음과 같은 페인 포인트를 발견하였습니다.  

- 원하는 결과를 얻기 위해 **직접 훈련시키는 과정이 번거로움**
- **프롬프트 작성 방법을 모름**
- **트렌드 반영 여부에 대한 의문**
- **개인화된 답변 부족**

이러한 문제를 해결하기 위해 다음과 같은 기능을 갖춘 서비스를 기획하였습니다.  

1. 직무를 입력하면 해당 직무의 **채용 공고 생성**
2. 채용 공고를 바탕으로 **필요한 역량 추출**
3. 해당 역량을 어필할 수 있도록 **자소서 글감 구성**
4. 활동 기반 면접과 지식 기반 면접을 모두 준비할 수 있도록 **예상 질문 생성**

이 프로세스를 **자동화하여**, 사용자가 **번거로운 과정 없이** 실질적인 도움을 받을 수 있도록 설계하였습니다.  

***

# 🔍 Methodology

## 🏆 01. Prompt Engineering
LLM을 활용하기 위해 효과적인 프롬프트를 설계하는 과정이 중요합니다.  
이에 따라 **Zero-shot, One-shot, Few-shot 프롬프팅 기법**을 실험하고 최적의 프롬프트를 선정하였습니다.

## 📚 02. RAG (Retrieval-Augmented Generation)
RAG를 활용하여 LLM이 **실제 채용 공고 데이터를 기반으로 보다 정확한 응답을 생성**하도록 설계하였습니다.  
RAG의 핵심 단계는 다음과 같습니다.

1️⃣ **Retrieval**: 사용자의 입력을 바탕으로 관련 정보를 검색  
2️⃣ **Augmentation**: 검색된 문서를 필터링하여 가장 관련성이 높은 데이터 제공  
3️⃣ **Generation**: LLM이 최적화된 결과를 생성  

이를 통해 **신뢰성 있는 정보를 기반으로 한 답변을 생성**할 수 있도록 구현하였습니다.

## 🔍 03. Query Expansion
**Query Expansion 기법**을 활용하여 검색 성능을 개선하였습니다.  
직무와 회사명을 입력하면 **가상의 채용 공고를 생성한 후, 이 공고 텍스트를 활용해 유사 공고를 검색**하는 방식으로 진행됩니다.  
이를 통해 검색의 **정확성과 확장성을 높였습니다.**  

***

# 🛠 Implementation

## 🔗 01. API 및 데이터
- **OpenAI API**: gpt-4o-mini 모델을 사용하여 프롬프트 기반의 답변을 생성  
- **딥서치 뉴스 API**: 최신 트렌드를 반영한 면접 질문을 생성하기 위해 국내 뉴스 기사 수집  
- **채용 공고 DB**: 네이버, 토스, 카카오 등의 채용 페이지를 크롤링하여 421건의 채용 공고 데이터를 확보  
- **자기소개서 DB**: 다양한 자기소개서 데이터를 수집하여 5개 카테고리로 분류  
  - [동기/포부], [성장/가치관], [역량/경험], [협업/성과], [기업/아이디어]

## 📑 02. 공고 생성 및 유사도 계산
- 채용 공고가 기존 DB에 존재하지 않는 경우, gpt-4o-mini 모델을 활용하여 **가상의 채용 공고 생성**
- 이때 가상의 채용 공고를 생성하는 이유는 사용자가 입력한 직무 + 회사명만으로는 RAG를 위한 유의미한 검색이 불가능하기 때문
- 기존 DB의 공고와 **코사인 유사도 계산**을 통해 관련성이 높은 공고 선별  

## 🎯 03. 필요 역량 파악
- 생성된 공고와 유사 공고 데이터를 기반으로 **기술적 역량과 비기술적 역량 도출**  

## 📝 04. 자기소개서 글감 구성
- 사용자의 활동과 직무 필요 역량을 기반으로 **자기소개서 글감 생성**  
- RAG 방식을 활용하여 **데이터 기반의 맞춤형 내용 제공**  

## 🎤 05. 예상 면접 질문 생성
### ✅ 활동 기반 면접 질문
- 사용자의 **자기소개서 글감을 기반으로 예상 질문 생성**  

### ✅ 지식 기반 면접 질문
- **뉴스 트렌드 질문**: 최신 뉴스 수집 후 관련 면접 질문 생성  
- **단순 기술 질문**: 직무 관련 기초 지식을 평가하는 질문 생성  

---

# 🎯 Result

## 📌 결과 예시
사용자의 직무와 회사 정보를 입력하면, 다음과 같은 출력이 제공됩니다.

✅ **채용 공고 생성**: 직무와 회사명을 기반으로 맞춤형 공고 생성  
✅ **필요 역량 도출**: 직무 수행에 필요한 기술적/비기술적 역량 정리  
✅ **자기소개서 글감 구성**: 사용자의 활동 이력을 바탕으로 맞춤형 글감 및 개요 제공  
✅ **면접 질문 생성**: 활동 기반 및 지식 기반 면접 질문 제공  

## 🌟 의의
이 서비스는 기존의 LLM 활용 방식에서 벗어나,  
**자동화된 AI Agent의 개념을 적용하여 사용자 친화적인 경험을 제공합니다.**  

- **RAG 기반 검색**, **Query Expansion**, **프롬프트 엔지니어링** 등의 기법을 활용  
- **신뢰성 있는 데이터를 바탕으로 개인화된 정보 제공**  
- **빠르고 비용 효율적인 방식으로 운영되어 사용자 편의성 극대화**  

## ⚠ 한계
1️⃣ **데이터 품질 의존성**  
   - 입력된 데이터가 부정확할 경우, 생성된 결과물의 품질이 저하될 가능성이 있음  

2️⃣ **최신 트렌드 반영의 어려움**  
   - 실시간 데이터 업데이트가 제한적일 경우 최신성이 유지되지 않을 수 있음  

3️⃣ **사용자 피드백 기반 개선 필요**  
   - 자기소개서 품질 및 면접 질문의 유효성을 지속적으로 평가하고 개선해야 함  


## 🎯 결론
이 서비스는 **취업 준비 과정에서 발생하는 불편함을 최소화**하고,  
보다 **효율적이고 정확한 정보를 제공**함으로써 **취준생들에게 실질적인 도움**을 주는 것을 목표로 합니다. 🚀  


</details>

<details>
  <summary>📌 README (ENG) - Click to View</summary>

## 📌 Service Overview

### 🔥 Background of Service Development
> **📢 An AI Agent for Busy Job Seekers!**  

As students reach their senior years, many start preparing for employment.  
We wanted to develop a service that provides **practical assistance in job preparation**.  
Through research on **the difficulties of using LLMs for job preparation**, we identified the following pain points:  

- **The process of training the model to get desired results is cumbersome**
- **Users lack knowledge on how to craft effective prompts**
- **Uncertainty about whether responses reflect industry trends**
- **Lack of personalized answers**

To address these issues, we designed a service with the following features:  

1. **Generate job postings** based on user-input job roles  
2. **Extract required competencies** from job postings  
3. **Compose resume (CV) content** to highlight key competencies  
4. **Generate expected interview questions** for both behavioral and technical interviews  

This process is **fully automated** to minimize complexity and maximize practical support for job seekers.  

---

# 🔍 Methodology

## 🏆 01. Prompt Engineering
To effectively use LLMs, designing optimal prompts is crucial.  
We experimented with **Zero-shot, One-shot, and Few-shot prompting techniques** to determine the best approach.

## 📚 02. RAG (Retrieval-Augmented Generation)
We implemented RAG to allow the LLM to **generate responses based on real job postings** for higher accuracy.  
RAG consists of three key steps:

1️⃣ **Retrieval**: Search for relevant information based on user input  
2️⃣ **Augmentation**: Filter and provide the most relevant data  
3️⃣ **Generation**: Produce optimized results with the LLM  

This ensures that the system **delivers reliable, data-driven responses**.

## 🔍 03. Query Expansion
We applied **Query Expansion techniques** to improve search performance.  
When users enter a job role and company name, the system **creates a synthetic job posting**,  
which is then used to search for similar real postings.  
This enhances both **search accuracy and scalability**.  

---

# 🛠 Implementation

## 🔗 01. API & Data Sources
- **OpenAI API**: Used gpt-4o-mini to generate responses based on prompts  
- **DeepSearch News API**: Collected domestic news articles to generate interview questions reflecting industry trends  
- **Job Posting Database**: Crawled **421 job postings** from Naver, Toss, Kakao, and other hiring platforms  
- **Resume (CV) Database**: Collected various personal statements categorized into:  
  - [Motivation/Goals], [Growth/Values], [Skills/Experience], [Collaboration/Performance], [Company/Ideas]

## 📑 02. Job Posting Generation & Similarity Calculation
- If no matching job posting exists in the database, gpt-4o-mini **generates a synthetic job posting**
- This is necessary because **a job title + company name alone are insufficient for meaningful RAG-based searches**
- **Cosine similarity calculations** are used to identify and rank the most relevant job postings  

## 🎯 03. Identifying Required Skills
- Extracting **technical and non-technical competencies** from both real and synthetic job postings  

## 📝 04. Resume (CV) Content Generation
- Generating personalized resume content based on **user experience and required competencies**  
- Using RAG techniques to **deliver tailored, data-driven content**  

## 🎤 05. Generating Expected Interview Questions
### ✅ Behavioral-Based Questions
- Generating expected questions based on **resume content**  

### ✅ Knowledge-Based Questions
- **Industry Trend Questions**: Generated by analyzing the latest news  
- **Basic Technical Questions**: Evaluating foundational knowledge relevant to the job  

---

# 🎯 Result

## 📌 Example Output
When users enter their **desired job role and company**, the system provides:  

✅ **Job Posting Generation**: Creates a customized job posting based on input  
✅ **Required Skills Extraction**: Identifies key technical and non-technical competencies  
✅ **Resume Content Generation**: Generates structured writing prompts based on user experience  
✅ **Interview Question Generation**: Provides both behavioral and knowledge-based interview questions  

## 🌟 Significance
This service moves beyond traditional LLM applications by integrating  
**automated AI Agent concepts to provide a user-friendly experience**.  

- Implements **RAG-based search, Query Expansion, and Prompt Engineering**  
- Delivers **personalized information based on reliable data sources**  
- Ensures **fast and cost-effective operations for maximum user convenience**  

## ⚠ Limitations
1️⃣ **Dependence on Data Quality**  
   - If input data is inaccurate, the quality of generated results may be affected  

2️⃣ **Challenges in Reflecting Real-Time Trends**  
   - If real-time data updates are limited, maintaining up-to-date content may be difficult  

3️⃣ **Need for Continuous User Feedback & Improvement**  
   - Ongoing evaluation and refinement of resume quality and interview question relevance  

---

## 🎯 Conclusion
This service **minimizes pain points in job preparation**,  
delivering **efficient and precise information** to provide **practical assistance for job seekers**. 🚀  


</details>
