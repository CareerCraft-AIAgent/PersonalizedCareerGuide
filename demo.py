import os
import pickle
import numpy as np
import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
from kiwipiepy import Kiwi
import re
from kiwipiepy.utils import Stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = ""
client = openai.OpenAI() 

db = pd.read_excel("processed_final_with_summaries.xlsx") # 기존 공고 요약 DB


# DB에서 관련 데이터 검색 함수
def search_db(user_full_job, db):
    """
    DB에서 user_full_job을 기반으로 관련 데이터를 검색합니다.
    """
    filtered_data = db[
        db['job'].str.contains(user_full_job, na=False)  # user_full_job을 'job' 열에서 검색
    ]
    if not filtered_data.empty:
        return filtered_data.iloc[0]  # 가장 관련성 높은 데이터 반환
    else:
        return None


#############################################################
#### 채용 공고 생성 

def generate_job_posting(job, company):
    """
    1. 사용자 입력(직무, 회사)에 기반한 '채용 공고 요약' 프롬프트 생성
    2. client.chat.completions.create()로 GPT API 호출 → 채용 공고(초안) 생성
    """

    # 기존 DB에서 검색
    db_data = search_db(user_full_job,db)
    
    if db_data is not None:
        # DB에서 검색된 데이터 활용
        org_summary = db_data['org_sum']
        work_summary = db_data['work_sum']
        skills_summary = db_data['skills_sum']
        
        # 프롬프트 생성 (기존 데이터 활용)
        prompt = (
            f"""
            아래는 {company}의 {job} 직무와 관련된 기존 데이터입니다:
            
            1. 조직 설명: {org_summary}
            2. 직무 설명: {work_summary}
            3. 필요 역량: {skills_summary}
            
            위 내용을 바탕으로 더 완성도 높은 채용 공고를 작성해주세요.
            부족한 내용은 보완하고, 전문적이고 직무에 적합한 문구를 사용하여 작성해주세요.
            
            **출력 형식**:
            1. 조직 설명: [수정 또는 보완된 조직 설명]
            2. 직무 설명: [수정 또는 보완된 직무 설명]
            3. 필요 역량: [수정 또는 보완된 필요 역량]
            """
        )
    else:
        # DB에서 검색된 데이터가 없을 경우 기본 프롬프트
        prompt = (
        f"""
        아래의 직무와 회사 정보를 기반으로 채용 공고 내용을 요약하세요. 아래는 참고 예제입니다:

        [예제1]
        직무: 쇼콜라티에
				회사: 신라호텔
				조직 설명: 신라호텔의 디저트 제작 팀은 고급 디저트 문화를 선도하며, 고객에게 독창적이고 정교한 초콜릿 경험을 제공하는 것을 목표로 합니다. 
								 이 팀은 초콜릿 조각과 공예를 통해 호텔의 품격을 반영하는 작품을 제작하며, 맞춤형 디저트와 특별한 행사를 위한 창의적인 초콜릿 아트를 선보이고 있습니다.

				직무 설명: 쇼콜라티에는 신라호텔의 고급 초콜릿 디저트와 예술 작품을 기획하고 제작하는 역할을 담당합니다. 주요 업무에는 고객 맞춤형 초콜릿 디자인, 대규모 행사 및 웨딩 초콜릿 세트 제작, 계절별 신제품 개발, 
								 그리고 초콜릿 품질 관리와 디스플레이 준비가 포함됩니다. 또한, 호텔 브랜드를 대표할 수 있는 초콜릿 작품 개발과 국제 초콜릿 공모전 출품을 위한 창의적 디자인도 포함됩니다.

				필요 역량: 초콜릿 공예 및 디저트 제작 분야에서 풍부한 경험과 전문성을 갖춘 지원자를 찾습니다. 초콜릿 작업에 필요한 재료 특성에 대한 깊은 이해와 이를 바탕으로 한 품질 관리 능력이 요구되며, 
								 창의적이고 독창적인 초콜릿 디자인을 개발할 수 있는 역량이 필요합니다. 디저트 및 초콜릿 관련 공모전에서 수상한 경력이나 이를 증명할 수 있는 포트폴리오를 보유한 지원자를 우대하며, 
								 대규모 행사 및 고객 맞춤형 초콜릿 프로젝트 경험을 가진 지원자는 더욱 환영합니다.

		[예제2]
        직무: AIML쇼핑검색개인화기술연구개발
        회사: 네이버쇼핑
        조직 설명: 해당 조직은 개인화된 쇼핑 검색 경험을 구현하는 팀으로, 사용자의 취향과 이력을 기반으로 한 맞춤형 검색 결과를 제공하는 것을 목표로 합니다. 
                이 팀은 브랜드 선호도, 가격, 스타일, 구매 패턴 등 다양한 신호를 분석하여 사용자에게 더 효율적이고 탐색적인 쇼핑 경험을 제공하는 데 중점을 두고 있습니다.
        직무 설명: 직무는 사용자 취향 및 의도를 반영한 개인화 검색 추천 모델을 설계, 개발 및 고도화하는 것입니다. 
                주요 업무에는 대규모 로그 분석을 통한 특성 추출, 모델 피처 엔지니어링, AB 테스트를 통한 지표 모니터링 및 모델 성능 평가가 포함됩니다. 또한, 추천 시스템과 관련된 연구 개발 및 AIML 기반 추천 모델의 서비스 적용도 포함됩니다.
        필요 역량: 직무를 성공적으로 수행하기 위해서는 AIML 기반 추천 모델 개발 및 서비스 경험이 3년 이상 필요하며, 
                사용자 분석, 콘텐츠 이해, 모델링 로직 설계 등 문제 정의 및 해결 능력이 요구됩니다. 
                또한, LLM 최신 기술 및 NLP, RecSys 관련 기술 활용 경험이 필요하며, 
                검색 추천 관련 학회에 논문을 게재하거나 오픈소스에 기여한 경험이 있는 것이 바람직합니다.

        **입력**:
        직무 이름: {job}
        회사 이름: {company}

        **출력 형식**:
        1. 조직 설명: [해당 회사 및 직무가 포함된 팀이나 조직의 역할, 목표 및 성격을 명확히 서술]
        2. 직무 설명: [입력된 직무와 관련된 주요 업무, 책임, 그리고 기대되는 활동을 구체적으로 서술]
        3. 필요 역량: [직무를 성공적으로 수행하기 위해 요구되는 기술적(예: 특정 소프트웨어나 툴 사용 능력) 및 비기술적 역량(예: 소통 능력, 문제 해결 능력 등)을 상세히 서술]
        """
    )
    

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0
        )
        generated_posting_text = response.choices[0].message.content.strip()
        return generated_posting_text
    except Exception as e:
        return f"API 호출 중 오류가 발생했습니다: {str(e)}"



###############################################################################
# 2. 생성된 채용 공고(텍스트)를 Upstage solar 임베딩 모델로 임베딩한다.
###############################################################################
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI  # Requires openai==1.52.2

# Upstage solar 모델을 사용하기 위한 API 불러오기
client2 = OpenAI(
    api_key="",  # 본인의 OpenAI API 키 입력
    base_url="https://api.upstage.ai/v1/solar"
)


# 텍스트 임베딩 생성 함수
def get_embedding(text):
    if not text.strip():  # 빈 문자열 처리
        raise ValueError("Input text is empty.")
    
    response = client2.embeddings.create(input=text, model="embedding-query")
    return response.data[0].embedding

###############################################################################
# 3. 기존 DB(embeddings.pkl)에 저장된 각 공고의 임베딩과 (3)번에서 구한 임베딩 간
#    유사도를 코사인 유사도 기반으로 계산한다.
###############################################################################
def retrieval(result, top_n=3):

    user_org, user_work, user_skills = "", "", ""
    for line in result.split("\n"):
        if "조직 설명" in line:
            user_org = line.split(":", 1)[-1].strip()
        elif "직무 설명" in line:
            user_work = line.split(":", 1)[-1].strip()
        elif "필요 역량" in line:
            user_skills = line.split(":", 1)[-1].strip()

    # 기존 임베딩 로드
    with open('embeddings.pkl', 'rb') as f:
        data = pickle.load(f)

    # 유저 입력 텍스트 임베딩
    org_emb = np.array(get_embedding(user_org)).reshape(1, -1)
    work_emb = np.array(get_embedding(user_work)).reshape(1, -1)
    skills_emb = np.array(get_embedding(user_skills)).reshape(1, -1)

    # 열별 코사인 유사도 계산
    cosim_org = cosine_similarity(org_emb, data["org_sum"])[0]       # shape: (num_jobs,)
    cosim_work = cosine_similarity(work_emb, data["work_sum"])[0]    # shape: (num_jobs,)
    cosim_skills = cosine_similarity(skills_emb, data["skills_sum"])[0]  # shape: (num_jobs,)


    org_top_indices = np.argsort(-cosim_org)[:top_n]
    work_top_indices = np.argsort(-cosim_work)[:top_n]
    skills_top_indices = np.argsort(-cosim_skills)[:top_n]

    candidates_indices = list(set(org_top_indices) | set(work_top_indices) | set(skills_top_indices))

    weights = {"org": 0.2, "work": 0.4, "skills": 0.4}
    final_scores = []
    for idx in candidates_indices:
        score = (
            weights['org'] * cosim_org[idx]
            + weights['work'] * cosim_work[idx]
            + weights['skills'] * cosim_skills[idx]
        )
        final_scores.append((idx, score))
    
    final_scores_sorted = sorted(final_scores, key=lambda x: x[1], reverse=True)
    rerank_top = final_scores_sorted[:top_n]
    rerank_idx = [x[0] for x in rerank_top]

    #return top_indices
    return rerank_idx


###############################################################################
# 5. 1번에서 생성된 공고와 4번에서 추출한 기존 공고들을 함께 활용하여,
#    최종적으로 필요한 역량(기술적 3개, 비기술적 3개)을 다시 GPT 모델에게 요청한다.
###############################################################################
def get_required_skills(job, job_posting, similar_sum):
    
    prompt = (
    f"""
    아래는 새로 생성된 '{job}'에 관한 채용 공고와 기존 유사 공고입니다. 
    이 내용을 바탕으로 직무를 성공적으로 수행하기 위해 필요한 역량(기술적 3개, 비기술적 3개)을 구체적으로 도출하세요.

    [새로 생성된 공고]
    '{job_posting}'

    [기존 유사 공고]
    1. '{similar_sum[0]}'
    2. '{similar_sum[1]}'
    3. '{similar_sum[2]}'

    **출력 형식**
    1. 기술적 역량:
    - [역량1: 설명]
    - [역량2: 설명]
    - [역량3: 설명]

    2. 비기술적 역량:
    - [역량1: 설명]
    - [역량2: 설명]
    - [역량3: 설명]
    """
)
    try:
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0
        )
        user_skills = final_response.choices[0].message.content.strip()
        return user_skills
        
    except Exception as e:
        return f"API 호출 중 오류가 발생했습니다: {str(e)}"
    


def generate_personal_statement(job, selected_categories, activities, skills):
    """
    사용자가 선택한 항목과 입력한 활동, 그리고 RAG에서 추출된 필요 역량을 기반으로 글감과 개요를 생성하는 함수.
    """
    responses = {}
    
    for category, activity in zip(selected_categories, activities):
        
        prompt = f"""
        당신은 자기소개서 글감 및 개요 추출을 전문으로 하는 AI 조력자입니다.
        사용자가 제공한 입력 정보를 바탕으로, 아래 지침에 따라 자기소개서 글감과 개요를 생성하세요.

        1. **직무 정보**:
           - 직무 이름: {job}
           - 이 직무와 관련된 필요 역량은 다음과 같습니다: {skills}

        2. **사용자 활동 정보**:
           - **{category}**: {activity}

        3. **요청 사항**:
           - {skills} 중 해당 활동을 통해 강조할 수 있는 역량을 하나만 골라 글감과 개요를 작성하세요.
           - 글감: 입력한 활동에서 도출된 주요 주제를 간결히 표현하세요.
           - 개요: 글감에서 도출된 주제를 구체적으로 확장하여 자기소개서에서 활용 가능한 세부 내용을 포함하세요.
             - **배경 설명**: [활동의 배경 및 맥락]
             - **성과/결과**: [활동의 결과나 성취]
             - **직무/기업과의 연결**: [직무나 기업의 요구 사항에 대한 연관성]

        4. **출력 형식**:
           - **강조 역량**: [강조할 역량]
             **글감**: [사용자의 활동에서 추출된 주요 주제]
             **개요**:
               - **배경 설명**: [활동의 배경 및 맥락]
               - **성과/결과**: [활동의 결과나 성취]
               - **직무/기업과의 연결**: [직무나 기업의 요구 사항에 대한 연관성]
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3
            )
            responses[category] = response.choices[0].message.content.strip()
        except Exception as e:
            responses[category] = f"API 호출 중 오류가 발생했습니다: {str(e)}"

    return responses



def generate_q1(job, personal_statement):
    
    prompt = f"""
당신은 활동 기반 면접 질문을 생성하는 전문가입니다. 
주어진 텍스트는 지원자가 입력한 활동을 기반으로 필요한 역량을 강조할 수 있도록 작성된 자기소개서 글감 개요입니다. 
이 텍스트를 바탕으로 활동 기반 면접 질문 3개를 생성하세요. 
각 질문은 {job} 직무에 지원하는 지원자의 경험과 역량을 효과적으로 평가할 수 있어야 합니다. 
다음 지침을 따르세요:

1. 첫 번째 질문은 텍스트에서 언급된 주요 활동이나 경험의 구체적인 내용을 탐구하는 질문이어야 합니다.
2. 두 번째 질문은 지원자의 문제 해결 능력, 협업 경험, 또는 의사결정 과정을 평가할 수 있는 질문이어야 합니다.
3. 세 번째 질문은 지원자의 해당 경험에서 얻은 교훈이나 배운 점을 직무와 연결 지을 수 있도록 구성해야 합니다.

입력 텍스트:
"{personal_statement}"

출력 (활동 기반 면접 질문):
1. [주요 활동이나 경험을 탐구하는 질문]
2. [문제 해결, 협업, 의사결정과 관련된 질문]
3. [경험과 직무 관련 역량을 연결하는 질문]
"""
    
    # OpenAI API 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0
        )
        # 응답 내용 추출
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API 호출 중 오류가 발생했습니다: {str(e)}"



###############################################################################
# 뉴스 키워드 뽑는 함수
###############################################################################

def generate_news_keyword(job, company):
    prompt = f"""
당신은 트렌드 관련 키워드 생성 전문가입니다. 어떤 직무와 회사가 주어지더라도, 관련된 2025년 최신 트렌드, 기술, 혹은 사례를 조사할 때 유용한 **구체적이고 세부적인 키워드** 3개를 작성하세요.
'{company}'의 '{job}' 직무에 관련된 트렌드 정보 및 배경 지식을 조사할 때 유용하게 활용할 수 있는 **구체적이고 세부적인 키워드** 3개를 도출하세요.

키워드는 다음 조건을 충족해야 합니다:
1. '{job}' 직무와 밀접하게 관련된 2025년 최신 트렌드, 기술, 혹은 사례를 반영해야 합니다.
2. 뉴스 사이트나 검색 엔진에서 검색했을 때, **구체적이고 신뢰할 수 있는 정보를 바로 찾을 수 있는 형태**여야 합니다.
3. 키워드는 특정 활동, 기술, 혹은 트렌드처럼 **명확하고 세부적인 주제**를 포함해야 합니다.
4. 키워드는 **명사 + 명사** 형식으로 작성해야 하며, "~의" 같은 조사나 불필요한 접속사를 포함하지 않아야 합니다.
5. 입력된 직무나 회사의 **산업 도메인 특성**을 반영하여, 일반적이지 않고 관련성이 높은 키워드를 작성해야 합니다.

출력 형식:
- 키워드 1: [구체적인 기술/활동/트렌드]
- 키워드 2: [구체적인 기술/활동/트렌드]
- 키워드 3: [구체적인 기술/활동/트렌드]

"""
    # OpenAI API 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.8
        )
        # 응답 내용 추출
        content = response.choices[0].message.content.strip()

        # 키워드 리스트 생성
        keywords = []
        for line in content.split("\n"):
            if line.startswith("- 키워드"):
                keyword = line.split(":")[1].strip()
                keyword = keyword.strip('"').strip("'")
                keywords.append(keyword)

        return keywords
    except Exception as e:
        return f"API 호출 중 오류가 발생했습니다: {str(e)}"
    
# 뉴스 키워드 함수 호출 및 출력

###############################################################################
# 뉴스 검색하는 함수
###############################################################################
import requests

def search_news_by_keyword(keyword):

    # API URL 및 파라미터
    url = "https://api-v2.deepsearch.com/v1/articles"
    params = {
        "keyword": keyword,
        # "date_from": "2024-12-01",
        "page_size": 30,
        "api_key": ""
    }

    # API 요청
    response = requests.get(url, params=params)

    # 응답 처리
    if response.status_code == 200:
        # JSON 데이터 파싱
        data = response.json()

        # 데이터에서 필요한 항목만 추출
        articles = data.get("data", [])
        extracted_data = []
        for article in articles:
            extracted_data.append({
                "Title": article.get("title", "N/A"),
                "Date": article.get("published_at", "N/A").split("T")[0] if article.get("published_at") else "N/A",
                "Section": ", ".join(article.get("sections", ["N/A"])),
                "Publisher": article.get("publisher", "N/A"),
                "Summary": article.get("summary", "N/A").replace("\n", " "),
                "Content URL": article.get("content_url", "N/A")
            })

        # DataFrame 생성
        news_df = pd.DataFrame(extracted_data)
        return news_df
    else:
        print(f"API 호출 실패: 상태 코드 {response.status_code}")
        print(response.text)
        return pd.DataFrame()  # 빈 DataFrame 반환
    

# Kiwi 초기화
kiwi = Kiwi()
stopwords_dict = Stopwords()

# 데이터 전처리 함수 정의
def Kr_preprocessing(text):
    text = text.strip()
    text = re.sub(r'[^\d\s\w]', ' ', text)
    kiwi_tokens = kiwi.tokenize(text, stopwords=stopwords_dict)
    noun_words = [token.form for token in kiwi_tokens if 'NN' in token.tag and len(token.form) > 1]
    return ' '.join(noun_words)

def cluster_news(news):

    # 뉴스 요약 데이터를 리스트로 변환
    total_docs = []
    for i in range(len(news)):
        total_docs.append(news.loc[i, 'Summary'])

    # 전처리 수행
    filtered_docs = [Kr_preprocessing(doc) for doc in total_docs]

    # TF-IDF 기반 DTM 생성
    tfidf_vectorizer = TfidfVectorizer()
    DTM_tfidf = tfidf_vectorizer.fit_transform(filtered_docs)
    DTM_TFIDF = np.array(DTM_tfidf.todense())

    # PCA 수행 (8개의 주성분)
    pca = PCA(n_components=8)
    pca_results_tfidf = pca.fit_transform(DTM_TFIDF)
    
    # 실루엣 스코어 기반 최적 클러스터 수 선택
    best_n_clusters = 0
    best_score = -1
    scores = []  # 모든 n_clusters에 대한 실루엣 스코어 저장

    # n_clusters 범위를 지정 (2부터 len(filtered_docs)까지)
    for i in range(2, len(filtered_docs)):
        kmeans = KMeans(n_clusters=i, random_state=42)
        cluster_pca_ifidf = kmeans.fit_predict(pca_results_tfidf)

        # 현재 n_clusters에 대한 실루엣 스코어 계산
        score = silhouette_score(pca_results_tfidf, cluster_pca_ifidf)
        scores.append(score)

        # 최고 실루엣 스코어와 해당 n_clusters 업데이트
        if score > best_score:
            best_score = score
            best_n_clusters = i

    # 최적의 클러스터 수로 KMeans 실행
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    final_clusters = kmeans.fit_predict(pca_results_tfidf)

    news['cluster_id'] = [cluster_id + 1 for cluster_id in final_clusters]
    news = news.sort_values(by='cluster_id').reset_index(drop=True)

    cluster_dict = {
        cluster_id: cluster_df.reset_index(drop=True)
        for cluster_id, cluster_df in news.groupby('cluster_id')
    }

    return cluster_dict


def create_cluster_text(clustered_news):
    cluster_summaries = {}

    for key, value in clustered_news.items():
        summaries = value['Summary'].tolist()
        cluster_summaries[key] = summaries

    # 클러스터별 요약 내용을 하나의 문자열로 합치기
    output_string = []

    for cluster_id, summaries in cluster_summaries.items():
        output_string.append(f"Cluster {cluster_id}")
        output_string.extend(summaries)
        output_string.append("")

    return "\n".join(output_string)


# 뉴스 선정 함수

def summarize_cluster(text, job, company, keyword):
    prompt = f"""
당신은 직무 관련 트렌드 전문가입니다. 
아래 text는 {company} 회사의 {job} 직무에 지원하는 지원자가 면접 준비 과정에서 {keyword}를 검색어로 뉴스 기사를 검색한 결과입니다.
여러 클러스터로 분류되어 있는데, 이때 각 클러스터는 특정 주제를 중심으로 구성된 기사의 요약으로 이루어져 있습니다. 
아래 텍스트를 바탕으로 각 클러스터의 **주제**를 도출하고, 클러스터 내 문장들을 분석하여 **적당한 길이의 핵심 요약**을 작성하세요.

출력 시 가장 중요한 클러스터 top 3만 출력해야 합니다. 정렬 기준은 다음과 같습니다.
1. (가장 중요) 지원자의 {job} 채용 준비 과정에서 **직무 트렌드를 깊이 이해하는 데 실질적인 도움을 주는가** (40%)
2. {keyword}와의 연관성이 높으며, 직무 수행에 있어 필수적인 키워드를 포함하고 있는가 (30%)
3. 주제가 특정 기업의 홍보에 치우치지 않고, 업계 전반의 흐름을 포괄하며 다양한 시각을 제공하는가 (30%)


예시로, 다음과 같이 클러스터를 선정해야 합니다.

‘디지털 금융의 발전과 핀테크 산업 성장’ (적절한 클러스터)
: 핀테크 및 디지털 금융 트렌드, 기업들의 투자 및 혁신 사례가 포함되어 있어 금융 분야의 변화 흐름을 이해하는 데 적합함.

‘글로벌 경기 둔화와 국내 금융 정책 대응’ (적절한 클러스터)
: 경제 환경 변화에 따른 정책적 대응 전략을 포함하고 있어 지원자가 경제 전반의 맥락을 파악하는 데 도움을 줄 수 있음.

‘은행별 새로운 서비스 출시 소식’ (부적절한 클러스터)
: 특정 은행의 개별 서비스 홍보에 초점이 맞춰져 있으며, 산업 전체의 흐름을 이해하는 데는 한계가 있음.


**출력 형식**:
최종 클러스터: a, b, c
1. [클러스터 주제]
: [클러스터 a의 요약문들을 바탕으로 생성된 핵심 요약]

2. [클러스터 주제]
: [클러스터 b의 요약문들을 바탕으로 생성된 핵심 요약]

3. [클러스터 주제]
: [클러스터 c의 요약문들을 바탕으로 생성된 핵심 요약]
...

**입력 텍스트**:
{text}

**출력 지침**:
- 클러스터 a, b, c에는 각각 해당 클러스터의 id가 들어갑니다.
- 각 클러스터의 주제는 해당 클러스터를 대표할 수 있는 단 하나의 문장으로 작성합니다.
- 핵심 요약은 클러스터에 포함된 문장들을 기반으로 간결하고 일관되게 작성하되, 중요한 정보를 빠뜨리지 않도록 주의하세요.
- 요약은 클러스터 내 문장들의 주요 내용을 종합한 형태로 작성하며, 지나치게 세부적이거나 불필요한 정보는 제외합니다.

...

**출력 예시**:
1. 국내 AI 창업자의 글로벌 영향력 및 성공 사례
: 한국 AI 창업자들이 글로벌 시장에서 두각을 나타내며 성공적인 투자 유치와 혁신적인 AI 솔루션을 제공하고 있음. 
포브스가 선정한 '주목할 AI 창업자'로 소개되며 글로벌 AI 산업에서의 입지를 강화함.

"""

    # OpenAI API 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0
        )
        # 응답 내용 추출
        content = response.choices[0].message.content.strip()
        
        return content
    except Exception as e:
        return f"API 호출 중 오류가 발생했습니다: {str(e)}"
    

def generate_q2(job, company, k_idx, topic, text):
    """
    요약된 뉴스 내용을 기반으로 질문 생성

    """
    # 프롬프트 작성
    prompt = (f"""
    아래 text는 {company} 회사의 {job} 직무에 지원하는 지원자가 면접 준비 과정에서 {news_keywords[k_idx-1]}를 검색어로 뉴스 기사를 검색한 후,
    자신이 원하는 주제의 기사만 골라서 요약한 결과입니다. 
    아래 뉴스 텍스트를 바탕으로 '{job}' 직무를 준비하는 지원자에게 도움이 될 만한 질문 5개를 생성하고, 각 질문을 평가한 뒤, Top 3개의 질문만 출력하세요.

    **입력 내용**:
    뉴스 주제: {topic}
    요약 내용: {text}
        
    **질문 생성 및 평가 지침**
    [질문 생성]
        - 도메인 관심도 평가: 기사 내용 및 관련 트렌드를 기반으로 지원자가 해당 도메인에 대한 관심과 이해도를 나타낼 수 있는 질문을 작성하세요.
        - 상황 기반 질문: 기사에서 다루는 특정 상황을 바탕으로 지원자의 사고력과 문제 해결 능력을 평가할 수 있는 질문을 작성하세요.
        - 토론 유도 질문: 기사에서 언급된 산업 동향, 경쟁 상황, 또는 소비자 행동 변화와 관련하여 지원자의 의견을 유도할 수 있는 질문을 작성하세요.
        - 창의적 사고 질문: 기사에서 다룬 주제를 확장하거나 새로운 아이디어를 제시할 수 있도록 창의적인 사고를 유도하는 질문을 작성하세요.

    [질문 평가]
      - 도메인 관련 지식 평가 (최대 5점): '{job}' 직무와 관련된 도메인의 {news_keywords[k_idx-1]} 지식을 평가할 수 있어야 함.
      - 직무 수행에 필요한 이해도 평가 (최대 5점): 직무 수행에 필요한 도메인 이해도를 평가할 수 있어야 함.
      - 질문의 적절성 및 범용성 (최대 3점): 특정 기업이나 기술에 치우치지 않으며 보편적으로 평가 가능해야 함. 지엽적일 경우 감점(-3점).
      - 창의적 사고 유도 (최대 3점): 창의적 사고를 유도하는 질문일 경우 높은 점수.
      - 지나치게 일반적인 질문 방지 (-2점): 지나치게 일반적인 질문은 감점 처리.

    

    **출력 형식**
    위의 질문 생성 및 평가 지침을 활용하여, 점수를 기반으로 상위 3개의 질문만 출력하세요.
    이때 질문만 출력하고, 점수 등 평가 내용은 출력하지 마세요.
       1. [질문 내용]
       2. [질문 내용]
       3. [질문 내용]
    """
)

    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0
        )
        
        # 응답에서 질문 추출
        generated_questions = response.choices[0].message.content.strip()
        questions = [q.strip() for q in generated_questions.split("\n") if q.strip()]
        
        # 4개 질문만 반환
        return questions
        #return generated_questions
    
    except Exception as e:
        return f"API 호출 중 오류가 발생했습니다: {str(e)}"
        


kiwi = Kiwi()
stopwords_dict = Stopwords()

def Kr_preprocessing2(text):
    custom_stopwords = ['토스', '서비스', '경험', '문제', '업무', '필요', '관련', '기술', 
                        '다양', '해결', '이해', '제품', '보유', '작성', '이상', '과정', '주도',
                        '활용', '중요', '능력', '조직', '제안']
    for word in custom_stopwords:
        stopwords_dict.add((word, 'NNG'))
    
    text = text.strip()
    text = re.sub(r'[^\d\s\w]', ' ', text)
    
    kiwi_tokens = kiwi.tokenize(text, stopwords=stopwords_dict)
    noun_words = [token.form for token in kiwi_tokens if 'NN' in token.tag and len(token.form) > 1]
    return noun_words


def extract_keywords(job_posting, max_keywords=10):
    
    processed_data = Kr_preprocessing2(job_posting)
    
    vectorizer = TfidfVectorizer(max_features=max_keywords)
    tfidf_matrix = vectorizer.fit_transform(processed_data)
    keywords = vectorizer.get_feature_names_out()

    return list(keywords)


def generate_q3(job, keywords, job_posting):
    """
    LLM을 이용하여 직무와 관련된 지식 중심의 질문을 생성합니다.
    """
    prompt = f"""
    당신은 직무 관련 면접 질문을 생성하는 전문가입니다.
    주어진 직무와 키워드를 바탕으로, 면접 질문에서 사용자의 특정 지식이나 기술 이해도를 평가할 수 있는 질문 3개를 생성하세요.
    질문은 다음 조건을 충족해야 합니다:
    1. 질문은 "{job}" 직무에서 필요한 구체적인 지식(예: 기술, 개념, 이론)에 대해 물어야 합니다.
    2. 질문 형식은 간단하며, 예를 들어 "{job}"와 관련된 특정 개념이나 기술에 대해 설명을 요청하는 형태여야 합니다.
    3. 채용 공고인 '{job_posting}'을 1순위로 참고하고, '{keywords}'는 2순위로 참고하세요.

    예시:
    직무: "데이터 분석가"
    추출된 키워드: "데이터, 분석, 문제 해결"
    출력 (면접 질문):
    1. 데이터 분석에서 'hierarchical clustering'이란 무엇인가요?
    2. 'k-means clustering'의 작동 방식을 설명해주세요.
    3. 데이터 분석 과정에서 'PCA(주성분 분석)'가 사용되는 이유는 무엇인가요?


    이제 아래 정보를 바탕으로 면접 질문 3개를 생성하세요.

    직무: "{job}"
    채용 공고: "{job_posting}"
    추출된 키워드: {keywords}

    출력 (면접 질문 3개):
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional interview question generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    # 응답에서 메시지 내용 추출
    content = response.choices[0].message.content
    return content




####################
#### streamlit code
####################

import streamlit as st

st.set_page_config(
    page_title="바쁘다 바빠 취준생을 위한 AI Agent",  # 페이지 제목
    page_icon="🗂️")          


# Step 1: 직무 및 회사 입력
st.header("1. 직무 및 회사 입력")
user_job = st.text_input("직무를 입력하세요:")
user_company = st.text_input("회사를 입력하세요:")
user_full_job = f"{user_company} {user_job}"

if st.button("채용 공고 생성"):
    if user_job and user_company:
        user_job_posting = generate_job_posting(user_job, user_company)
        st.session_state["job_posting"] = user_job_posting
        st.success("채용 공고가 생성되었습니다.")
        st.text_area("생성된 채용 공고", user_job_posting, height=300)
    else:
        st.error("직무와 회사를 모두 입력하세요.")

# Step 2: 필요 역량 추출
st.header("2. 필요 역량 추출")
if "job_posting" in st.session_state:
    similar_sum = db.loc[retrieval(st.session_state["job_posting"], 3), 'total_sum'].tolist()
    user_skills = get_required_skills(user_job, st.session_state["job_posting"], similar_sum)
    st.session_state["skills"] = user_skills
    st.success("필요 역량이 추출되었습니다")
    st.text_area("추출된 필요 역량", user_skills, height=500)

# Step 3: 자소서 글감 생성
st.header("3. 자기소개서 글감 생성")
categories = ["동기/포부", "성장/가치관", "역량/경험", "협업/성과", "기업/아이디어"]
selected_categories = st.multiselect("도움이 필요한 문항을 선택하세요", categories)

if selected_categories:
    activities = []
    for category in selected_categories:
        activity = st.text_area(f"'{category}'에 대한 활동을 입력하세요:", height=200)
        activities.append(activity)
    
    if st.button("자기소개서 글감 및 개요 생성"):
        if "job_posting" in st.session_state:
            job_posting = st.session_state["job_posting"]
            user_personal_statement = generate_personal_statement(user_job, selected_categories, activities, user_skills)
            st.session_state["personal_statement"] = user_personal_statement
            st.success("자기소개서 글감 및 개요가 생성되었습니다.")
            for category, content in user_personal_statement.items():
                st.subheader(category)
                st.write(content)
        else:
            st.error("먼저 채용 공고를 생성하세요.")

# Step 4: 면접 질문 생성
st.header("4. 면접 질문 생성")

if "personal_statement" in st.session_state:
    tab1, tab2, tab3 = st.tabs(["활동 기반 질문", "뉴스 트렌드 질문", "단순 기술 질문"])

    with tab1:
        st.subheader("활동 기반 질문")
        user_q1 = generate_q1(user_job, activities)
        st.write(user_q1)

    with tab2:
        st.subheader("뉴스 트렌드 질문")

        # 세션 상태에서 기존 뉴스 키워드를 확인하고, 없으면 생성
        if "news_keywords" not in st.session_state:
            st.session_state["news_keywords"] = generate_news_keyword(user_job, user_company)

        news_keywords = st.session_state["news_keywords"]

        # 세션 상태에서 뉴스 데이터 확인하고, 없으면 검색
        if "news_data" not in st.session_state:
            dataframes = {}
            for idx, keyword in enumerate(news_keywords, start=1):
                dataframe_name = f"news{idx}"
                dataframes[dataframe_name] = search_news_by_keyword(keyword)

            # 뉴스 데이터 저장
            st.session_state["news_data"] = {
                "news1": dataframes['news1'],
                "news2": dataframes['news2'],
                "news3": dataframes['news3']
            }

        # 세션에서 뉴스 데이터 로드
        news1 = st.session_state["news_data"]["news1"]
        news2 = st.session_state["news_data"]["news2"]
        news3 = st.session_state["news_data"]["news3"]

        # 뉴스 클러스터 및 텍스트 요약
        if "news_summaries" not in st.session_state:
            st.session_state["news_summaries"] = {
                "news_dict1": cluster_news(news1),
                "news_dict2": cluster_news(news2),
                "news_dict3": cluster_news(news3),
                "cluster_text1": create_cluster_text(cluster_news(news1)),
                "cluster_text2": create_cluster_text(cluster_news(news2)),
                "cluster_text3": create_cluster_text(cluster_news(news3))
            }

        cluster_text1 = st.session_state["news_summaries"]["cluster_text1"]
        cluster_text2 = st.session_state["news_summaries"]["cluster_text2"]
        cluster_text3 = st.session_state["news_summaries"]["cluster_text3"]

        # 뉴스 트렌드 요약 데이터 캐싱
        if "trends" not in st.session_state:
            st.session_state["trends"] = {
                "trend1": summarize_cluster(cluster_text1, user_job, user_company, news_keywords[0]).split('\n\n'),
                "trend2": summarize_cluster(cluster_text2, user_job, user_company, news_keywords[1]).split('\n\n'),
                "trend3": summarize_cluster(cluster_text3, user_job, user_company, news_keywords[2]).split('\n\n')
            }

        trend1 = st.session_state["trends"]["trend1"]
        trend2 = st.session_state["trends"]["trend2"]
        trend3 = st.session_state["trends"]["trend3"]

        # trend_idx 세션에서 불러오기
        if "trend_idx" not in st.session_state:
            st.session_state["trend_idx"] = {
                1: list(map(int, re.findall(r'\d+', trend1[0]))),
                2: list(map(int, re.findall(r'\d+', trend2[0]))),
                3: list(map(int, re.findall(r'\d+', trend3[0])))
            }

        trend_idx1 = st.session_state["trend_idx"][1]
        trend_idx2 = st.session_state["trend_idx"][2]
        trend_idx3 = st.session_state["trend_idx"][3]

        def view_topic(keyword_idx, topic_idx):
            news_dict = {
                1: st.session_state["news_summaries"]["news_dict1"],
                2: st.session_state["news_summaries"]["news_dict2"],
                3: st.session_state["news_summaries"]["news_dict3"]
            }

            cluster_text = {
                1: cluster_text1,
                2: cluster_text2,
                3: cluster_text3
            }

            trend = {
                1: trend1,
                2: trend2,
                3: trend3
            }

            trend_idx = {
                1: trend_idx1,
                2: trend_idx2,
                3: trend_idx3
            }

            cluster_idx = trend_idx[keyword_idx][topic_idx-1]

            # 검색에 쓸 full text
            full_text = cluster_text[keyword_idx].split("\n\n")[cluster_idx-1][10:]

            df = news_dict[keyword_idx][cluster_idx]
            titles = df['Title'].tolist()
            urls = df['Content URL'].tolist()

            topic = trend[keyword_idx][topic_idx].split('\n')[0][3:]

            # 제목과 URL을 한 줄씩 표시하는 문자열 생성
            articles = "\n".join([f"{titles[i]} {urls[i]}" for i in range(len(df))])

            return topic, full_text, articles

        # 토픽별 체크박스 선택 처리
        selected_topic = st.session_state.get("selected_topic", None)

        for i, keyword in enumerate(news_keywords):
            st.write(f"**키워드: {keyword}**")
            trends = [trend1, trend2, trend3][i]

            for idx, topic in enumerate(trends[1:], 1):
                topic_lines = topic.split("\n")
                topic_title = topic_lines[0][3:].strip()  # 제목 추출 및 공백 제거
                topic_summary = topic_lines[1].strip() if len(topic_lines) > 1 else "요약 없음"

                col1, col2 = st.columns([5, 1])
                with col1:
                    st.write(f"{idx}. {topic_title}")
                    st.write(f"{topic_summary}")

                with col2:
                    checkbox = st.checkbox("", key=f"{keyword}_{idx}", value=(selected_topic == topic_title))
                    if checkbox:
                        st.session_state["selected_topic"] = topic_title

            st.write("---")

        # 선택한 토픽이 있을 경우 기사 및 질문 표시
        if "selected_topic" in st.session_state:
            selected_topic = st.session_state["selected_topic"]

            st.success(f"선택된 토픽: {selected_topic.strip()}")

            # 모든 트렌드 리스트에서 검색
            all_trends = [trend1[1:], trend2[1:], trend3[1:]]
            t_idx = None
            k_idx = None

            for idx, trend_list in enumerate(all_trends, 1):
                try:
                    t_idx = next(i for i, t in enumerate(trend_list) if selected_topic.strip() in t.strip()) + 1
                    k_idx = idx
                    break
                except StopIteration:
                    continue
            
            if k_idx and t_idx:
                selected_topic, full_text, related_articles = view_topic(k_idx, t_idx)
                st.text_area("관련 기사", related_articles, height=200)

                user_q2 = generate_q2(user_job, user_company, k_idx, selected_topic, full_text)
                for idx, question in enumerate(user_q2, 1):
                    st.write(f"{question}")

    with tab3:
        st.subheader("단순 기술 질문")
        s_keywords = extract_keywords(st.session_state["job_posting"], max_keywords=10)
        user_q3 = generate_q3(user_job, s_keywords, st.session_state["job_posting"])
        st.write(user_q3)
