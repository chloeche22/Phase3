import os
import shutil
import logging
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import PersistentClient
from langchain_chroma import Chroma

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# .env 파일에서 환경 변수 불러오기
load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")

# 1. PDF 파일 목록 및 중요도 설정
pdf_files_with_importance = [
    ("/mnt/c/Users/kec91/Desktop/capstone_git/refer/근로기준법(법률)(제18176호)(20211119).pdf", "high"),
    ("/mnt/c/Users/kec91/Desktop/capstone_git/refer/2023년 표준 취업규칙.pdf", "medium"),
    ("/mnt/c/Users/kec91/Desktop/capstone_git/refer/공단 취업규칙.pdf", "low")
]

# 2. 중요도에 따른 가중치 설정
importance_weights = {"high": 5, "medium": 3, "low": 2}

# 3. 임베딩 모델 설정
embedding_function = UpstageEmbeddings(model="solar-embedding-1-large")

# Chroma 벡터스토어의 저장 디렉토리 설정
persist_directory = "/mnt/c/Users/kec91/Desktop/capstone_git/vector_db"

# 기존 벡터스토어 삭제
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    logging.info("기존 벡터스토어가 삭제되었습니다.")

# 새로운 Chroma 클라이언트 및 벡터스토어 생성
client = PersistentClient(path=persist_directory)
collection = client.get_or_create_collection("my_collection")
vectorstore = Chroma(client=client, collection_name="my_collection", embedding_function=embedding_function)

# 문서 로딩 및 중요도에 따른 임베딩 처리
try:
    for pdf_path, importance in pdf_files_with_importance:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        for _ in range(importance_weights[importance]):
            vectorstore.add_documents(pages)

    logging.info("모든 PDF 파일이 새로운 벡터스토어에 성공적으로 임베딩되었습니다.")

except Exception as e:
    logging.error(f"벡터스토어 생성 중 오류 발생: {e}")

logging.info("새로운 벡터스토어가 로컬에 저장되었습니다.")

# Vectorstore 정보 출력
logging.info(f"Vectorstore에 저장된 문서 수: {vectorstore._collection.count()}")

# 검색 결과에 가중치를 반영하는 함수
def adjust_score_by_importance(results, importance_scores):
    adjusted_results = []
    for doc, score in results:
        file_name = os.path.basename(doc.metadata.get('source', ''))
        adjusted_score = score * importance_scores.get(file_name, 1)
        adjusted_results.append({"doc": doc, "score": adjusted_score})
    return adjusted_results

# 중요도 점수 맵
importance_scores = {
    "근로기준법(법률)(제18176호)(20211119).pdf": 1.2,
    "2023년 표준 취업규칙.pdf": 0.8,
    "공단 취업규칙.pdf": 0.2
}

# Retriever 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 테스트 쿼리
test_queries = [
    "근로시간에 대해 알려주세요",
    "연차 휴가의 법적 기준은 무엇인가요?",
    "최저임금에 대해 설명해주세요"
]

# 각 쿼리에 대해 retriever 테스트
for query in test_queries:
    logging.info(f"\n쿼리: {query}")
    
    # Retriever 사용
    docs = retriever.get_relevant_documents(query)
    logging.info(f"Retriever로 검색된 문서 수: {len(docs)}")
    
    for i, doc in enumerate(docs, 1):
        logging.info(f"\nRetriever 문서 {i}:")
        logging.info(f"내용: {doc.page_content[:200]}...")
        logging.info(f"메타데이터: {doc.metadata}")
    
    # 직접 similarity_search 사용
    similar_docs = vectorstore.similarity_search_with_score(query, k=3)
    logging.info(f"\nSimilarity Search로 검색된 문서 수: {len(similar_docs)}")
    
    # 검색 결과를 중요도에 따라 조정
    adjusted_results = adjust_score_by_importance(similar_docs, importance_scores)
    
    for i, item in enumerate(adjusted_results, 1):
        doc = item["doc"]
        score = item["score"]
        logging.info(f"\nSimilarity Search 문서 {i}:")
        logging.info(f"내용: {doc.page_content[:200]}...")
        logging.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        logging.info(f"Adjusted Score: {score}")
    
    logging.info("\n" + "="*50)

# 추가: 전체 문서 내용 확인
all_docs = vectorstore.get()
logging.info(f"\n전체 문서 수: {len(all_docs['ids'])}")
for i, (doc_id, doc_content) in enumerate(zip(all_docs['ids'], all_docs['documents']), 1):
    logging.info(f"\n문서 {i} (ID: {doc_id}):")
    logging.info(f"내용: {doc_content[:200]}...")  # 처음 200자만 출력