import os
import logging
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

# 환경 변수 로드
load_dotenv()

# Chroma DB 로드 함수
def load_local_chroma_db(persist_directory: str):
    try:
        client = PersistentClient(path=persist_directory)
        embedding_function = UpstageEmbeddings(model="solar-embedding-1-large")
        return Chroma(client=client, embedding_function=embedding_function)
    except Exception as e:
        print(f"로컬 Chroma DB 로드 중 오류 발생: {e}")
        return None

# Chroma DB 디렉토리 경로
persist_directory = "/mnt/c/Users/kec91/Desktop/capstone_git/vector_db"

# 로컬 Chroma DB 로드
vectorstore = load_local_chroma_db(persist_directory)

if vectorstore is None:
    print("Vectorstore 로드 실패")
else:
    # Vectorstore 정보 출력
    print(f"Vectorstore에 저장된 문서 수: {vectorstore._collection.count()}")

    # 임베딩 함수 테스트
    embedding_function = UpstageEmbeddings(model="solar-embedding-1-large")
    test_embedding = embedding_function.embed_query("테스트 쿼리")
    print(f"임베딩 차원: {len(test_embedding)}")

    # Retriever 생성 (설정 수정)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 테스트 쿼리
    test_queries = [
        "근로시간에 대해 알려주세요",
        "연차 휴가의 법적 기준은 무엇인가요?",
        "최저임금에 대해 설명해주세요"
    ]

    # 각 쿼리에 대해 retriever 테스트
    for query in test_queries:
        print(f"\n쿼리: {query}")
        
        # Retriever 사용
        docs = retriever.get_relevant_documents(query)
        print(f"Retriever로 검색된 문서 수: {len(docs)}")
        
        for i, doc in enumerate(docs, 1):
            print(f"\nRetriever 문서 {i}:")
            print(f"내용: {doc.page_content[:200]}...")
            print(f"메타데이터: {doc.metadata}")
        
        # 직접 similarity_search 사용
        similar_docs = vectorstore.similarity_search(query, k=3)
        print(f"\nSimilarity Search로 검색된 문서 수: {len(similar_docs)}")
        
        for i, doc in enumerate(similar_docs, 1):
            print(f"\nSimilarity Search 문서 {i}:")
            print(f"내용: {doc.page_content[:200]}...")
            print(f"메타데이터: {doc.metadata}")
        
        print("\n" + "="*50)

    # 추가: 전체 문서 내용 확인
    all_docs = vectorstore.get()
    print(f"\n전체 문서 수: {len(all_docs['ids'])}")
    for i, (doc_id, doc_content) in enumerate(zip(all_docs['ids'], all_docs['documents']), 1):
        print(f"\n문서 {i} (ID: {doc_id}):")
        print(f"내용: {doc_content[:200]}...")  # 처음 200자만 출력
