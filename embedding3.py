import os
import re
import logging
import shutil
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from chromadb import PersistentClient
from langchain_chroma import Chroma

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# .env 파일에서 환경 변수 불러오기
load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")

# PDF 파일 목록 및 중요도 설정
pdf_files_with_importance = [
    ("/mnt/c/Users/kec91/Desktop/capstone_git/refer/근로기준법(법률)(제18176호)(20211119).pdf", 1.2),
    ("/mnt/c/Users/kec91/Desktop/capstone_git/refer/2023년 표준 취업규칙.pdf", 0.8)]

# 임베딩 모델 설정
embedding_function = UpstageEmbeddings(model="solar-embedding-1-large")

# Chroma 벡터스토어의 저장 디렉토리 설정
persist_directory = "/mnt/c/Users/kec91/Desktop/capstone_git/vector_db"


# 기존 벡터스토어 삭제
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    logging.info("기존 벡터스토어가 삭제되었습니다.")

# Chroma 클라이언트 및 벡터스토어 생성
client = PersistentClient(path=persist_directory)
collection = client.get_or_create_collection("my_collection")
vectorstore = Chroma(client=client, collection_name="my_collection", embedding_function=embedding_function)

def split_text(page_content):
    split_pattern = r'(?=제\d+조(\의\d+)?\([^)]+\)\s*[^제]*)'
    split_text = re.split(split_pattern, page_content)
    cleaned_split_text = [part.strip() for part in split_text if part and len(part.strip()) > 20]
    return cleaned_split_text

def extract_metadata(text, page, labor_id, source, importance):
    match = re.search(r'제(\d+조(\의\d+)?)(\([^)]+\))\s*(.+)', text, re.DOTALL)
    if match:
        law_id = int(re.search(r'\d+', match.group(1)).group())
        law_num = f"제{match.group(1).strip()}"
        title = match.group(3).strip('()')
        content = match.group(4).strip()
        return {
            "labor_id": labor_id,
            "law_id": law_id,
            "law_num": law_num,
            "title": title,
            "page": page,
            "source": source,
            "importance": importance,
            "contents": content
        }
    return None

def process_pdf(pdf_path, importance):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    final_docs = []
    labor_id_counter = 1
    end_of_data = False

    for page_num, page in enumerate(pages):
        if end_of_data:
            break
        page_content = page.page_content
        if '\n부     칙 ' in page_content:
            page_content = page_content.split('\n부     칙 ')[0]
            end_of_data = True
        
        cleaned_split_text = split_text(page_content)
        for item in cleaned_split_text:
            meta = extract_metadata(item, page_num + 1, labor_id_counter, os.path.basename(pdf_path), importance)
            if meta:
                new_doc = Document(page_content=meta['contents'], metadata=meta)
                final_docs.append(new_doc)
                labor_id_counter += 1

    return final_docs

# 문서 처리 및 임베딩
for pdf_path, importance in pdf_files_with_importance:
    try:
        docs = process_pdf(pdf_path, importance)
        vectorstore.add_documents(docs)
        logging.info(f"{pdf_path} 파일이 성공적으로 처리되고 임베딩되었습니다.")
    except Exception as e:
        logging.error(f"{pdf_path} 파일 처리 중 오류 발생: {e}")

logging.info("모든 PDF 파일의 처리 및 임베딩이 완료되었습니다.")
logging.info(f"Vectorstore에 저장된 문서 수: {vectorstore._collection.count()}")

# 가중치를 반영한 검색 함수
def weighted_similarity_search(query, k=5):
    results = vectorstore.similarity_search_with_score(query, k=k)
    weighted_results = []
    for doc, score in results:
        importance = doc.metadata.get("importance", 1.0)
        weighted_score = score * importance
        weighted_results.append((doc, weighted_score))
    return sorted(weighted_results, key=lambda x: x[1], reverse=True)

# 테스트 쿼리
test_queries = [
    "근로시간에 대해 알려주세요",
    "연차 휴가의 법적 기준은 무엇인가요?",
    "최저임금에 대해 설명해주세요"
]

# 각 쿼리에 대해 가중치 검색 테스트
for query in test_queries:
    logging.info(f"\n쿼리: {query}")
    results = weighted_similarity_search(query)
    
    for i, (doc, score) in enumerate(results, 1):
        logging.info(f"\n검색 결과 {i}:")
        logging.info(f"법조문: {doc.metadata.get('law_num')} {doc.metadata.get('title')}")
        logging.info(f"내용: {doc.page_content[:200]}...")
        logging.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        logging.info(f"Importance: {doc.metadata.get('importance', 1.0)}")
        logging.info(f"Weighted Score: {score}")
    
    logging.info("\n" + "="*50)