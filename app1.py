import streamlit as st
import os
import json
import hashlib
import uuid
import base64
from typing import Dict, List, Any
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_chroma import Chroma
from langchain_core.messages import ChatMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from chromadb import PersistentClient

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="Legal Advice AI Assistant", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

# 이미지를 base64로 인코딩하는 함수
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# 이미지 경로 설정 
chatbot_image_path = "/mnt/c/Users/kec91/Desktop/capstone_git/static/판사.png"
user_image_path = "/mnt/c/Users/kec91/Desktop/capstone_git/static/molu.png"
logo_path = "/mnt/c/Users/kec91/Desktop/capstone_git/static/logo.png"


# 이미지를 base64로 인코딩
chatbot_image = get_image_base64(chatbot_image_path)
user_image = get_image_base64(user_image_path)
logo = get_image_base64(logo_path)

# CSS를 사용하여 레이아웃 및 색상 조정
st.markdown("""
<style>
    body {background-color: #e8e1d9;}
    .stApp, header, footer {background-color: #e8e1d9 !important;}
    .stTextInput > div > div > input {background-color: #f5f0e8; border: 1px solid #d0c8b5;}
    .stButton > button {background-color: #c41230; color: white;}
    .stButton > button:hover {background-color: #9e0e26;}
    .chat-container {display: flex; flex-direction: column; padding: 10px;}
    .message {border-radius: 20px; padding: 10px 15px; margin-bottom: 10px; max-width: 70%; display: flex; align-items: flex-start;}
    .user-message {background-color: #f5f0e8; color: #1c1c1c; align-self: flex-end; flex-direction: row-reverse;}
    .bot-message {background-color: #ffffff; color: #1c1c1c; align-self: flex-start;}
    .message-content {margin: 0 10px;}
    .avatar {width: 30px; height: 30px; border-radius: 50%;}
    .logo-container {display: flex; justify-content: center; margin-bottom: 20px;}
    .logo {width: 80%; max-width: 300px;}
    .main-logo-container {display: flex; justify-content: center; margin-bottom: 2rem;}
    .main-logo {width: 40%; max-width: 300px;}
</style>
""", unsafe_allow_html=True)


# 로고 HTML
logo_html = f"""
<div class="logo-container">
    <img src="data:image/png;base64,{logo}" class="logo">
</div>
"""

# 사용자 및 세션 데이터 파일 경로
USERS_FILE = "users.json"
SESSIONS_FILE = "sessions.json"

# 사용자 및 세션 데이터 관리 함수들 (이전과 동일)
def load_data(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}

def save_data(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    users = load_data(USERS_FILE)
    if username in users:
        return False
    users[username] = hash_password(password)
    save_data(users, USERS_FILE)
    return True

def authenticate_user(username, password):
    users = load_data(USERS_FILE)
    return username in users and users[username] == hash_password(password)

# Chroma DB 로드 함수 (수정됨)
@st.cache_resource
def load_local_chroma_db(persist_directory: str):
    try:
        client = PersistentClient(path=persist_directory)
        embedding_function = UpstageEmbeddings(model="solar-embedding-1-large")
        collection = client.get_or_create_collection("my_collection")
        vectorstore = Chroma(client=client, collection_name="my_collection", embedding_function=embedding_function)
        return vectorstore
    except Exception as e:
        st.error(f"로컬 Chroma DB 로드 중 오류 발생: {e}")
        return None

# 메시지 토큰 수 제한 함수
def truncate_messages(messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    truncated = []
    current_tokens = 0
    for message in reversed(messages):
        message_tokens = len(message['content'].split())  # 간단한 토큰 계산
        if current_tokens + message_tokens > max_tokens:
            break
        truncated.insert(0, message)
        current_tokens += message_tokens
    return truncated

# 로그인 페이지 (구현 추가)
def login_page():
    st.sidebar.title("Login / Register")
    st.markdown(f"""
    <div class="main-logo-container">
        <img src="data:image/png;base64,{logo}" class="main-logo">
    </div>
    """, unsafe_allow_html=True)
    
    st.title("⚖️ Legal Advice AI Assistant - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.user = username
                st.session_state.logged_in = True
                st.success("로그인 성공!")
                st.rerun()
            else:
                st.error("잘못된 사용자명 또는 비밀번호입니다.")
    with col2:
        if st.button("Register"):
            if register_user(username, password):
                st.success("등록 성공! 이제 로그인하세요.")
            else:
                st.error("이미 존재하는 사용자명입니다.")
# 메인 애플리케이션 UI (개선됨)
def main_app():
    st.sidebar.markdown(logo_html, unsafe_allow_html=True)
    
    st.sidebar.title("세션 관리")
    user_sessions = get_user_sessions(st.session_state.user)
    
    if st.sidebar.button("새 세션 만들기"):
        new_session_id = create_new_session(st.session_state.user)
        st.session_state.current_session = new_session_id
        st.rerun()

    session_options = list(user_sessions.keys())
    if session_options:
        selected_session = st.sidebar.selectbox("세션 선택", session_options, index=session_options.index(st.session_state.get('current_session', session_options[0])))
        if selected_session != st.session_state.get('current_session'):
            st.session_state.current_session = selected_session
            st.rerun()
    else:
        st.sidebar.warning("세션이 없습니다. 새 세션을 만드세요.")
        return

    st.title("⚖️ Legal Advice AI Assistant")

    # 현재 세션의 메시지 로드
    current_messages = user_sessions[st.session_state.current_session]["messages"]
    
    # 최근 N개의 메시지만 사용하도록 설정
    MAX_MESSAGES = 5
    recent_messages = current_messages[-MAX_MESSAGES:] if len(current_messages) > MAX_MESSAGES else current_messages

    # Chroma DB 디렉토리 경로
    persist_directory = "/mnt/c/Users/kec91/Desktop/capstone_git/vector_db"

    # 로컬 Chroma DB 로드
    vectorstore = load_local_chroma_db(persist_directory)
    if vectorstore is None:
        st.error("Vectorstore 로드에 실패했습니다. 관리자에게 문의하세요.")
        return

    # Retriever 생성 (개선됨)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ChatUpstage 초기화
    chat = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"))

    # 프롬프트 파일 경로
    qa_system_prompt_path = os.path.join("/mnt/c/Users/kec91/Desktop/capstone_git/prompts", "qa_testpromt1.txt")
    contextualize_q_system_prompt_path = os.path.join("/mnt/c/Users/kec91/Desktop/capstone_git/prompts", "system_prompt.txt")

    # 프롬프트 파일 읽기
    with open(qa_system_prompt_path, "r", encoding="utf-8") as f:
        qa_system_prompt = f.read()

    with open(contextualize_q_system_prompt_path, "r", encoding="utf-8") as f:
        contextualize_q_system_prompt = f.read()

    # contextualize_q_prompt 정의
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 이전 대화를 기억하는 리트리버 생성
    history_aware_retriever = create_history_aware_retriever(chat, retriever, contextualize_q_prompt)

    # qa_prompt 정의
    qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("human", "Context: {context}")])

    # 체인 생성
    question_answer_chain = create_stuff_documents_chain(
    chat, 
    qa_prompt,
    document_variable_name="context")
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 채팅 메시지를 표시할 컨테이너
    chat_container = st.container()
    
    # 메인 채팅 영역에 대화 표시
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in current_messages:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="message user-message">
                    <img src="data:image/png;base64,{user_image}" class="avatar">
                    <div class="message-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message bot-message">
                    <img src="data:image/png;base64,{chatbot_image}" class="avatar">
                    <div class="message-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 사용자 입력 처리
    if prompt := st.chat_input("법률 관련 질문을 입력하세요."):
        current_messages.append({"role": "user", "content": prompt})
        
        # 최근 N개의 메시지만 포함하는 채팅 히스토리 생성
        chat_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in recent_messages
        ]
        
        # AI의 답변을 받아서 저장하고, 보여주기
        with st.spinner("답변 생성 중..."):
            MAX_TOKENS = 1000  # 적절한 값으로 조정
            truncated_messages = truncate_messages(current_messages, MAX_TOKENS)
            result = rag_chain.invoke(
                {'input': prompt, 'chat_history': truncated_messages}
            )
            
            full_response = result['answer']

        current_messages.append({"role": "assistant", "content": full_response})
        save_session_messages(st.session_state.user, st.session_state.current_session, current_messages)
        st.rerun()

    # 로그아웃 버튼
    if st.sidebar.button("로그아웃"):
        st.session_state.user = None
        st.session_state.logged_in = False
        st.rerun()

# 세션 관리 함수들 (이전과 동일)
def create_new_session(username):
    sessions = load_data(SESSIONS_FILE)
    if username not in sessions:
        sessions[username] = {}
    session_id = str(uuid.uuid4())
    sessions[username][session_id] = {"messages": []}
    save_data(sessions, SESSIONS_FILE)
    return session_id

def get_user_sessions(username):
    sessions = load_data(SESSIONS_FILE)
    return sessions.get(username, {})

def save_session_messages(username, session_id, messages):
    sessions = load_data(SESSIONS_FILE)
    sessions[username][session_id]["messages"] = messages
    save_data(sessions, SESSIONS_FILE)

# 메인 실행 부분
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:
    main_app()