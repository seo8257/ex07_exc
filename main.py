import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import (    RecursiveCharacterTextSplitter )
from langchain_chroma import (    Chroma  )
from langchain_openai import   OpenAIEmbeddings,    ChatOpenAI 
from langchain_classic.chains import (   create_retrieval_chain )
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import (   ChatPromptTemplate )
from langchain_core.callbacks import   BaseCallbackHandler

@st.cache_data
def load_csv_documents(uploaded_file):
    """업로드된 CSV 파일을 읽어 LangChain Document 리스트로 반환합니다."""
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # CSVLoader를 사용하여 문서 로드
    try:
        loader = CSVLoader(file_path=tmp_file_path, encoding='utf-8')
        documents = loader.load()
    except Exception as e:
        # UTF-8 실패시 CP949 (한국어 윈도우 엑셀 등) 로 시도
        loader = CSVLoader(file_path=tmp_file_path, encoding='cp949')
        documents = loader.load()
    finally:
        # 임시 파일 삭제
        os.remove(tmp_file_path)
        
    return documents

st.title("📄 CSV File Reader")
st.write("----------------")

openai_key = st.text_input(  "OPENAI_API_KEY",    type="password" )

uploaded_file = st.file_uploader(   "CSV 파일을 올려주세요",   type=["csv"] )
st.write("----------------")

class StreamHandler(  BaseCallbackHandler ):
    """
    GPT가 토큰을 생성할 때마다
    Streamlit 화면에 출력하는 Handler

    예:
    GPT:   안녕하세요
    생성 과정:
    안
    안녕
    안녕하세요

    처럼 실시간 출력
    """
    def __init__(  self,    container  ):
        self.container = container
        self.text = ""

    def on_llm_new_token(  self,  token,   **kwargs ):
        # 새 토큰 누적
        self.text += token
        # 화면 갱신
        self.container.markdown(    self.text  )

if uploaded_file is not None:
    documents = load_csv_documents(uploaded_file)
    # st.success(   f"CSV 문서 : {len(documents)}"  )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    texts = text_splitter.split_documents(    documents   )

    # st.info(  f"문서 조각 : {len(texts)}"  )

    embeddings = OpenAIEmbeddings(  api_key=openai_key   )

    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings
    )

    retriever = db.as_retriever(
        search_kwargs={
            "k":3
        }
    )

    st.header(   "CSV 파일에게 질문하세요"   )
    question = st.text_input(   "질문 입력"    )

    if st.button(   "질문하기"   ):
        if question == "":
            st.warning( "질문을 입력하세요"   )
        else:
            with st.spinner(  "답변 생성중..."  ,show_time=True  ): 

                chat_box = st.empty()

                handler = StreamHandler(      chat_box       )

                llm = ChatOpenAI(
                    model="gpt-4.1-mini",
                    temperature=0,
                    api_key=openai_key,
                    streaming=True,
                    callbacks=[ handler  ]
                )

                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 CSV 파일 분석 AI 입니다.
                    Context:   {context}
                    Question:  {input}
                    답변:
                    """
                )

                document_chain = ( create_stuff_documents_chain(   llm,    prompt    )   )

                qa_chain = create_retrieval_chain(
                    retriever,
                    document_chain
                )

                qa_chain.invoke(    {    "input": question    }      )        