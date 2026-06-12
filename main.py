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

st.title("📄 CSV File Reader")
st.write("----------------")

openai_key = st.text_input(  "OPENAI_API_KEY",    type="password" )

uploaded_file = st.file_uploader(   "CSV 파일을 올려주세요",   type=["csv"] )
st.write("----------------")