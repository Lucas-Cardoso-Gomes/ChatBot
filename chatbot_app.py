import streamlit as st
import ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

PASTA_DA_BASE_VETORIAL = "base_de_conhecimento_faiss"

@st.cache_resource
def carregar_recursos():
    """Carrega a base de conhecimento e o modelo de linguagem."""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.load_local(PASTA_DA_BASE_VETORIAL, embeddings, allow_dangerous_deserialization=True)
        llm = Ollama(model="gemma3:1b")
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        return chain
    except Exception as e:
        st.error(f"Erro ao carregar a base de conhecimento: {e}")
        st.warning("Você já executou o script 'processar_documentos.py' para criar a base de conhecimento?")
        return None

st.set_page_config(page_title="Chat com Documentos", page_icon="📚", layout="wide")
st.title("📚 Chatbot com Seus Documentos (RAG)")
st.write("Faça perguntas sobre os arquivos do seu HDD externo.")

chain = carregar_recursos()

if chain:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Faça uma pergunta sobre seus documentos..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando e pesquisando nos seus arquivos..."):
                result = chain({"question": prompt, "chat_history": st.session_state.chat_history})
                response = result["answer"]
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.session_state.chat_history.append((prompt, response))

                with st.expander("Fontes da Resposta"):
                    for doc in result["source_documents"]:
                        st.write(f"- **Arquivo:** {doc.metadata.get('source', 'N/A')}")
                        st.write(f"> {doc.page_content[:200]}...")

else:
    st.info("A base de conhecimento não foi carregada. Siga as instruções para criá-la.")