import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredEmailLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

CAMINHO_DA_PASTA = "dados-Brutos"

PASTA_DA_BASE_VETORIAL = "base_de_conhecimento_faiss"

def carregar_documentos(caminho_pasta):
    """Lê todos os arquivos de uma pasta e seus subdiretórios."""
    documentos = []
    
    st.info(f"Iniciando a leitura dos arquivos em '{caminho_pasta}'...")
    progress_bar = st.progress(0, text="Lendo arquivos...")
    
    arquivos_encontrados = list(os.walk(caminho_pasta))
    total_files = sum(len(files) for _, _, files in arquivos_encontrados)
    files_processed = 0

    for root, _, files in arquivos_encontrados:
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documentos.extend(loader.load())
                elif file.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                    documentos.extend(loader.load())
                elif file.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    documentos.extend(loader.load())
                elif file.endswith('.eml'):
                    loader = UnstructuredEmailLoader(file_path)
                    documentos.extend(loader.load())
                # Adicionar loader para Excel se necessário (pode ser complexo)
                # elif file.endswith('.xlsx'):
                #     loader = # ... um loader para excel
                #     documentos.extend(loader.load())
            except Exception as e:
                print(f"Erro ao ler o arquivo {file_path}: {e}")
            
            files_processed += 1
            progress_bar.progress(files_processed / total_files, text=f"Lendo: {file}")

    progress_bar.empty()
    st.success(f"{len(documentos)} documentos carregados com sucesso.")
    return documentos

def main():
    st.title("Construtor de Base de Conhecimento com RAG")

    if not os.path.isdir(CAMINHO_DA_PASTA):
        st.error(f"O caminho '{CAMINHO_DA_PASTA}' não é uma pasta válida. Por favor, edite o script `processar_documentos.py`.")
        return

    if st.button("Iniciar Processamento dos Documentos"):
        documentos = carregar_documentos(CAMINHO_DA_PASTA)
        if not documentos:
            st.warning("Nenhum documento foi carregado. Verifique a pasta e os tipos de arquivo.")
            return

        st.info("Dividindo os documentos em pedaços menores (chunks)...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documentos)
        st.success(f"{len(chunks)} chunks de texto criados.")

        st.info("Criando a base de conhecimento vetorial (isso pode demorar)...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text") 
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        vectorstore.save_local(PASTA_DA_BASE_VETORIAL)
        st.success(f"Base de conhecimento salva com sucesso na pasta '{PASTA_DA_BASE_VETORIAL}'!")
        st.balloons()

if __name__ == '__main__':
    main()