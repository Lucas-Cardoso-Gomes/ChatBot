import streamlit as st
import ollama
import io
import pypdf 
import docx

st.set_page_config(page_title="IA - PM Logística", page_icon="PM Logo - rbg.png", layout="wide")
st.image("PM Logo - rbg.png", width=100, caption="PM Logística")
st.title("Chatbot Local com Gemma/Gemini 3 via Ollama")
st.info("Esse Chat BOT está rodando localmente nos equipamentos da TI, seus dados estão seguros!")

def ler_pdf(arquivo):
    """
    Função para extrair texto de um arquivo PDF.
    Recebe o arquivo carregado pelo Streamlit e retorna o texto contido nele.
    """
    try:
        leitor_pdf = pypdf.PdfReader(io.BytesIO(arquivo.read()))
        texto_completo = ""
        for pagina in leitor_pdf.pages:
            texto_completo += pagina.extract_text() + "\n"
        return texto_completo
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
        return None

def ler_docx(arquivo):
    """
    Função para extrair texto de um arquivo DOCX (Word).
    Recebe o arquivo carregado pelo Streamlit e retorna o texto contido nele.
    """
    try:
        documento = docx.Document(io.BytesIO(arquivo.read()))
        texto_completo = ""
        for paragrafo in documento.paragraphs:
            texto_completo += paragrafo.text + "\n"
        return texto_completo
    except Exception as e:
        st.error(f"Erro ao ler o arquivo DOCX: {e}")
        return None

def obter_resposta_local(prompt_usuario, historico_chat):
    """
    Função para obter a resposta do modelo local via Ollama em modo streaming.
    """
    try:
        historico_chat.append({'role': 'user', 'content': prompt_usuario})
        
        response_stream = ollama.chat(
            model='gemma3:1b',
            messages=historico_chat,
            stream=True
        )
        
        resposta_completa = []
        def stream_wrapper():
            for chunk in response_stream:
                token = chunk['message']['content']
                resposta_completa.append(token)
                yield token
            
            historico_chat.append({'role': 'assistant', 'content': "".join(resposta_completa)})

        return stream_wrapper
        
    except Exception as e:
        st.error(f"Erro ao contatar o Ollama. Verifique se ele está em execução. Detalhe: {e}")
        if historico_chat:
            historico_chat.pop()
        return None

if "historico_chat" not in st.session_state:
    st.session_state.historico_chat = []

for mensagem in st.session_state.historico_chat:
    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])

arquivo_enviado = st.file_uploader(
    "Anexe um arquivo para análise (.txt, .pdf, .docx)",
    type=['txt', 'pdf', 'docx']
)

if arquivo_enviado is not None:
    extensao = arquivo_enviado.name.split('.')[-1].lower()
    conteudo_texto = ""

    if extensao == "txt":
        conteudo_texto = arquivo_enviado.read().decode("utf-8")
    elif extensao == "pdf":
        conteudo_texto = ler_pdf(arquivo_enviado)
    elif extensao == "docx":
        conteudo_texto = ler_docx(arquivo_enviado)
    
    if conteudo_texto:
        st.success(f"Arquivo '{arquivo_enviado.name}' carregado e lido com sucesso!")
        
        prompt_arquivo = f"""
        Analise o seguinte texto e me forneça um resumo conciso,
        seguido pelos pontos mais importantes em formato de lista (bullet points).
        
        Texto para análise:
        ---
        {conteudo_texto}
        ---
        """
        
        with st.chat_message("user"):
            st.markdown(f"Por favor, analise o arquivo '{arquivo_enviado.name}'.")

        with st.chat_message("assistant"):
            with st.spinner("Analisando o texto localmente..."):
                resposta_stream = obter_resposta_local(prompt_arquivo, st.session_state.historico_chat)
                if resposta_stream:
                    st.write_stream(resposta_stream)

prompt_usuario = st.chat_input("Converse com o seu assistente local...")

if prompt_usuario:
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
    
    with st.chat_message("assistant"):
        resposta_stream = obter_resposta_local(prompt_usuario, st.session_state.historico_chat)
        if resposta_stream:
            st.write_stream(resposta_stream)