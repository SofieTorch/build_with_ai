__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import time
import streamlit as st
from retriever import query_papers
from prompt_transformer import get_antithesis

st.set_page_config(page_title="BBreak Search Engine", page_icon="📄")
st.title("BBreak: Redescubre las cosas")

input = st.text_area(
    "Escribe algo que quieras verificar o conocer a profundidad :)",
    ""
)

if st.button("Examinar"):
    col1, col2 = st.columns(2)
    with st.spinner('Procesando...'):
        antithesis = get_antithesis(input)
        response1 = query_papers(input)
        response2 = query_papers(antithesis)
    col1.warning(response1)
    col2.info(
f"""**Contraste:** {antithesis}

---
{response2}""")
