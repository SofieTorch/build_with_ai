__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pandas as pd
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
        print(response1)
        response2 = query_papers(antithesis)
        print(response2)
    col1.warning(response1['answer'])
    col2.info(
f"""**Contraste:** {antithesis}

---
{response2['answer']}""")
    
    df1 = pd.DataFrame([f"(Página {i.metadata['page']}) {i.metadata['title']} - {i.metadata['author']}" for i in response1['context']], columns=["Referencias"])
    df2 = pd.DataFrame([f"(Página {i.metadata['page']}) {i.metadata['title']} - {i.metadata['author']}" for i in response2['context']], columns=["Referencias"])
    col1.table(df1)
    col2.table(df2)
