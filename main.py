import streamlit as st
import pandas as pd
from matcher import MatchTitle, MatcherText, upMatcherText


df = pd.read_csv('mos_news.csv')
matcher = MatchTitle(df)
# main_matcher = MatcherText()
text_matcher = upMatcherText()
st.title("Проверка фейковой новости")

def main():
    st.subheader("Проверка новости по заголовку и тексту")
    with st.form(key='my_form'):
        title = st.text_input("Заголовок новости")
        text = st.text_area("Текст новости", height=450)
        submit_button = st.form_submit_button(label='Submit')

    if  submit_button and len(title)>0 and len(text)>0:
        result = matcher.get_orig_title(title, 1)
        # result["sim_text"] = main_matcher.get_score_fake(result.text.to_list(), text)
        result['sim_text2'] = text_matcher.get_score_fake(result.text.to_list(), text)
        st.table(result[['site','date','sim_title', 'sim_text2']])
        

if __name__ == "__main__":
    main()