import streamlit as st
import pandas as pd
from matcher import MatchTitle, MatcherText, upMatcherText
from QA_matcher import qaMatcher


df = pd.read_csv('mos_news.csv')
matcher = MatchTitle(df)
# main_matcher = MatcherText()
text_matcher = upMatcherText(device='cpu')
qa_matcher = qaMatcher()
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
        result['sim_text'] = text_matcher.get_score_fake(result.text.to_list(), text)
        doubtful_results, main_info = qa_matcher.predict(result.text.values[0], text)
        result = result.to_dict('r')[0]
        site = result['site']
        date = result['date']
        sim_title = round(result['sim_title'] * 100, 1)
        sim_text = round((1 - result['sim_text']) * 100, 1)
        max_fake_qa_score = round(main_info['max_fake_score'] * 100, 1)
        mean_fake_qa_score = round(main_info['mean_fake_score'] * 100, 1)
        st.write(f'Ссылка на оригинальную статью {site}')
        st.write(f'Дата и время публикации оригинальной новости {date}')
        st.write(f'Схожесть заголовков данной статьи и оригинальной {sim_title}%')
        st.write(f'Вероятность, что данная статья является фейковой {sim_text}%')
        st.write(f'\nМаксимальная расхожесть ответов составляет {max_fake_qa_score}%')
        # st.write(f'\nСредний показатель фейковости данной статьи составляет {mean_fake_qa_score}%')
        if len(doubtful_results) != 0:
            st.write('\n\nСистема засомневалась в следующих утверждениях:')

            for res in doubtful_results:
                answer = f'\n\nВопрос: "{res["question"]}"'
                answer += f'\nОтвет из заданной статьи: "{res["answer_from_article"]}"'
                answer += f'\nОтвет из достоверной статьи: "{res["true_answer"]}"'
                answer += f'\nСхожесть ответов: {round(res["confidence"] * 100, 1)}%\n'
                st.text(answer)
        

if __name__ == "__main__":
    main()