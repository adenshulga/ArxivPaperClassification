# import streamlit as st

# st.title("This is a title")
# st.header("This is a header")
# st.subheader("This is a subheader")
# st.text("This is a text")
# st.markdown("# This is a markdown header 1")
# st.markdown("## This is a markdown header 2")
# st.markdown("### This is a markdown header 3")
# st.markdown("This is a markdown: *bold* **italic** `inline code` ~strikethrough~")
# st.markdown("""This is a code block with syntax highlighting
# ```python
# print("Hello world!")
# ```
# """)
# st.html(
#     "image from url example with html: "
#     "<img src='https://www.wallpaperflare.com/static/450/825/286/kitten-cute-animals-grass-5k-wallpaper.jpg' width=400px>",
# )


# st.write("Text with write")
# st.write(range(10))

# st.success("Success")
# st.info("Information")
# st.warning("Warning")
# st.error("Error")
# exp = ZeroDivisionError("Trying to divide by Zero")
# st.exception(exp)

# # инициализируем переменные
# st.session_state.key1 = "value1"  # Attribute API
# st.session_state["key2"] = "value2"  # Dictionary like API

# # посмотреть что в st.session_state
# st.write(st.session_state)

# # magic
# st.session_state

# # ошибка если неправильный ключ
# # st.write(st.session_state["missing_key"])

# import streamlit as st
# from transformers import pipeline


# @st.cache_resource  # кэширование
# def load_model():
#     return pipeline("sentiment-analysis")  # скачивание модели


# model = load_model()

# query = st.text_input("Your query", value="I love Streamlit! 🎈")
# if query:
#     result = model(query)[0]  # классифицируем
#     st.write(query)
#     st.write(result)
