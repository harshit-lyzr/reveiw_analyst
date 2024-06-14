import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType

st.set_page_config(
    page_title="AI Review Aggregator and Summarizer",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

api = st.sidebar.text_input("Enter Your OPENAI API KEY HERE", type="password")

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("AI Review Aggregator and Summarizer💻")
st.sidebar.markdown("## Welcome to the AI Review Aggregator and Summarizer!")
st.sidebar.markdown(
    "This App Harnesses power of Lyzr Automata to Generate Review analysis and summaries. You Need to input Your reviews and this app Perform Analysis and create summaries for reviews")

if api:
    openai_model = OpenAIModel(
        api_key=api,
        parameters={
            "model": "gpt-4-turbo-preview",
            "temperature": 0.2,
            "max_tokens": 1500,
        },
    )
else:
    st.sidebar.error("Please Enter Your OPENAI API KEY")


def review_analyst(reviews):
    review_agent = Agent(
        prompt_persona="You are an Expert Review Aggregator and Summarizer",
        role="Review Aggregator and Summarizer",
    )

    review_analysis_task = Task(
        name="Review Analysis Task",
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
        model=openai_model,
        agent=review_agent,
        log_output=True,
        instructions=f"""Perform sentiment analysis to classify reviews into positive, negative, or neutral categories.
        Aggregate reviews based on common themes, sentiments, and key points.Summarize multiple reviews into a single coherent review that captures the essence of individual feedback.
        Use techniques like text summarization (extractive and abstractive) to create concise summaries.
        Display consolidated reviews in a user-friendly format, highlighting key insights, pros and cons, and overall sentiment. 
        Provide visual aids like star ratings, sentiment graphs, and keyword clouds to enhance readability.

        Reviews: {reviews}

        ##Output Requirements:
        ##Movie Name:
        ##Overview:
        ##Summarized Reviews:
        ##Key Insights:
            ###Pros:
            ###Cons:
        ##Overall Sentiment:
            ###Star Ratings: ⭐(use this emoji for rating️) 
            ###Sentiment Graph:
            ###Keyword Cloud:

        """,
    )

    output = LinearSyncPipeline(
        name="review Analysis",
        completion_message="Review Analysis Done!",
        tasks=[
            review_analysis_task
        ],
    ).run()
    return output[0]['task_output']


review = st.text_area("Enter Your Reviews", height=300)

if st.button("Convert"):
    solution = review_analyst(review)
    st.markdown(solution)
