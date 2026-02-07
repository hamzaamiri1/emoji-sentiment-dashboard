import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(
    page_title="Emoji Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }

    h1, h2, h3, h4 {
        color: #ffffff;
    }

    p, span, div {
        color: #d1d5db;
        font-size: 16px;
    }

    .stMetricLabel {
        color: #9ca3af;
    }

    .stMetricValue {
        color: #ffffff;
        font-size: 28px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#9ca3af",
    "axes.labelcolor": "#e6e6e6",
    "xtick.color": "#d1d5db",
    "ytick.color": "#d1d5db",
    "text.color": "#e6e6e6",
    "axes.titleweight": "bold"
})


st.title("Emoji Sentiment Dataset – Data Cleaning & Analysis")
st.markdown(
    """
    This interactive app presents insights obtained **after preprocessing**
    the Emoji-Sentiment dataset.  
    The focus is on **data cleaning, feature engineering, and exploratory analysis**.
    """
)

st.divider()


@st.cache_data
def load_data():
    df = pd.read_csv("emoji-sentiment.csv")

    df = df.rename(columns={
        'Char': 'char',
        'Image [twemoji]': 'image',
        'Unicode codepoint': 'unicode_codepoint',
        'Occurrences [5...max]': 'occurrences',
        'Position [0...1]': 'position',
        'Neg [0...1]': 'negative',
        'Neut [0...1]': 'neutral',
        'Pos [0...1]': 'positive',
        'Sentiment bar (c.i. 95%)': 'sentiment_bar',
        'Unicode name': 'unicode_name',
        'Unicode block': 'unicode_block'
    })

    df = df[
        ['char', 'unicode_name', 'occurrences',
         'position', 'negative', 'neutral', 'positive']
    ]

    df["sentiment"] = df["positive"] - df["negative"]
    df["positive flag"] = df["sentiment"] > 0

    return df


df = load_data()


st.subheader("1️⃣ Overall sentiment distribution")

positive_percent = df["positive flag"].mean() * 100

col1, col2 = st.columns(2)
col1.metric("Positive emojis (%)", f"{positive_percent:.2f}%")
col2.metric("Total emojis", len(df))

sentiment_counts = df["positive flag"].value_counts()

fig, ax = plt.subplots(figsize=(4, 2.5), dpi=120)
ax.bar(
    ["Negative", "Positive"],
    [sentiment_counts.get(False, 0), sentiment_counts.get(True, 0)]
)
ax.set_ylabel("Number of emojis")
ax.set_title("Emoji sentiment distribution")
plt.tight_layout()
st.pyplot(fig)

st.divider()


st.subheader("2️⃣ Top 20 most frequent emojis")

top20 = df.sort_values("occurrences", ascending=False).head(20)
positive_top = top20["positive flag"].mean() * 100

st.write(f"**Percentage of positive emojis in Top 20:** {positive_top:.2f}%")

st.dataframe(top20.astype(str), use_container_width=True)

fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
ax.bar(top20["char"], top20["occurrences"])
ax.set_xlabel("Emoji")
ax.set_ylabel("Occurrences")
ax.set_title("Top 20 most frequent emojis")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
st.pyplot(fig)

st.divider()


st.subheader("3️⃣ Most extreme sentiments (occurrences > 500)")

df_popular = df.query("occurrences > 500")

max_positive = df_popular.loc[df_popular["sentiment"].idxmax()]
max_negative = df_popular.loc[df_popular["sentiment"].idxmin()]

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Most positive emoji**")
    st.write(max_positive.astype(str))

with col2:
    st.markdown("**Most negative emoji**")
    st.write(max_negative.astype(str))

st.divider()


st.subheader("4️⃣ Average emoji position in text")

avg_position = df["position"].mean()

if avg_position > 0.6:
    st.success(f"Average position is towards the END ({avg_position:.2f})")
elif avg_position < 0.4:
    st.success(f"Average position is towards the BEGINNING ({avg_position:.2f})")
else:
    st.success(f"Average position is in the MIDDLE ({avg_position:.2f})")

st.divider()


st.subheader("5️⃣ Placement of positive vs negative emojis")

positive_emojis = df[df["positive flag"]]
negative_emojis = df[~df["positive flag"]]

avg_pos_positive = positive_emojis["position"].mean()
avg_pos_negative = negative_emojis["position"].mean()

st.write(f"Average position of **POSITIVE** emojis: {avg_pos_positive:.3f}")
st.write(f"Average position of **NEGATIVE** emojis: {avg_pos_negative:.3f}")
st.write(f"Difference: {abs(avg_pos_positive - avg_pos_negative):.3f}")

if avg_pos_positive > avg_pos_negative:
    st.info("Positive emojis tend to appear closer to the END")
else:
    st.info("Negative emojis tend to appear closer to the END")

fig, ax = plt.subplots(figsize=(4, 2.5), dpi=120)
ax.bar(
    ["Positive emojis", "Negative emojis"],
    [avg_pos_positive, avg_pos_negative]
)
ax.set_ylabel("Average position in text")
ax.set_title("Emoji placement comparison")
plt.tight_layout()
st.pyplot(fig)
