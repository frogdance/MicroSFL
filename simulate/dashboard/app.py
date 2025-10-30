import os
import pandas as pd
import psycopg2
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
from streamlit_autorefresh import st_autorefresh

# -----------------------
# Database config
# -----------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Connection for psycopg2 (realtime charts)
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

# Connection for SQLAlchemy (participant table)
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)

# -----------------------
# Streamlit app
# -----------------------
st.set_page_config(page_title="Federated Learning Dashboard", layout="wide")
st.title("Federated Learning Dashboard")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="loss_accuracy_refresh")

# -----------------------
# Fetch data
# -----------------------
df_loss = pd.read_sql(
    "SELECT global_round, loss, client_id FROM metrics ORDER BY global_round ASC",
    conn
)

df_acc = pd.read_sql(
    "SELECT global_round, accuracy, client_id, timestamp FROM accuracies ORDER BY timestamp ASC",
    conn
)
if not df_acc.empty:
    df_acc["timestamp"] = pd.to_datetime(df_acc["timestamp"])

# -----------------------
# Loss & Accuracy Charts (1 hàng)
# -----------------------
st.subheader("Training Metrics")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Loss Chart")
    if not df_loss.empty:
        fig_loss = px.line(
            df_loss,
            x="global_round",
            y="loss",
            color="client_id",
            markers=True,
            labels={"global_round":"Global Round","loss":"Loss","client_id":"Client"},
            title="Client-wise Loss over Global Rounds"
        )
        fig_loss.update_layout(width=450, height=400)
        st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.info("No loss data available yet.")

with col2:
    st.markdown("### Accuracy Chart")
    if not df_acc.empty:
        fig_acc = px.line(
            df_acc,
            x="global_round",
            y="accuracy",
            color="client_id",
            markers=True,
            labels={"global_round":"Global Round","accuracy":"Accuracy","client_id":"Client"},
            title="Client-wise Accuracy over Global Rounds"
        )
        fig_acc.update_layout(width=450, height=400)
        st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.info("No accuracy data available yet.")

# -----------------------
# Participant Table + Client Delay Scatter (1 hàng)
# -----------------------
st.subheader("Participants & Client Delay")
col1, col2 = st.columns([1,2])  # table nhỏ, scatter lớn

with col1:
    tabs = st.tabs(["Clients", "Fed Server", "Split Server"])
    roles = ["client", "fed", "split"]
    for tab, role in zip(tabs, roles):
        with tab:
            st.subheader(f"{role.capitalize()} Participants")
            df_clients = pd.read_sql(f"SELECT client_id, ip FROM clients WHERE role='{role}'", engine)
            if not df_clients.empty:
                st.dataframe(df_clients, use_container_width=True)
            else:
                st.info(f"No {role} participants yet.")

with col2:
    st.subheader("Client Accuracy Timeline (Delay)")
    if not df_acc.empty:
        fig_scatter = px.scatter(
            df_acc,
            x="timestamp",
            y="client_id",
            color="accuracy",
            size_max=15,
            hover_data=["accuracy", "global_round"],
            title="Client Accuracy Updates Over Time (Delay)"
        )
        fig_scatter.update_yaxes(categoryorder="total ascending")
        fig_scatter.update_layout(width=900, height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No accuracy data available yet.")
