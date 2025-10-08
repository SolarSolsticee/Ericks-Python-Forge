import streamlit as st

# Set the page configuration for a clean, modern look
# This should be the first Streamlit command in your script
st.set_page_config(
    page_title="Erick's Python Forge",
    page_icon="🛠️",  # An emoji that fits the "Forge" theme
    layout="wide",  # Use the full page width
    initial_sidebar_state="expanded",  # The sidebar starts open
    menu_items={
        'About': "# This is a hub for custom Python data tools."
    }
)

# --- Sidebar Content ---
# This content will be shown on every page
with st.sidebar:
    st.title("Erick's Python Forge 🛠️")
    st.info("Select a tool from the list above to get started.")
    st.success("New tools and updates will appear here over time.")

# --- Main Page Content ---

# Main title with a fitting emoji
st.title("Welcome to the Forge 🔥")

# A visual separator
st.divider()

# Introduction and instructions laid out in columns
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("What is This?")
    st.markdown(
        """
        The **Python Forge** is a personal collection of custom-built applications designed for **measurement, 
        data analysis, and visualization**. 

        Each tool in this hub has been crafted to solve specific engineering and research challenges, 
        transforming raw data into clear, actionable insights.
        """
    )

with col2:
    st.header("How to Use")
    st.markdown(
        """
        Getting started is simple:

        1. **Navigate**: Use the sidebar on the left to browse the available tools.
        2. **Select**: Click on the name of the tool you wish to use.
        3. **Interact**: Follow the instructions within each tool to upload your data or adjust the parameters.

        Results like plots, measurements, or data tables will be displayed directly in the app.
        """
    )

st.divider()

# A little footer to add a professional touch
st.markdown("Created by Erick Pepek | Research Engineer")