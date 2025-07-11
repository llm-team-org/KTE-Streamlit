import streamlit as st
import asyncio
from functions import get_law_summary

# Configure Streamlit page
st.set_page_config(
    page_title="Korean Law Summarizer",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Title
st.title("Korean Law Summarizer")
st.markdown("Ask questions about Korean law and get AI-powered summaries with sources.")

# Sidebar for session management
with st.sidebar:
    # st.header("Chat Data")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    language = st.selectbox(
        "Choose Language",
        options=["English", "Korean"],
        index=0
    )
    st.markdown("---")
    # st.markdown(f"**Messages:** {len(st.session_state.messages)}")
#
#     if st.session_state.messages:
#         st.markdown("**Recent queries:**")
#         for i, msg in enumerate(reversed(st.session_state.messages[-6:])):
#             if msg["role"] == "user":
#                 st.caption(f"• {msg['content'][:40]}...")

# Chat interface
st.markdown("### Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.code(f"{i}. {source}", language=None)

# Chat input
if prompt := st.chat_input("Ask your legal question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query..."):
            try:
                print(f"[DEBUG] Starting query processing for: {prompt}")
                summary, sources = asyncio.run(get_law_summary(prompt,language=language))
                print(f"[DEBUG] Query completed successfully")
                print(f"[DEBUG] Summary length: {len(summary) if summary else 0}")
                print(f"[DEBUG] Number of sources: {len(sources) if sources else 0}")

                # Display summary
                st.write(summary)

                # Display sources
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources, 1):
                            if source.startswith(('http://', 'https://')):
                                st.markdown(f"{i}. [{source}]({source})")
                            else:
                                st.code(f"{i}. {source}", language=None)

                # Add assistant message to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary,
                    "sources": sources
                })

            except Exception as e:
                print(f"[ERROR] Exception occurred: {type(e).__name__}: {str(e)}")
                print(f"[ERROR] Full traceback:")
                import traceback

                traceback.print_exc()

                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.markdown("---")