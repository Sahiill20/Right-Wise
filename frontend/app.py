import streamlit as st

from src.rag.rag_service import RAGService


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RightWise",
    page_icon="⚖️",
    layout="centered",
)


# --------------------------------------------------
# RAG service
# --------------------------------------------------

@st.cache_resource
def get_rag_service() -> RAGService:
    return RAGService()


rag_service = get_rag_service()


# --------------------------------------------------
# Uploaded PDF
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a receipt, warranty card, agreement, or other PDF",
    type=["pdf"],
)


if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = None


if uploaded_file is not None:

    try:
        st.session_state.uploaded_chunks = (
            rag_service.prepare_uploaded_pdf(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )
        )

        st.success(
            f"Loaded {uploaded_file.name} "
            f"({len(st.session_state.uploaded_chunks)} chunks)."
        )

    except ValueError as exc:

        st.error(str(exc))

        st.session_state.uploaded_chunks = None


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("⚖️ RightWise")

st.subheader(
    "Consumer-rights information assistant"
)

st.write(
    "Describe your consumer problem in plain language. "
    "RightWise retrieves relevant provisions from the "
    "Consumer Protection Act, 2019 and explains them in "
    "simple language."
)


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# User input
# --------------------------------------------------

query = st.chat_input(
    "Describe your consumer problem..."
)


# --------------------------------------------------
# Process query
# --------------------------------------------------

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)


    with st.chat_message("assistant"):

        with st.spinner("Searching the law..."):

            try:

                result = rag_service.answer_query(
                    query,
                    extra_chunks=(
                        st.session_state.uploaded_chunks
                    ),
                    conversation_history=(
                        st.session_state.messages[:-1]
                    ),
                )

                answer = result["answer"]
                citations = result["citations"]

                # ----------------------------------
                # Answer
                # ----------------------------------

                st.markdown(answer)

                # ----------------------------------
                # Sources
                # ----------------------------------

                if citations:

                    st.markdown("### Sources")

                    for citation in citations:

                        source_type = citation.get(
                            "source_type",
                            "act",
                        )

                        # --------------------------
                        # Uploaded PDF
                        # --------------------------

                        if source_type == "uploaded_pdf":

                            source_name = citation.get(
                                "source_name",
                                "Uploaded document",
                            )

                            page_start = citation[
                                "page_start"
                            ]

                            page_end = citation[
                                "page_end"
                            ]

                            if page_start == page_end:

                                page_text = (
                                    f"Page {page_start}"
                                )

                            else:

                                page_text = (
                                    f"Pages {page_start}-"
                                    f"{page_end}"
                                )

                            st.markdown(
                                f"**{source_name}**  \n"
                                f"{page_text}  \n"
                                f"`{citation['chunk_id']}`"
                            )

                        # --------------------------
                        # Consumer Protection Act
                        # --------------------------

                        else:

                            page_start = citation[
                                "page_start"
                            ]

                            page_end = citation[
                                "page_end"
                            ]

                            if page_start == page_end:

                                page_text = (
                                    f"Page {page_start}"
                                )

                            else:

                                page_text = (
                                    f"Pages {page_start}-"
                                    f"{page_end}"
                                )

                            st.markdown(
                                f"**Section "
                                f"{citation['section_number']} — "
                                f"{citation['section_title']}**  \n"
                                f"{page_text}  \n"
                                f"`{citation['chunk_id']}`"
                            )

                # ----------------------------------
                # Disclaimer
                # ----------------------------------

                st.caption(
                    "This response provides legal information, "
                    "not legal advice."
                )

                # ----------------------------------
                # Save assistant response
                # ----------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as exc:

                st.error(
                    "RightWise could not generate a response "
                    "right now."
                )

                st.exception(exc)