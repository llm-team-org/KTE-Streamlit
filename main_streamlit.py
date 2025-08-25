import streamlit as st
import asyncio
import tempfile
import os
import pandas as pd
import json

# Import all functions from their respective modules
from functions_5_3 import drawing_chatbot, get_pdf_info
from functions_5_4 import extraction_review, cleanup_temp_directory
from functions_5_5 import auto_complete, cleanup_temp_directory_excel
from functions_5_6 import finishing_material_comparison,cleanup_temp_directories_finishing
from functions_5_7 import multi_doc_query
# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.title("📄 PDF Assistant")
    st.markdown("---")

    # Module selection
    selected_module = st.selectbox(
        "Select Module",
        ["Drawing Chatbot", "Extraction Review", "Excel Auto Complete", "Finishing Comparison", "Multi-Doc Answers"]
    )

    # Conditional UI for the chatbot
    if selected_module == "Drawing Chatbot":
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

# --- Main Area ---

# ==============================================================================
# MODULE 1: DRAWING CHATBOT
# ==============================================================================
if selected_module == "Drawing Chatbot":
    # Initialize session state for chatbot
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("🎨 Drawing Chatbot")
    st.markdown("Upload a PDF and ask questions about any page.")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="chatbot_uploader")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        pdf_info = get_pdf_info(tmp_file_path)
        total_pages = pdf_info["num_pages"]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ PDF uploaded: {uploaded_file.name}")
        with col2:
            page_number = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1
            )
        st.info(f"Total pages: {total_pages}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about the drawing..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()


                async def run_chatbot():
                    full_content = ""
                    async for status, content in drawing_chatbot(
                            page_no=page_number,
                            user_query=prompt,
                            pdf_file_path=tmp_file_path
                    ):
                        if status == "streaming":
                            full_content += content
                            message_placeholder.markdown(full_content + "▌")
                        elif status == "complete":
                            full_content = content
                            message_placeholder.markdown(full_content)
                            return full_content
                        elif status == "error":
                            st.error(content)
                            return None


                full_response = asyncio.run(run_chatbot())
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
    else:
        st.info("👆 Please upload a PDF file to start")

# ==============================================================================
# MODULE 2: EXTRACTION REVIEW
# ==============================================================================
elif selected_module == "Extraction Review":
    st.title("📊 Extraction Review")
    st.markdown("Upload a PDF file to automatically extract review tables and download them as an Excel file.")

    uploaded_file = st.file_uploader("Choose a PDF file for extraction", type="pdf", key="extraction_uploader")

    # This block runs ONLY when a new file is uploaded
    if uploaded_file is not None and st.session_state.get("processed_file_name") != uploaded_file.name:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with st.status("Processing PDF...") as status:
            def progress_handler(message):
                status.update(label=message)


            try:
                # 1. Run the extraction
                result = asyncio.run(extraction_review(
                    pdf_file_path=tmp_file_path,
                    progress_callback=progress_handler
                ))
                status.update(label="Extraction complete!", state="complete")

                # 2. Read file bytes into memory
                with open(result['excel_file_path'], "rb") as f:
                    excel_bytes = f.read()

                # 3. Store the essential data (NOT file paths) in session state
                st.session_state.extraction_df = result['df']
                st.session_state.extraction_excel_bytes = excel_bytes
                st.session_state.processed_file_name = uploaded_file.name

                # 4. Clean up the temp directory immediately after reading the file
                cleanup_temp_directory(result['temp_dir'])

            except Exception as e:
                st.error(f"An error occurred during extraction: {e}")
                st.session_state.processed_file_name = None  # Clear state on error
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

    # This block displays the results if they exist in the session state
    if st.session_state.get("extraction_excel_bytes"):
        st.success("✅ Extraction finished successfully!")

        st.markdown("### Extracted Data Preview (First 5 Rows)")
        st.dataframe(st.session_state.extraction_df.head(5))

        # The download button now uses the bytes directly from the session state
        st.download_button(
            label="📥 Download Excel File",
            data=st.session_state.extraction_excel_bytes,
            file_name=f"{st.session_state.processed_file_name.replace('.pdf', '')}_review_extraction.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    elif uploaded_file is None:
        st.info("👆 Please upload a PDF file to begin the extraction process.")

# ==============================================================================
# MODULE 3: EXCEL AUTO COMPLETE
# ==============================================================================
elif selected_module == "Excel Auto Complete":
    st.title("🤖 Excel Auto Complete")
    st.markdown("Upload an Excel file (`.xlsx`) to automatically fill the legal review column using AI.")

    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type="xlsx",
        key="excel_uploader"
    )

    # This block runs ONLY when a new file is uploaded
    if uploaded_file is not None and st.session_state.get("processed_excel_file") != uploaded_file.name:

        # Write uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with st.status("Processing Excel file...") as status:
            def progress_handler(message):
                status.update(label=message)


            try:
                # 1. Run the main auto-complete function
                result = asyncio.run(auto_complete(
                    filepath=tmp_file_path,
                    progress_callback=progress_handler
                ))
                status.update(label="Processing complete!", state="complete")

                # 2. Read the final excel file bytes into memory
                with open(result['excel_file_path'], "rb") as f:
                    excel_bytes = f.read()

                # 3. Store essential data in session state to persist it
                st.session_state.autocomplete_df = result['df']
                st.session_state.autocomplete_excel_bytes = excel_bytes
                st.session_state.processed_excel_file = uploaded_file.name

                # 4. Clean up the temp directory immediately
                cleanup_temp_directory_excel(result['temp_dir'])

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.session_state.processed_excel_file = None  # Clear state on error
            finally:
                # Clean up the initial temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

    # This block displays results if they exist in the session state
    if st.session_state.get("autocomplete_excel_bytes"):
        st.success("✅ Excel file has been processed successfully!")

        st.markdown("### Processed Data Preview (First 5 Rows)")
        st.dataframe(st.session_state.autocomplete_df.head(5))

        # The download button uses the bytes directly from the session state
        st.download_button(
            label="📥 Download Completed Excel File",
            data=st.session_state.autocomplete_excel_bytes,
            file_name=f"{st.session_state.processed_excel_file.replace('.xlsx', '')}_completed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    elif uploaded_file is None:
        st.info("👆 Please upload an Excel file to begin.")

# ==============================================================================
# MODULE 4: FINISHING COMPARISON
# ==============================================================================
elif selected_module == "Finishing Comparison":
    st.title("🔍 Finishing Material Comparison")
    st.markdown(
        "Upload two PDF files (Drawing and Contract) to compare finishing materials and generate an Excel report.")

    # Create two columns for file uploads
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Drawing PDF")
        drawing_file = st.file_uploader(
            "Choose Drawing PDF file",
            type="pdf",
            key="drawing_uploader"
        )

    with col2:
        st.markdown("### Contract PDF")
        contract_file = st.file_uploader(
            "Choose Contract PDF file",
            type="pdf",
            key="contract_uploader"
        )

    # Only process when both files are uploaded and they're new files
    if drawing_file is not None and contract_file is not None:
        # Check if these are new files
        if (st.session_state.get("comparison_drawing_file") != drawing_file.name or
                st.session_state.get("comparison_contract_file") != contract_file.name):

            # Create temporary files for both PDFs
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_drawing:
                tmp_drawing.write(drawing_file.getvalue())
                drawing_path = tmp_drawing.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_contract:
                tmp_contract.write(contract_file.getvalue())
                contract_path = tmp_contract.name

            with st.status("Processing PDFs for comparison...") as status:
                def progress_handler(message):
                    status.update(label=message)


                try:
                    # Run the comparison
                    result = asyncio.run(finishing_material_comparison(
                        drawing_pdf_path=drawing_path,
                        contract_pdf_path=contract_path,
                        progress_callback=progress_handler
                    ))
                    status.update(label="Comparison complete!", state="complete")

                    # Read file bytes into memory
                    with open(result['excel_file_path'], "rb") as f:
                        excel_bytes = f.read()

                    # Store data in session state
                    st.session_state.comparison_df = result['df']
                    st.session_state.comparison_excel_bytes = excel_bytes
                    st.session_state.comparison_drawing_file = drawing_file.name
                    st.session_state.comparison_contract_file = contract_file.name

                    # Clean up temp directories
                    cleanup_temp_directories_finishing(result['temp_dir'], result['temp_dir2'])

                except Exception as e:
                    st.error(f"An error occurred during comparison: {e}")
                    # Clear state on error
                    st.session_state.comparison_drawing_file = None
                    st.session_state.comparison_contract_file = None
                finally:
                    # Clean up the temporary PDF files
                    if os.path.exists(drawing_path):
                        os.unlink(drawing_path)
                    if os.path.exists(contract_path):
                        os.unlink(contract_path)

    # Display results if they exist in session state
    if st.session_state.get("comparison_excel_bytes"):
        st.success("✅ Comparison finished successfully!")

        st.markdown("### Comparison Results Preview (First 5 Rows)")
        st.dataframe(st.session_state.comparison_df.head(5))

        # Download button
        st.download_button(
            label="📥 Download Comparison Excel Report",
            data=st.session_state.comparison_excel_bytes,
            file_name=f"finishing_material_comparison_{st.session_state.comparison_drawing_file.replace('.pdf', '')}_{st.session_state.comparison_contract_file.replace('.pdf', '')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    elif drawing_file is None or contract_file is None:
        st.info("👆 Please upload both Drawing and Contract PDF files to begin the comparison.")

# ==============================================================================
# MODULE 5: MULTI-DOC ANSWERS
# ==============================================================================
elif selected_module == "Multi-Doc Answers":
    st.title("📚 Multi-Document Q&A")
    st.markdown("Upload multiple documents (PDF, DOCX, or XLSX) and ask a question to get answers from each document.")

    # Info box about supported file types
    st.info("🔍 **Supported file types:** PDF, DOCX, XLSX, XLS")

    # File uploader for multiple documents
    uploaded_files = st.file_uploader(
        "Choose documents (PDF, DOCX, or XLSX files)",
        type=["pdf", "docx", "xlsx", "xls"],
        accept_multiple_files=True,
        key="multidoc_uploader"
    )

    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="What would you like to know about these documents?",
        key="multidoc_query"
    )

    # Process button
    if st.button("Submit", disabled=(not uploaded_files or not query)):
        # Check if this is a new set of files or query
        file_names = [f.name for f in uploaded_files]
        if (st.session_state.get("multidoc_files_processed") != file_names or
                st.session_state.get("multidoc_query_processed") != query):

            # Create temporary files for all PDFs
            temp_files = []
            try:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False,
                                                     suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_files.append((tmp_file.name, uploaded_file.name))

                with st.status("Processing documents...") as status:
                    def progress_handler(message):
                        status.update(label=message)


                    try:
                        # Run the multi-doc query
                        results = asyncio.run(multi_doc_query(
                            files=temp_files,
                            query=query,
                            progress_callback=progress_handler
                        ))
                        status.update(label="Analysis complete!", state="complete")

                        # Store results in session state
                        st.session_state.multidoc_results = results
                        st.session_state.multidoc_files_processed = file_names
                        st.session_state.multidoc_query_processed = query

                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                        # Clear state on error
                        st.session_state.multidoc_files_processed = None
                        st.session_state.multidoc_query_processed = None

            finally:
                # Clean up all temporary files
                for temp_path, _ in temp_files:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

    # Display results if they exist in session state
    if st.session_state.get("multidoc_results"):
        results = st.session_state.multidoc_results

        st.markdown("---")
        st.markdown(f"### 🔍 Query: *{results['query']}*")
        st.markdown(f"**Analyzed {results['num_documents']} documents**")

        # Display answers from each document
        for i, answer in enumerate(results['answers'], 1):
            with st.expander(f"📄 {answer['doc_name']}", expanded=True):
                st.markdown(answer['doc_answer'])

        # Option to download results as JSON
        results_json = json.dumps(results, indent=2)
        st.download_button(
            label="📥 Download Results (JSON)",
            data=results_json,
            file_name="multi_doc_answers.json",
            mime="application/json"
        )

    elif not uploaded_files:
        st.info("👆 Please upload one or more documents (PDF, DOCX, or XLSX) to begin.")

    elif not query:
        st.info("✍️ Please enter a question to ask about the documents.")