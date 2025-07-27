import streamlit as st
import joblib
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="🧠 Sentiment Analyzer Pro",
    page_icon="💌",
    layout="wide",  # Use the full page width
    initial_sidebar_state="expanded"
)

# --- 2. Helper Function to Load Artifacts ---
# This function is cached so the app doesn't reload these files on every interaction
@st.cache_resource
def load_artifacts():
    """Load model, mapping, performance metrics, and the original dataset."""
    model_path = './model.pkl'
    label_map_path = './label_mapping.pkl'
    metrics_path = './performance_metrics.json'
    dataset_path = './sentiment_dataset_50k.csv'

    # Check if all required files exist
    if not all(os.path.exists(p) for p in [model_path, label_map_path, metrics_path, dataset_path]):
        st.error(
            "Missing necessary files. Please run `train_model_upgraded.py` first to generate "
            "`model.pkl`, `label_mapping.pkl`, `performance_metrics.json`, and `confusion_matrix.png`."
        )
        st.stop()
        
    model = joblib.load(model_path)
    label_mapping = joblib.load(label_map_path)
    
    with open(metrics_path, 'r') as f:
        performance_metrics = json.load(f)
        
    df_train = pd.read_csv(dataset_path)
    
    return model, label_mapping, performance_metrics, df_train

# Load all necessary files at the start
try:
    model, label_mapping, performance_metrics, df_train = load_artifacts()
    files_loaded = True
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    files_loaded = False

# --- 3. Sidebar UI ---
with st.sidebar:
    st.title("📊 Model Insights")
    st.markdown("Details about the model and its training data.")

    if files_loaded:
        # Training Data Insights
        st.subheader("Training Data Distribution")
        sentiment_counts = df_train['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        st.write("The chart shows the number of samples for each sentiment in the dataset used to train the model.")

        # Model Performance Summary
        st.subheader("Model Performance")
        accuracy = performance_metrics.get('accuracy', 0)
        st.info(f"**Overall Accuracy:** `{accuracy:.2%}`")
        st.write("This score represents the model's accuracy on the unseen test dataset.")
    
    st.markdown("---")
    st.write("App developed by an Awesome Coder!")

# --- 4. Main Page UI ---
st.title("💬 Moodify Pro: Advanced Sentiment Analyzer")
st.markdown("Analyze text sentiment with detailed insights. Built with a `TfidfVectorizer` and a `Multinomial Naive Bayes` classifier.")

if files_loaded:
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🚀 Analyze Single Text", "📄 Analyze File", "⚙️ How It Works"])

    # --- Single Text Analysis Tab ---
    with tab1:
        st.header("Analyze a Single Piece of Text")
        
        user_input = st.text_area(
            "Enter the text below:", 
            height=170,
            placeholder="e.g., 'This product is fantastic and exceeded all my expectations!'"
        )

        if st.button("✨ Analyze Sentiment", type="primary", use_container_width=True):
            if user_input.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner('Analyzing...'):
                    # Predict probabilities to get confidence scores
                    probabilities = model.predict_proba([user_input.strip()])[0]
                    predicted_index = probabilities.argmax()
                    confidence = probabilities[predicted_index]
                    sentiment_label = label_mapping[predicted_index]

                    # --- Display Results in a visually appealing way ---
                    st.markdown("---")
                    st.subheader("🪞 Analysis Result")
                    
                    col1, col2 = st.columns(2)

                    with col1:
                        with st.container(border=True):
                            # Use st.metric for a polished look
                            delta_text = f"{confidence:.1%} Confidence"
                            st.metric(
                                label=f"Sentiment: {sentiment_label}",
                                value=user_input[:50] + "..." if len(user_input) > 50 else user_input,
                                delta=delta_text,
                                delta_color="off" # We use the text color to show sentiment
                            )

                    with col2:
                        with st.container(border=True):
                            st.write("##### Confidence Distribution")
                            prob_df = pd.DataFrame({
                                'Sentiment': [label_mapping[i] for i in range(len(probabilities))],
                                'Probability': probabilities
                            }).set_index('Sentiment')
                            
                            st.bar_chart(prob_df, y='Probability', color=["#00aaff"]) # Custom color
    
    # --- Batch File Analysis Tab ---
    with tab2:
        st.header("Analyze a File (CSV, TXT, or XLSX)")
        st.markdown(
            "Upload a file, select the column containing text, and the app will predict the sentiment for each row."
        )
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "xlsx"])
        
        if uploaded_file is not None:
            try:
                # Determine file type and read it into a DataFrame
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                
                if file_extension == ".csv":
                    df_upload = pd.read_csv(uploaded_file)
                elif file_extension == ".xlsx":
                    df_upload = pd.read_excel(uploaded_file)
                elif file_extension == ".txt":
                    lines = [line.decode('utf-8').strip() for line in uploaded_file.readlines()]
                    df_upload = pd.DataFrame(lines, columns=['text'])
                else:
                    st.error("Unsupported file type. Please upload a CSV, TXT, or XLSX file.")
                    st.stop()

                st.write("##### File Preview")
                st.dataframe(df_upload.head())

                # --- CHANGE HERE: Automatically find text columns ---
                # This makes the app smarter and prevents users from selecting non-text columns.
                text_like_columns = df_upload.select_dtypes(include=['object', 'string']).columns.tolist()

                if not text_like_columns:
                    st.warning("No text-based columns found in the uploaded file. Please check your file.")
                    st.stop()
                
                # Ask user to select the column with text from the filtered list
                text_column = st.selectbox(
                    "Select the column with text to analyze:", 
                    options=text_like_columns,
                    index=None,
                    placeholder="Choose a text column..."
                )

                if text_column:
                    if st.button("📊 Analyze File", use_container_width=True, type="primary"):
                        with st.spinner('Analyzing all rows... This may take a moment.'):
                            # Ensure the column is treated as string and handle potential missing values
                            texts_to_analyze = df_upload[text_column].fillna('').astype(str)
                            
                            # Get predictions
                            predictions_numeric = model.predict(texts_to_analyze)
                            
                            # Map numeric labels back to string labels
                            df_upload['predicted_sentiment'] = [label_mapping[p] for p in predictions_numeric]
                            
                            st.success("Analysis Complete!")
                            st.write("##### Results with Predicted Sentiment")
                            st.dataframe(df_upload)
                            
                            # --- Provide download link for the results ---
                            @st.cache_data
                            def convert_df_to_csv(df):
                                # IMPORTANT: Cache the conversion to prevent re-running on every page interaction.
                                return df.to_csv(index=False).encode('utf-8')

                            csv_to_download = convert_df_to_csv(df_upload)

                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_to_download,
                                file_name='sentiment_analysis_results.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")


    # --- How It Works Tab ---
    with tab3:
        st.header("Behind the Scenes: The Model")
        
        st.markdown(
            """
            This application uses a classic machine learning pipeline for text classification. 
            Here’s a breakdown of the components:
            - **TfidfVectorizer**: This converts raw text into numerical features. Unlike simple counting, it gives higher weight to words that are more unique and important to a specific document.
            - **Multinomial Naive Bayes**: A simple yet powerful probabilistic algorithm that's well-suited for text classification tasks. It calculates the probability of a text belonging to each sentiment class and picks the one with the highest probability.
            - **GridSearchCV**: We used this to automatically tune the model's hyperparameters, ensuring we found the best-performing combination for our specific dataset.
            """
        )
        st.markdown("---")
        
        st.subheader("Model Evaluation Metrics")
        st.write("These metrics were calculated on a hidden test set (20% of the original data) to provide an unbiased evaluation of the model's performance.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.write("##### Classification Report")
                report_df = pd.DataFrame(performance_metrics['classification_report']).transpose()
                st.dataframe(report_df.style.format("{:.3f}"))
                st.caption(
                    """
                    - **Precision**: Of all the predictions for a sentiment, how many were correct?
                    - **Recall**: Of all the actual instances of a sentiment, how many did the model correctly identify?
                    - **F1-Score**: The harmonic mean of Precision and Recall. A good overall measure.
                    """
                )

        with col2:
            with st.container(border=True):
                st.write("##### Confusion Matrix")
                if os.path.exists('confusion_matrix.png'):
                    st.image('confusion_matrix.png', use_container_width=True)
                else:
                    st.warning("Confusion matrix plot (`confusion_matrix.png`) not found.")
                st.caption("This shows where the model got things right and wrong. The diagonal (top-left to bottom-right) represents correct predictions.")
if __name__ == "__main__":
    app.run(debug=True)
    
    
  

