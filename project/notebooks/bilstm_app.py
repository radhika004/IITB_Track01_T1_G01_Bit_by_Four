# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import RobustScaler # Make sure RobustScaler is imported

# # --- Model & Scaler Loading ---
# # Use st.cache_resource to load the model and scaler only once
# @st.cache_resource
# def load_prediction_model(model_path):
#     """Loads the trained BiLSTM model"""
#     try:
#         # Load model with compile=False to fix the 'mse' deserialization error
#         model = load_model(model_path, compile=False)
#         return model
#     except Exception as e:
#         st.error(f"Error loading BiLSTM model: {e}")
#         return None

# @st.cache_resource
# def create_data_scaler(data_path):
#     """
#     Loads the original training data to fit a new scaler.
#     This is a workaround for not having the saved 'final_scaler.pkl'.
#     """
#     try:
#         df = pd.read_csv(data_path)
#         non_feature_cols = ['QuestionKey', 'Participant', 'ResponseTime', 'Category']
#         features_df = df.drop(columns=non_feature_cols, errors='ignore')
        
#         # Clean data just in case
#         features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
#         features_df.fillna(features_df.median(), inplace=True)
        
#         scaler = RobustScaler()
#         scaler.fit(features_df)
#         st.session_state.scaler_features = features_df.columns.tolist() # Save feature names
#         return scaler
#     except FileNotFoundError:
#         st.error(f"Error: Could not find the original dataset '{data_path}'. This file is required to create the scaler. Please add it to the directory.")
#         return None
#     except Exception as e:
#         st.error(f"Error creating scaler: {e}")
#         return None

# # --- Main App ---
# st.set_page_config(page_title="Response Time Prediction", layout="wide")
# st.title("🧠 Response Time Prediction using BiLSTM")
# st.write("Upload your aggregated features CSV file to predict the response time.")

# # --- Load Model and Scaler ---
# MODEL_PATH = r'C:\Users\ASUS\OneDrive\Desktop\IITB_internship\Final_submission\project\models\bilstm.h5' # Your model name
# # Path to the *original* data file, used to recreate the scaler
# DATA_PATH = r'Final_submission/project/data/EEG_IVT_EYE_final_merged_data.csv' 

# model = load_prediction_model(MODEL_PATH)
# # Instead of loading a .pkl, we create the scaler by fitting to the original data
# scaler = create_data_scaler(DATA_PATH) 

# if model is None or scaler is None:
#     st.error("Model or scaler could not be loaded/created. Please make sure 'bilstm.h5' and 'EEG_IVT_EYE_final_merged_data.csv' are in the correct directory.")
# else:
#     st.success("✅ Prediction engine (BiLSTM Model + Scaler) loaded successfully!")

#     # --- File Uploader ---
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

#     if uploaded_file is not None:
#         try:
#             # Load the new data
#             new_df = pd.read_csv(uploaded_file)
#             st.subheader("Uploaded Data (Preview)")
#             st.dataframe(new_df.head())

#             # --- Preprocessing ---
#             # 1. Identify feature columns
#             # Use the feature names we saved when creating the scaler
#             if 'scaler_features' in st.session_state:
#                 original_features = st.session_state.scaler_features
#             else:
#                 st.error("Scaler features not found. Please reload the app.")
#                 st.stop()

#             try:
#                 features_df = new_df[original_features]
#             except KeyError as e:
#                 st.error(f"Missing required feature: {e}. Please ensure the uploaded CSV contains all necessary columns.")
#                 st.stop()

#             # 2. Clean data (handle inf and NaN)
#             features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
#             # Fill NaNs with the median *from the scaler*
#             if hasattr(scaler, 'center_'):
#                 fill_values = pd.Series(scaler.center_, index=original_features)
#             else:
#                 fill_values = features_df.median()
                
#             features_df.fillna(fill_values, inplace=True)

#             # 3. Scale data
#             scaled_data = scaler.transform(features_df)

#             # 4. Reshape for BiLSTM (This is the crucial step from your notebook)
#             # Shape should be: (samples, timesteps, features)
#             # Your notebook used 1 timestep: (n_samples, 1, n_features)
#             data_reshaped = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))

#             st.info(f"Data preprocessed and reshaped to: {data_reshaped.shape}")

#             # --- Prediction ---
#             if st.button("🚀 Predict Response Times", type="primary"):
#                 with st.spinner("Running BiLSTM model..."):
#                     predictions = model.predict(data_reshaped)
                
#                 st.success("Prediction Complete!")
                
#                 # --- Display Results ---
#                 # Check for 'QuestionKey' and 'Participant' before including them
#                 id_cols = ['QuestionKey', 'Participant']
#                 results_cols = [col for col in id_cols if col in new_df.columns]
#                 results_df = new_df[results_cols].copy()
                
#                 results_df['Predicted_ResponseTime'] = predictions.flatten() # Flatten the prediction output
                
#                 st.subheader("Prediction Results")
#                 st.dataframe(results_df)
                
#                 # Add a download button for the results
#                 @st.cache_data
#                 def convert_df_to_csv(df):
#                     return df.to_csv(index=False).encode('utf-8')

#                 csv = convert_df_to_csv(results_df)
#                 st.download_button(
#                     label="Download Predictions as CSV",
#                     data=csv,
#                     file_name="response_time_predictions.csv",
#                     mime="text/csv",
#                 )

#         except Exception as e:
#             st.error(f"An error occurred during processing: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
import joblib # Using joblib to load/save scaler would be ideal, but we'll recreate it as planned

# --- Model & Data Loading ---

@st.cache_resource
def load_prediction_model(model_path):
    """Loads the trained BiLSTM model"""
    try:
        # Load model with compile=False to fix the 'mse' deserialization error
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading BiLSTM model: {e}")
        return None

@st.cache_resource
def create_data_scaler(data_path):
    """
    Loads the original training data to fit a new scaler.
    This is a workaround for not having the saved 'final_scaler.pkl'.
    """
    try:
        df = pd.read_csv(data_path)
        non_feature_cols = ['QuestionKey', 'Participant', 'ResponseTime', 'Category']
        features_df = df.drop(columns=non_feature_cols, errors='ignore')
        
        # Clean data just in case
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.fillna(features_df.median(), inplace=True)
        
        scaler = RobustScaler()
        scaler.fit(features_df)
        st.session_state.scaler_features = features_df.columns.tolist() # Save feature names
        return scaler
    except FileNotFoundError:
        st.error(f"Error: Could not find the original dataset '{data_path}'. This file is required to create the scaler. Please add it to the directory.")
        return None
    except Exception as e:
        st.error(f"Error creating scaler: {e}")
        return None

# --- Main App ---
st.set_page_config(page_title="Response Time Prediction", layout="wide")
st.title("🧠 Response Time Prediction using BiLSTM")
st.write("Predict response time by uploading a CSV or pasting a single feature row.")

# --- Load Model and Scaler ---
# Using relative paths for better portability
# # --- Load Model and Scaler ---
MODEL_PATH = r'C:\Users\ASUS\OneDrive\Desktop\IITB_internship\Final_submission\project\models\bilstm.h5' # Your model name
# # Path to the *original* data file, used to recreate the scaler
DATA_PATH = r'Final_submission/project/data/EEG_IVT_EYE_final_merged_data.csv'

model = load_prediction_model(MODEL_PATH)
# Instead of loading a .pkl, we create the scaler by fitting to the original data
scaler = create_data_scaler(DATA_PATH) 

if model is None or scaler is None:
    st.error("Model or scaler could not be loaded/created. Please make sure 'bilstm.h5' and 'EEG_IVT_EYE_final_merged_data.csv' are in the correct directory.")
else:
    st.success("✅ Prediction engine (BiLSTM Model + Scaler) loaded successfully!")
    
    # --- Sidebar Input Selection ---
    input_method = st.sidebar.radio(
        "Select Prediction Method",
        ("Upload CSV File", "Paste Feature Row")
    )
    
    # Get feature info from session state
    feature_list = st.session_state.get('scaler_features', [])

    # --- CSV File Upload ---
    if input_method == "Upload CSV File":
        st.header("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                # Load the new data
                new_df = pd.read_csv(uploaded_file)
                st.subheader("Uploaded Data (Preview)")
                st.dataframe(new_df.head())

                # --- Preprocessing ---
                if not feature_list:
                    st.error("Scaler features not found. Please reload the app.")
                    st.stop()

                try:
                    features_df = new_df[feature_list]
                except KeyError as e:
                    st.error(f"Missing required feature: {e}. Please ensure the uploaded CSV contains all necessary columns.")
                    st.stop()

                # 2. Clean data (handle inf and NaN)
                features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                # Fill NaNs with the median *from the scaler*
                if hasattr(scaler, 'center_'):
                    fill_values = pd.Series(scaler.center_, index=feature_list)
                else:
                    fill_values = features_df.median()
                    
                features_df.fillna(fill_values, inplace=True)

                # 3. Scale data
                scaled_data = scaler.transform(features_df)

                # 4. Reshape for BiLSTM (This is the crucial step from your notebook)
                data_reshaped = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))

                st.info(f"Data preprocessed and reshaped to: {data_reshaped.shape}")

                # --- Prediction ---
                if st.button("🚀 Predict Response Times (from CSV)", type="primary"):
                    with st.spinner("Running BiLSTM model..."):
                        predictions = model.predict(data_reshaped)
                    
                    st.success("Prediction Complete!")
                    
                    # --- Display Results ---
                    id_cols = ['QuestionKey', 'Participant']
                    results_cols = [col for col in id_cols if col in new_df.columns]
                    results_df = new_df[results_cols].copy()
                    
                    results_df['Predicted_ResponseTime'] = predictions.flatten() # Flatten the prediction output
                    
                    st.subheader("Prediction Results")
                    st.dataframe(results_df)
                    
                    # Add a download button for the results
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv = convert_df_to_csv(results_df)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="response_time_predictions.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

    # --- Manual Feature Entry (Paste) ---
    elif input_method == "Paste Feature Row":
        st.header("Paste a Single Row of Features")
        
        if not feature_list:
            st.error("Feature list not loaded. Please reload the app.")
            st.stop()
            
        feature_count = len(feature_list)
        
        st.info(f"Please paste a single row of **{feature_count}** comma-separated feature values.")
        
        # Add a text area for pasting the data
        pasted_data = st.text_area("Paste features here:", height=100,
                                   placeholder="0.827, 0.817, 0.275, 0.135, 3.922, 0.614, 0.610, ...")
        
        if st.button("🚀 Predict Response Time (from Paste)", type="primary"):
            if not pasted_data:
                st.warning("Please paste your feature data in the text box.")
                st.stop()
                
            try:
                # --- Preprocessing ---
                # 1. Parse the pasted string
                # Split by comma, strip whitespace, and convert to float
                values = [float(v.strip()) for v in pasted_data.split(',')]
                
                # 2. Check feature count
                if len(values) != feature_count:
                    st.error(f"Error: Expected {feature_count} features, but you pasted {len(values)}.")
                    st.stop()
                
                # 3. Create DataFrame
                features_df = pd.DataFrame([values], columns=feature_list)
                
                # 4. Scale data
                scaled_data = scaler.transform(features_df)
                
                # 5. Reshape for BiLSTM
                data_reshaped = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))

                st.info(f"Data preprocessed and reshaped to: {data_reshaped.shape}")

                # --- Prediction ---
                with st.spinner("Running BiLSTM model..."):
                    prediction = model.predict(data_reshaped)
                
                st.success("Prediction Complete!")
                
                # --- Display Results ---
                result_val = prediction.flatten()[0]
                st.metric(label="Predicted Response Time (seconds)", value=f"{result_val:.2f} s")
                st.balloons()

            except ValueError:
                st.error("Error: Could not parse all values. Please ensure all values are numbers and separated by commas.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

