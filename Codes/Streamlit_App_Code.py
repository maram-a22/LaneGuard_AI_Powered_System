import streamlit as st
import cv2
import tempfile
import os
import csv
from datetime import datetime, timedelta
from ultralytics import YOLO
import pandas as pd
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.express as px
import random
import requests




# Set page config
st.set_page_config(layout="wide", page_title="LaneGuard - AI Traffic Management")

# Define lane lines 
roi_rect_top_left = (250, 200)
roi_rect_bottom_right = (1670, 1000)

roi1_line = [(370, 1000), (860, 250)]
roi2_line = [(697, 1000), (955, 250)]
roi3_line = [(1050, 1000), (1050, 250)]
roi4_line = [(1415, 1000), (1140, 250)]

def draw_lanes_and_roi(frame, roi_rect_top_left, roi_rect_bottom_right):
    cv2.line(frame, roi1_line[0], roi1_line[1], color=(255, 255, 255), thickness=3)
    cv2.line(frame, roi2_line[0], roi2_line[1], color=(255, 255, 255), thickness=3)
    cv2.line(frame, roi3_line[0], roi3_line[1], color=(255, 255, 255), thickness=3)
    cv2.line(frame, roi4_line[0], roi4_line[1], color=(255, 255, 255), thickness=3)
    cv2.rectangle(frame, roi_rect_top_left, roi_rect_bottom_right, (128, 128, 128), 3)
    return frame

def get_lane(center_x, center_y):
    if center_x < roi1_line[0][0] + (roi1_line[1][0] - roi1_line[0][0]) * (1000 - center_y) / 750:
        return 1
    elif center_x < roi2_line[0][0] + (roi2_line[1][0] - roi2_line[0][0]) * (1000 - center_y) / 750:
        return 2
    elif center_x < roi3_line[0][0] + (roi3_line[1][0] - roi3_line[0][0]) * (1000 - center_y) / 750:
        return 3
    elif center_x < roi4_line[0][0] + (roi4_line[1][0] - roi4_line[0][0]) * (1000 - center_y) / 750:
        return 4
    else:
        return 5


@st.cache_resource
def load_model():
    url = 'https://raw.githubusercontent.com/Shahadfaiz/LaneGuard_AI_Powered_System/main/Code/best.pt'
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        try:
            model = YOLO(tmp_file_path)
            os.unlink(tmp_file_path)  # Delete the temporary file
            return model
        except Exception as e:
            os.unlink(tmp_file_path)  # Ensure the temp file is deleted even if loading fails
            raise e
    else:
        raise Exception(f"Failed to download the model file. Status code: {response.status_code}")

def analysis_page():
    col1, col2, col3 = st.columns(3) # Create three columns
    
    with col2: # Use the middle column to display the image
        st.image("https://raw.githubusercontent.com/Shahadfaiz/LaneGuard_AI_Powered_System/main/Code/logo_white_new.png", width=300)

    st.title("LaneGuard: AI-Powered Lane-Switching Violation Detection System")
    st.markdown("""
    LaneGuard is an innovative solution designed to combat traffic congestion caused by unnecessary lane changes during peak hours. 
    By leveraging state-of-the-art YOLOv8 object detection and advanced lane tracking algorithms, our system provides real-time 
    monitoring and violation detection to improve highway efficiency.

    ### Key Features:
    • Real-time vehicle tracking
    • Instant violation alerts
    • Traffic flow analytics

    ### How It Works:
    1. 📹 Upload traffic video
    2. 🚗 AI detects vehicles and lanes
    3. 🚦 System flags violations
    4. 📊 Get instant insights


    Ready to see LaneGuard in action? upload your highway video below!
    """)
    
    model = load_model()

    if 'current_video' not in st.session_state:
        st.session_state.current_video = None

    uploaded_file = st.file_uploader("Upload highway traffic video (MP4, AVI, MOV)", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None and uploaded_file != st.session_state.current_video:
        st.session_state.current_video = uploaded_file
        st.success("Video uploaded successfully. Initiating LaneGuard analysis...")
        
        # Process video and store results
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        current_vehicles = {}
        vehicle_count = 0
        violation_count = 0
        vehicle_last_lane = {}
        vehicle_violations = {}
        total_count = 0  # Initialize total_count

        t_outfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out = cv2.VideoWriter(t_outfile.name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(cap.get(3)), int(cap.get(4))))

        csv_data = []
        start_time = datetime(2023, 1, 1, 8, 0, 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        street_name = "Olaya Road"
        latitude = 24.7136
        longitude = 46.6753
        timestamps = [start_time + timedelta(minutes=2 * i) for i in range(frame_count)]

        progress_bar = st.progress(0)
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, show=False)
            tracks = results[0].boxes

            new_vehicles = {}
            for track in tracks:
                x1, y1, x2, y2 = track.xyxy[0]
                track_id = int(track.id[0])
                conf = track.conf[0]
                cls = track.cls[0]

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if (roi_rect_top_left[0] < center_x < roi_rect_bottom_right[0] and
                    roi_rect_top_left[1] < center_y < roi_rect_bottom_right[1]):
                    
                    new_vehicles[track_id] = True
                    if track_id not in current_vehicles:
                        vehicle_count += 1
                        total_count += 1  # Increment total_count for each new vehicle

                    current_lane = get_lane(center_x, center_y)

                    if track_id in vehicle_last_lane:
                        last_lane = vehicle_last_lane[track_id]
                        if last_lane != current_lane:
                            if track_id not in vehicle_violations:
                                violation_count += 1
                                vehicle_violations[track_id] = True
                            box_color = (0, 0, 255)
                            violation_text = "VIOLATION"
                        else:
                            box_color = (0, 255, 0)
                            violation_text = ""
                    else:
                        box_color = (0, 255, 0)
                        violation_text = ""

                    vehicle_last_lane[track_id] = current_lane

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                    cv2.putText(frame, f'Vehicle ID:{track_id} Conf:{conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
                    cv2.circle(frame, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

                    if violation_text:
                        cv2.putText(frame, violation_text, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

            current_vehicles = new_vehicles
            frame = draw_lanes_and_roi(frame, roi_rect_top_left, roi_rect_bottom_right)

            cv2.putText(frame, f'Current Numbers of Vehicles: {len(current_vehicles)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, f'Numbers of Violations: {violation_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            out.write(frame)
            current_time = timestamps[i]
            

            csv_data.append({
                'Timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Current_Vehicles_in_ROI': len(current_vehicles),
                'Violation_Count': violation_count,
                'Street_Name': street_name,
                'Latitude': latitude,
                'Longitude': longitude,
                'Hour_of_Day': current_time.hour,
                'Day_of_Week': current_time.strftime('%A'),
                'Total_Count': total_count  # Add total_count to CSV data
            })

            progress_bar.progress((i + 1) / frame_count)

        cap.release()
        out.release()

        os.remove(tfile.name)

        # Save processed video and CSV data to session state
        st.session_state['processed_video'] = t_outfile.name
        st.session_state['traffic_data'] = pd.DataFrame(csv_data)

        st.write("Vehicle Detection and Violation Tracking complete. You can now view and download the processed video.")
        st.video(t_outfile.name)

        # Add the "Download Processed Video" button right after the video
        with open(t_outfile.name, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="Processed Video.mp4",
                mime="video/mp4"
            )

        csv_filename = 'LaneGuard Traffic Data.csv'
        st.session_state['traffic_data'].to_csv(csv_filename, index=False)
        
        st.write("Sample data from CSV file:")
        st.dataframe(st.session_state['traffic_data'].head(10))

        # Add the "Download CSV Data" button after the CSV data
        with open(csv_filename, "rb") as file:
            st.download_button(
                label="Download CSV Data",
                data=file,
                file_name=csv_filename,
                mime="text/csv"
            )

    elif 'processed_video' in st.session_state and 'traffic_data' in st.session_state:
        st.write("Previously processed video:")
        st.video(st.session_state['processed_video'])

        # Add the "Download Processed Video" button right after the video
        with open(st.session_state['processed_video'], "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="Processed Video.mp4",
                mime="video/mp4"
            )

        st.write("Sample data from previously processed CSV file:")
        st.dataframe(st.session_state['traffic_data'].head(10))

        csv_filename = 'LaneGuard Traffic Data.csv'
        # Add the "Download CSV Data" button after the CSV data
        with open(csv_filename, "rb") as file:
            st.download_button(
                label="Download CSV Data",
                data=file,
                file_name=csv_filename,
                mime="text/csv"
            )

    else:
        st.info("Please upload a video to begin analysis.")
        
    st.markdown("""
    ---
    ### Impact
    
    LaneGuard aims to significantly reduce traffic congestion by:
    - Minimizing unnecessary lane changes during peak hours
    - Providing data-driven insights for traffic flow optimization
    - Enhancing overall highway safety and efficiency
    
    Our system is designed for traffic management authorities, urban planners, and researchers 
    committed to creating smarter, more efficient transportation networks.
    """)
        
        
def dashboard_page():
    df = st.session_state.get('traffic_data')
    
    if df is None:
        st.error("No traffic data available. Please run the analysis first.")
        return

    # Ensure the Timestamp column is in datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Fix: Convert min/max date to datetime.date() for use in date_input
    date_range = st.sidebar.date_input(
        "Date Range", 
        [df['Timestamp'].min().date(), df['Timestamp'].max().date()]
    )
    
    hour_range = st.sidebar.slider("Hour Range", 0, 23, (0, 23))
    days = st.sidebar.multiselect("Days of Week", df['Day_of_Week'].unique(), default=df['Day_of_Week'].unique())

    # Apply filters
    filtered_df = df[
        (df['Timestamp'].dt.date >= date_range[0]) &
        (df['Timestamp'].dt.date <= date_range[1]) &
        (df['Timestamp'].dt.hour >= hour_range[0]) &
        (df['Day_of_Week'].isin(days))
    ]

    # Main layout structure
    st.title("Riyadh Traffic Violations Dashboard")

    # Create layout structure with three main areas
    col1, col2, col3 = st.columns([1, 3, 1])

    # Left column for Gains/Losses style stats (Current Vehicles and Violation Count)
    with col1:
        st.markdown("### Traffic Alerts")
        
        # Calculate the violation percentage
        violation_percentage = (filtered_df['Violation_Count'].max() / filtered_df['Total_Count'].max()) * 100

        # "Gains/Losses" style for "Current Vehicles in ROI" and "Violation Percentage"
        st.markdown("""
        <div style='background-color: #333; padding: 10px; margin-bottom: 10px; border-radius: 8px; text-align: center;'>
            <h4 style='color: white; margin-bottom: 5px; padding-left: 10px; text-align: center;'>Total Vehicles</h4>
            <p style='font-size: 22px; color: #00FF00; margin-top: 0px;'>{}</p>
        </div>
        <div style='background-color: #333; padding: 10px; margin-bottom: 10px; border-radius: 8px; text-align: center;'>
            <h4 style='color: white; margin-bottom: 5px; padding-left: 10px; text-align: center;'>Violation Percentage</h4>
            <p style='font-size: 22px; color: #FF0000; margin-top: 0px;'>{:.2f}%</p>
        </div>
        """.format(filtered_df['Total_Count'].max(), violation_percentage), unsafe_allow_html=True)
                
        st.markdown("### Processed Video")
        # Display processed video
        if 'processed_video' in st.session_state:
            st.video(st.session_state['processed_video'])
        else:
            st.error("No processed video available.")

    
    # Center column for the map and charts
    with col2:
        # Main map
        fig = go.Figure(go.Scattermapbox(
            lat=filtered_df['Latitude'],
            lon=filtered_df['Longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(size=10),
            text=filtered_df['Street_Name']
        ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=24.7136, lon=46.6753),  # Center of Riyadh
                zoom=10
            ),
            showlegend=False,
            height=400,
            margin={"r":0,"t":55,"l":0,"b":0}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        

        # Add the two charts: Vehicles over time and Violations over time
        col21, col22 = st.columns(2)

    with col21:
        fig_vehicles = px.line(filtered_df, x='Timestamp', y='Total_Count', title='Vehicles Over Time')
        fig_vehicles.update_layout(
            height=200, 
            margin=dict(t=30, b=50),  # Increased bottom margin to accommodate rotated labels
            xaxis=dict(
                type='date',
                tickformat='%H:%M',
                dtick=3600000,  # show ticks every hour
                tickangle=45,  # Rotate labels by 45 degrees
                tickmode='linear'
            )
        )
        st.plotly_chart(fig_vehicles, use_container_width=True)

    with col22:
        fig_violations = px.line(filtered_df, x='Timestamp', y='Violation_Count', title='Violations Over Time')
        fig_violations.update_traces(line_color='red')  # Set the line color to red
        fig_violations.update_layout(
            height=200, 
            margin=dict(t=30, b=50),
            xaxis=dict(
                type='date',
                tickformat='%H:%M',
                dtick=3600000,
                tickangle=45,
                tickmode='linear'
            )
        )
        st.plotly_chart(fig_violations, use_container_width=True)



    # Right column for circular progress (Violation Percentage) and video
    with col3:
        # About section at the bottom with additional top padding to shift it down
        st.markdown("""
        <div style='background-color: #333; padding: 30px; border-radius: 10px; min-height: 400px; display: flex; flex-direction: column; justify-content: space-between; margin-top: 53px;'>
            <div>
                <h4 style='color: white; margin-bottom: 20px;'>About</h4>
                <p style='color: white; margin-bottom: 15px; justify-content: space-between;'>The system aims to improve road safety and optimize traffic flow by identifying and analyzing traffic patterns and violations.</p>
                <p style='color: white; margin-bottom: 15px; justify-content: space-between;'>Data is processed using the LaneGuard AI system, which detects lane-switching violations during peak traffic hours.</p>
            </div>
            <div>
                <p style='color: #888; font-size: 0.9em; margin-top: 20px;'>For more information, contact us at <a href="mailto:inquiries.laneguard@outlook.com" style="color: #ADD8E6; text-decoration: none;">inquiries.laneguard@outlook.com</a>.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


    

# Navigation options
st.sidebar.title("LaneGuard Navigation")
page = st.sidebar.radio("Go to", ["Analysis", "Dashboard"])

if page == "Analysis":
    analysis_page()
elif page == "Dashboard":
    dashboard_page()
