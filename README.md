# [IAARooms.streamlit.app](https://IAARooms.streamlit.app)

## **Overview**
The **IAA Conference Room Analysis App** is a data-driven Streamlit application designed to analyze booking patterns for conference rooms and provide detailed insights. The app includes statistics on room usage, student activity, time-based trends, and clustering analyses.

Data consists of bookings from the Institute for Advanced Analytics, NC State University, Class of 2025. Bookings are defined as 30-minute time slots, as established on the internal reservation site. (This is a student-made project and not affiliated with the IAA or NC State University.)

This repository contains the source code for the app, along with the necessary data files required for the analyses.

## **Features**
- **Room Statistics**:
  - Visualize the most booked rooms and booking proportions by floor.
  - Identify the top bookers for each room.
- **Student Statistics**:
  - Award students based on booking patterns (e.g., Early Bird Award, Late Night Warrior).
  - Explore detailed booking activity for individual students.
- **Time-Based Statistics**:
  - Analyze booking trends by time slots and days of the week.
  - Compare cumulative bookings between floors over time.
- **Cluster Analysis**:
  - Visualize student booking patterns using PCA and K-Nearest Neighbors clustering.

## **Data Files**
1. **`data/confrooms.xlsx`**:
   - Contains booking information for conference rooms.
   - **Columns**:
     - `Name`: Name of the student who booked the room.
     - `Room`: Name of the conference room.
     - `Time`: Time slot for the booking.
     - `Date`: Date of the booking.
     - Additional columns for detailed analysis (e.g., `Floor`, `DayOfWeek`).

2. **`data/student_headshots.csv`**:
   - Contains URLs to headshots of students in the dataset.
   - **Columns**:
     - `Name`: Full name of the student.
     - `Headshot URL`: Publicly accessible link to the student’s headshot.

## **App Structure**
- **Frontend**:
  - Developed using Streamlit for interactive visualizations and user-friendly navigation.
  - Side navigation menu allows access to different analysis sections.
- **Backend**:
  - Python-based data processing using `pandas`, `numpy`, and `scikit-learn`.
  - Draft visualizations created with `matplotlib` and `seaborn`, then revised for Streamlit using `plotly`.

## **Dependencies**
Ensure the following packages are installed (specified in `requirements.txt`):
- `streamlit`
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `plotly`
- `scikit-learn`
- `openpyxl`
- `fsspec`

Install dependencies using:
```bash
pip install -r requirements.txt
