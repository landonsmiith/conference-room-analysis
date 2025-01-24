import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# Load and preprocess the data
@st.cache_data
def load_data():
    confroom_data = pd.read_excel("data/confrooms.xlsx")
    headshots_data = pd.read_csv("data/student_headshots.csv") 
    
    # Standardize 'Name' columns
    confroom_data['Name'] = confroom_data['Name'].str.strip().str.title()
    headshots_data['Name'] = headshots_data['Name'].str.strip().str.title()
    
    # Merge headshot data to conf room data
    headshots_data.rename(columns={'Address': 'Headshot URL'}, inplace=True)
    merged_data = pd.merge(confroom_data, headshots_data, on='Name', how='left')
    
    # Clean and process the data
    merged_data.columns = merged_data.columns.str.strip()
    merged_data.rename(columns={'Room ': 'Room'}, inplace=True)
    merged_data['Room'] = merged_data['Room'].replace({'Perwinkle': 'Periwinkle'})
    merged_data = merged_data[~merged_data['Room'].isin(['Maple', 'Aspen'])]
    merged_data['Name'] = merged_data['Name'].replace({'Mmburns': 'Madison Burns'})
    merged_data['Time'] = pd.to_datetime(merged_data['Time'], format='%I:%M%p')
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    
    # Feature engineering
    first_floor_rooms = ['Lotus', 'Mallow', 'Periwinkle', 'Primrose', 'Oswego',
                         'Phlox', 'Verbena', 'Wisteria', 'Wyethia']
    merged_data['Floor'] = merged_data['Room'].apply(lambda x: '1' if x in first_floor_rooms else '2')
    merged_data['DayOfWeek'] = merged_data['Date'].dt.day_name()
    
    return merged_data

# Load the data
df = load_data()

# Sidebar Navigation
st.sidebar.title("Conference Room Analysis")
menu = st.sidebar.selectbox("Select Analysis Section", ["Home", "Room Statistics", "Student Statistics", "Time-Based Statistics", "Cluster Analysis"])

# About the Author 
st.sidebar.markdown("---")  
st.sidebar.subheader("About the Author")
st.sidebar.write("""
    Hi! I'm Landon Smith, a data enthusiast who loves turning analytics into stories that anyone can understand and enjoy. Learn more about me and view the code for this project below!
""")

st.sidebar.markdown("""
    - [LinkedIn Profile](https://www.linkedin.com/in/landongsmith)
    - [GitHub Repository](https://github.com/landonsmiith)
""")

# Home Page
if menu == "Home":
    st.title("Welcome!")
    
    st.subheader("Introduction")
    st.write("""
    Welcome to the **IAA Conference Room Analysis App**! This tool provides valuable insights into room bookings 
    and usage trends by NC State University's world-class **Institute for Advanced Analytics (IAA) Class of 2025**.
    
    The data analyzed in this app includes bookings made between **June-December 2024**.
    Explore room popularity, individual booking behaviors, and usage patterns over time.
    """)

    st.subheader("How to Use This App")
    st.write("""
    Use the dropdown menu on the left sidebar to navigate through the different sections:
    
    - **Room Statistics**: Explore which conference rooms were most frequently booked and trends in room usage.
    - **Student Statistics**: Dive into individual statistics, including booking behaviors and awards for 
      standout students, such as the **Most Valuable Booker** or the **Early Bird Award**.
    - **Time-Based Statistics**: Analyze booking trends over time, including cumulative floor-wise bookings 
      and popular days of the week.
    - **Cluster Analysis**: Check out what students are most similar in their booking patterns, thanks to k-nearest neighbors methods.
    """)

    st.subheader("About the Data")
    st.write("""
    The dataset comprises booking information logged by members of the **IAA Class of 2025**. Each booking 
    records the following:
    
    - **Name** of the person making the booking
    - **Room** used
    - **Date** and **time** of the booking

    This analysis helps understand how the conference rooms were utilized during the specified time period.
    """)

elif menu == "Room Statistics":
    st.header("Room Statistics")
    
    # Dropdown to select analysis
    analysis_option = st.selectbox(
        "Choose an analysis to view:",
        ["Most Popular Rooms by Bookings", "Room Popularity by Floor", "First vs. Second Floor Booking Proportions", "Most Frequent Booker By Room"]
    )
    
    if analysis_option == "Most Popular Rooms by Bookings":
        st.subheader("Most Popular Rooms by Bookings")
        
        # Calculate room counts
        room_counts = df['Room'].value_counts().reset_index()
        room_counts.columns = ['Room', 'Counts']
        
        fig = px.bar(
            room_counts,
            x='Room',
            y='Counts',
            labels={'Room': 'Room', 'Counts': 'Number of Bookings'},
            text='Counts'
        )
        fig.update_traces(marker_color='#780606', showlegend=False)
        fig.update_layout(
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=14,
            yaxis_tickfont_size=14,
            xaxis_tickangle=-50
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_option == "Room Popularity by Floor":
        st.subheader("Room Popularity by Floor")
        
        # Calculate room counts and merge floor information
        room_counts = df['Room'].value_counts().reset_index()
        room_counts.columns = ['Room', 'Counts']
        room_floor_counts = pd.merge(room_counts, df[['Room', 'Floor']].drop_duplicates(), on='Room')
        
        # Sort by counts
        room_floor_counts = room_floor_counts.sort_values(by='Counts', ascending=False)
        
        fig = px.bar(
            room_floor_counts,
            x='Room',
            y='Counts',
            color='Floor',
            labels={'Counts': 'Number of Bookings', 'Room': 'Room'},
            text='Counts',
            color_discrete_map={'1': '#3182bd', '2': '#e6550d'} 
        )
        fig.update_layout(
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=14,
            yaxis_tickfont_size=14,
            legend_title_font_size=16,
            xaxis_tickangle=-50
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_option == "First vs. Second Floor Booking Proportions":
        st.subheader("First vs. Second Floor Booking Proportions")
        
        # Aggregate bookings by floor
        floor_counts = df['Floor'].value_counts().reset_index()
        floor_counts.columns = ['Floor', 'Counts']
        
        fig = px.pie(
            floor_counts,
            values='Counts',
            names='Floor',
            color='Floor',
            color_discrete_map={'1': '#3182bd', '2': '#e6550d'} 
        )
        fig.update_traces(
            textinfo='percent+label',
            textfont_size=16,
            marker=dict(line=dict(color='white', width=1))
        )
        fig.update_layout(
            legend_title=dict(text='Floor'),
            legend_title_font_size=18  
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_option == "Most Frequent Booker By Room":
        st.subheader("Most Frequent Booker By Room")
    
        # Calculate the most frequent booker for each room
        most_frequent_bookers = (
        df.groupby(['Room', 'Name'])
        .size()
        .reset_index(name='Counts')
        .sort_values(['Room', 'Counts'], ascending=[True, False])
        .groupby('Room')
        .first()
        .reset_index()
        )

        fig = px.bar(
            most_frequent_bookers,
            x='Room',
            y='Counts',
            color='Name',
            labels={'Counts': 'Number of Bookings', 'Room': 'Room', 'Name': 'Booker'},
            text='Counts'
        )
        fig.update_traces(marker_color='#780606', showlegend=False)  
        fig.update_layout(
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=14,
            yaxis_tickfont_size=14,
            xaxis_tickangle=-50  
        )
        st.plotly_chart(fig, use_container_width=True)

elif menu == "Student Statistics":
    st.header("Student Statistics")
    
    # Calculate awards
    name_counts = df['Name'].value_counts()
    
    # 1st Place MVB
    most_valuable_booker = name_counts.index[0]  
    
    # 2nd Place MVB
    second_most_valuable_booker = name_counts.index[1] if len(name_counts) > 1 else None  
    
    # 3rd Place MVB
    third_most_valuable_booker = name_counts.index[2] if len(name_counts) > 2 else None  
    
    # Early Bird Award
    most_frequent_before_9am = df[df['Time'].dt.hour < 9]['Name'].value_counts().idxmax()
    
    # Free Spirit Award
    least_frequent_name = name_counts.index[-1]
    
    # Late Night Warrior
    after_5pm = df[df['Time'].dt.hour > 17]
    most_frequent_after_5pm = after_5pm['Name'].value_counts().idxmax()

    # First Floor Champion
    first_floor = df[df['Floor'] == '1']
    most_frequent_first_floor = first_floor['Name'].value_counts().idxmax()

    # Second Floor Champion
    second_floor = df[df['Floor'] == '2']
    most_frequent_second_floor = second_floor['Name'].value_counts().idxmax()

    # Jack-of-All-Rooms Award
    room_variety = df.groupby('Name')['Room'].nunique()
    most_variety = room_variety.idxmax()

    # Double-Booked Award
    overlapping_bookings = df.groupby(['Date', 'Time', 'Name']).size().reset_index(name='Count')
    double_booked = overlapping_bookings[overlapping_bookings['Count'] > 1]
    most_double_booked = double_booked['Name'].value_counts().idxmax() if not double_booked.empty else None
    
    # Add emojis to names for dropdown
    students = df[['Name', 'Headshot URL']].drop_duplicates().reset_index(drop=True)
    students['NameWithEmoji'] = students['Name']  # Start with the name
    
    for idx, row in students.iterrows():
        name = row['Name']
        emoji_list = []
        if name == most_valuable_booker:
            emoji_list.append("🥇")
        elif name == second_most_valuable_booker:
            emoji_list.append("🥈")
        elif name == third_most_valuable_booker:
            emoji_list.append("🥉")
        if name == most_frequent_before_9am:
            emoji_list.append("🌅")
        if name == most_frequent_after_5pm:
            emoji_list.append("🌙")
        if name == most_frequent_first_floor:
            emoji_list.append("🏆 (1F)")
        if name == most_frequent_second_floor:
            emoji_list.append("🏆 (2F)")
        if name == most_variety:
            emoji_list.append("🥾")
        if name == most_double_booked:
            emoji_list.append("😅")
        if name == least_frequent_name:
            emoji_list.append("✌")
        
        if emoji_list:
            students.loc[idx, 'NameWithEmoji'] = f"{name} {' '.join(emoji_list)}"
    
    # Dropdown with emojis
    selected_student_name = st.selectbox(
        "Search for a Student", 
        students['NameWithEmoji'].sort_values()
    )
    
    # Get the original name
    original_name = students.loc[students['NameWithEmoji'] == selected_student_name, 'Name'].values[0]
    selected_student = students[students['Name'] == original_name].iloc[0]
    
    # Display the student's headshot
    st.image(
        selected_student['Headshot URL'], 
        use_container_width=False, 
        width=400
    )
    
    # Display stats for the selected student
    student_data = df[df['Name'] == selected_student['Name']]
    total_bookings = len(student_data)
    total_time_hours = total_bookings * 30 / 60
    most_popular_room = student_data['Room'].mode().iloc[0] if not student_data.empty else "No bookings"
    most_popular_floor = student_data['Floor'].mode().iloc[0] if not student_data.empty else "No bookings"
    most_popular_time = student_data['Time'].dt.strftime('%I:%M %p').mode().iloc[0] if not student_data.empty else "No bookings"

    st.write(f"- Total bookings: {total_bookings}")
    st.write(f"- Total time booked: {total_time_hours:.2f} hours")
    st.write(f"- Most popular room: {most_popular_room}")
    st.write(f"- Most popular floor: {most_popular_floor}")
    st.write(f"- Most popular time to book: {most_popular_time}")
    
    # Awards
    if original_name == most_valuable_booker:
        st.write("🥇 **Most Valuable Booker (1st Place)**")
        st.balloons()
    elif original_name == second_most_valuable_booker:
        st.write("🥈 **Most Valuable Booker (2nd Place)**")
        st.balloons()
    elif original_name == third_most_valuable_booker:
        st.write("🥉 **Most Valuable Booker (3rd Place)**")
        st.balloons()
    if original_name == most_frequent_before_9am:
        st.write("🌅 **Early Bird Award**: Most frequent booker before 9 AM!")
    if original_name == most_frequent_after_5pm:
        st.write("🌙 **Late Night Warrior**: Most frequent booker after 5 PM!")
    if original_name == most_frequent_first_floor:
        st.write("🏆 **First Floor Champion**: Most frequent booker on the first floor!")
    if original_name == most_frequent_second_floor:
        st.write("🏆 **Second Floor Champion**: Most frequent booker on the second floor!")
    if original_name == most_variety:
        st.write("🥾 **Jack-of-All-Rooms**: The person who booked the largest variety of rooms!")
    if original_name == most_double_booked:
        st.write("😅 **Double-Booked Award**: The person with the most overlapping bookings!")
    if original_name == least_frequent_name:
        st.write("✌ **Free Spirit Award**: Least frequent booker!")

elif menu == "Time-Based Statistics":
    st.header("Time-Based Statistics")

    # Dropdown to select a specific graph
    graph_option = st.selectbox(
        "Choose a graph to display",
        [
            "Most Popular Days for Bookings",
            "First vs. Second Floor Bookings Over Time",
            "Top Booker by Time"
        ]
    )

    # Most Popular Days of the Week
    if graph_option == "Most Popular Days for Bookings":
        st.subheader("Most Popular Days for Bookings")
        day_counts = df['DayOfWeek'].value_counts().reset_index()
        day_counts.columns = ['DayOfWeek', 'Counts']

        fig = px.bar(
            day_counts,
            x='DayOfWeek',
            y='Counts',
            text='Counts',
            labels={'Counts': 'Number of Bookings', 'DayOfWeek': 'Day of the Week'}
        )
        fig.update_traces(marker=dict(color='#780606'))
        fig.update_layout(xaxis_title="Day of the Week", yaxis_title="Number of Bookings")
        st.plotly_chart(fig)

    # Floor-Wise Bookings Over Time 
    elif graph_option == "First vs. Second Floor Bookings Over Time":
        st.subheader("First vs. Second Floor Bookings Over Time")
        floor_bookings = df.groupby(['Date', 'Floor']).size().reset_index(name='Counts')
        floor_bookings['Cumulative Counts'] = floor_bookings.groupby('Floor')['Counts'].cumsum()
        
        fig = px.line(
            floor_bookings,
            x='Date',
            y='Cumulative Counts',
            color='Floor',
            labels={'Cumulative Counts': 'Cumulative Bookings', 'Date': 'Date'},
            color_discrete_map={'1': '#3182bd', '2': '#e6550d'},
            markers=True
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative Bookings")
        st.plotly_chart(fig)

    # Top Booker by Time 
    elif graph_option == "Top Booker by Time":
        st.subheader("Top Booker by Time")

        df['Time'] = pd.to_datetime(df['Time'], format='%I:%M%p')
        time_filtered_df = df[(df['Time'].dt.hour >= 7) & (df['Time'].dt.hour <= 19)]

        # Group data by time and find the top booker
        top_bookers = (
            time_filtered_df.groupby(time_filtered_df['Time'].dt.strftime('%I:%M %p'))['Name']
            .agg(lambda x: x.value_counts().idxmax()) 
            .reset_index()
            .rename(columns={'Time': 'TimeSlot', 'Name': 'Top Booker'})
        )
        slot_counts = time_filtered_df.groupby(time_filtered_df['Time'].dt.strftime('%I:%M %p')).size().reset_index(name='Counts')
        time_data = pd.merge(slot_counts, top_bookers, left_on='Time', right_on='TimeSlot')

        time_data['TimeSlot'] = pd.to_datetime(time_data['TimeSlot'], format='%I:%M %p')
        time_data = time_data.sort_values(by='TimeSlot') 
        time_data['TimeSlot'] = time_data['TimeSlot'].dt.strftime('%I:%M %p')  

        fig = px.scatter(
            time_data,
            x='TimeSlot',
            y='Counts',
            size='Counts',
            hover_data={'Top Booker': True},
            labels={'Counts': 'Number of Bookings', 'TimeSlot': 'Time Slot'}
        )
        fig.update_traces(marker=dict(color='#780606'))
        fig.update_layout(
            xaxis_title="Time Slot",
            yaxis_title="Number of Bookings",
            xaxis=dict(tickangle=45)  
        )
        st.plotly_chart(fig)

# Cluster Analysis Section
elif menu == "Cluster Analysis":
    st.header("Cluster Analysis")
    
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    student_data = df.groupby('Name').size().reset_index(name='ReservationCount')
    student_data['LastName'] = student_data['Name'].apply(lambda name: name.split()[-1])
    student_data['AvgTime'] = df.groupby('Name')['Time'].apply(lambda x: pd.to_datetime(x).dt.hour.mean()).values
    student_data['MostBookedRoom'] = df.groupby('Name')['Room'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').values
    student_data = pd.get_dummies(student_data, columns=['MostBookedRoom'], drop_first=True)

    # Standardize the data
    scaler = StandardScaler()
    scaled_student_data = scaler.fit_transform(student_data.drop(['Name', 'LastName'], axis=1))

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_student_data)
    student_data['PCA1'] = pca_data[:, 0]
    student_data['PCA2'] = pca_data[:, 1]

    # KNN clustering
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(pca_data)
    distances, indices = knn.kneighbors(pca_data)

    # Interactive Scatter Plot
    fig = px.scatter(
        student_data,
        x='PCA1',
        y='PCA2',
        hover_name='Name',  
        hover_data={'Name': False, 'ReservationCount': True, 'AvgTime': False},
        title='K-Nearest Neighbors - Booking Patterns Network',
        labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'}
    )

    # Add lines to connect neighbors
    for i in range(len(pca_data)):
        for j in indices[i]:
            fig.add_shape(
                type='line',
                x0=pca_data[i, 0],
                y0=pca_data[i, 1],
                x1=pca_data[j, 0],
                y1=pca_data[j, 1],
                line=dict(color='gray', width=1, dash='dot')
            )

    fig.update_traces(
        marker=dict(size=10, color='blue', line=dict(width=1, color='black'))
    )
    fig.update_layout(
        showlegend=False,
        width=1000,
        height=800
        )
    
    st.plotly_chart(fig, use_container_width=True)

