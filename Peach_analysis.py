'''

Peach Analysis streamlit app

'''
import os
import fnmatch
import platform

import streamlit as st
import numpy as np 
import pandas as pd
import scipy as sp
import glob
import plotly.graph_objects as go
from datetime import datetime
import timedelta
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

import scipy.integrate as integrate


st.image('rowing_can.png', width = 150)
st.title("Rowing Canada Peach Analysis")

def find_folder(root_path, folder_name):
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            if fnmatch.fnmatch(dir_name, folder_name):
                folder_path = os.path.join(root, dir_name)
                return folder_path

os_name = platform.system()

uploaded_data = st.file_uploader('select file for analysis')
'''
if os_name == 'Darwin':  # macOS
    root_path = '/'
    delimiter = '/'
elif os_name == 'Windows': #Windows (obviously)
    root_path = 'C:\\'
    delimiter = '\\'

target_folder = 'HP - Staff - SSSM'

#result = find_folder(root_path, target_folder)
og_path = f'{result}{delimiter}General{delimiter}Biomechanics{delimiter}Peach data{delimiter}'

#folders = glob.glob(f'{og_path}*{delimiter}', recursive = True)
	
#folder_list = []

#for f in folders:
#	folder = f.split('/')[-2]
#	folder_list.append(folder)

#session = st.selectbox('Select Session For Analysis', folder_list)
session = []

files = glob.glob(f'{og_path}{delimiter}{session}{delimiter}*.csv')
file_paths = []
session_names = []
for file in files: 
	if 'intervals' not in file:
		session_name = file.split('.')[-2]
		session_name = session_name.split('/')[-1]
		session_names.append(session_name)
		file_paths.append(file)

session_names.insert(0, 'Select Session')
#boat_select = st.selectbox('Select Boat', session_names)
'''

#for i in range(len(file_paths)): 
if uploaded_data is None:
	st.header("Select Session")
	st.stop()

#if boat_select == 'Select Session':
#	st.header("Select Session")
#	st.stop()
else:
	#data = pd.read_csv(f'{og_path}{delimiter}{session}{delimiter}{boat_select}.csv')
	data = pd.read_csv(uploaded_data)
	data_array = data.values
	data_list = data_array.tolist()
	aperiodic = []
	for row_idx, row in enumerate(data_list):
		if 'AvgBoatSpeed' in row:
			col_idx = row.index('AvgBoatSpeed')
			aperiodic.append(row_idx)			
		if 'lat' in row:
			col_idx_lat = row.index('lat')
			row_index_lat = np.where(data.iloc[:,col_idx_lat]=='lat')[0][0]	
		if 'Abbreviation' in row:
			col_idx_names = row.index('Abbreviation')
			row_idx_names = np.where(data.iloc[:,col_idx_names]=='Abbreviation')[0][0]	

	
	
	#Periodic Data
	periodic = np.where(data['File Info'] == 'Periodic')[0][0]
	periodic_data = data.iloc[periodic+1:].reset_index(drop=True)
	cutttoff = np.where(periodic_data.iloc[1] == 'Boat')[0][0]
	seats = int(periodic_data.iloc[1][1:cutttoff-1].max())
	names = data.iloc[row_idx_names+1:row_idx_names+seats+1,col_idx_names]
	
	
	if all(x for x in names):
		names = list(names)
	else: 
		st.write('names incomplete')
		
	

	#Aperiodic Data
	aperiodic_data = data.iloc[aperiodic[0]:].reset_index(drop=True)
	AP_cutttoff = np.where(aperiodic_data.iloc[0] == 'AvgBoatSpeed')[0][0]
	boat_speed = aperiodic_data.iloc[:,AP_cutttoff][3:np.where(aperiodic_data['File Info'] == 'Aperiodic')[0][0]].reset_index(drop=True)
	boat_speed = np.array(boat_speed.astype(float))
	boat_lat = data.iloc[row_index_lat+1:np.where(data['File Info'] == 'Aperiodic')[0][1],col_idx_lat].reset_index(drop = True)
	boat_lat = np.array(boat_lat.astype(float))
	boat_long = data.iloc[row_index_lat+1:np.where(data['File Info'] == 'Aperiodic')[0][1],col_idx_lat+1].reset_index(drop = True)
	boat_long = np.array(boat_long.astype(float))
	boat_dist = np.sqrt(boat_long**2 + boat_lat**2)

	threshold = st.number_input('Detect Velcoity Above:', value=4.5)

	Threshold_velocity = np.where(boat_speed >= threshold)[0]
	

	def group_data(array, threshold):
	    groups = []
	    current_group = [array[0]]

	    for i in range(1, len(array)):
	        diff = array[i] - array[i - 1]
	        if diff > threshold:
	            groups.append(current_group)
	            current_group = [array[i]]
	        else:
	            current_group.append(array[i])
	    if len(current_group) >= 3:  # Append the last group if it has at least 3 items
        	groups.append(current_group)

	    return groups

	section_indexes = group_data(Threshold_velocity, 1)

	
	fig1 = go.Figure()

	
	fig1.add_trace(go.Scatter(y=boat_speed,
    	fill=None,
    	mode='lines',
    	line_color = 'blue',
    	name = 'boat speed'))
	for section in section_indexes: 
		if len(section)>2:

			fig1.add_shape(
		        type="rect",
		        x0=section[0],
		        x1=section[-1],
		        y0=0,
		        y1= 7,
		        fillcolor="grey",
		        opacity=0.2)

			fig1.add_vline(x=section[0], line_width=3, line_dash="dash", line_color= '#2ca02c' , annotation_text= 'Piece Start', annotation_textangle = 270, annotation_position="top left")
			fig1.add_vline(x=section[-1], line_width=3, line_dash="dash", line_color= '#d62728' , annotation_text= 'Piece End', annotation_textangle = 270, annotation_position="top left")
	
	st.plotly_chart(fig1)

	angle_frame = pd.DataFrame()
	force_frame = pd.DataFrame()
	catch_frame = pd.DataFrame()
	finish_frame = pd.DataFrame()
	count = 0
	name_input = st.text_input('Enter Athlete Names (ie. AA,BB,CC)', value = ','.join(names))
	
	name_list = []

	for seat in range(0,seats): 
		name = name_input.split(',')[seat]
		name_list.append(name)

	if len(name_list)!= seats:
		st.write('enter names')
		st.stop()

	if name_list is None: 
		name_select = st.selectbox('select athlete for analysis', name_list)
		st.write("detected none")
	else: 
		name_select = st.selectbox('select athlete for analysis', name_list)

	for section in section_indexes: 
		if len(section)>2:

			count += 1


			time_on = aperiodic_data.iloc[:,0][section[0]+3]
			time_off = aperiodic_data.iloc[:,0][section[-1]]
			

			periodic_onset = np.where(periodic_data.iloc[:,0][2:].astype(int) >= int(time_on))[0][0]
			periodic_offset = np.where(periodic_data.iloc[:,0][2:].astype(int) >= int(time_off))[0][0]

			aperiodic_data.iloc[:,0] = pd.to_numeric(aperiodic_data.iloc[:,0],errors='coerce')
			aperiodic_onset = np.where(aperiodic_data.iloc[3:,0].astype(float) >= int(time_on))[0][0]
			aperiodic_offset = np.where(aperiodic_data.iloc[3:,0].astype(float) >= int(time_off))[0][0]
			
			gps_data = data.iloc[row_index_lat:]
			gps_data = gps_data[:np.where(gps_data['File Info'] == 'Aperiodic')[0][0]]

			boat_dist_onset = np.where(gps_data.iloc[2:,0].astype(int) >= int(time_on))[0][0]
			boat_dist_offset = np.where(gps_data.iloc[2:,0].astype(int) >= int(time_off))[0][0]
			power_data = np.where(aperiodic_data.iloc[0].str.endswith('Average Power'))[0]	
			power_data = aperiodic_data.iloc[:,power_data].dropna()
			power_data_crop = power_data[aperiodic_onset:aperiodic_offset]
			power_data_crop = power_data_crop.iloc[3:,1].astype(float)
			#power_data_crop = np.array(power_data_crop)
			
			swivel_pow = np.where(aperiodic_data.iloc[0].str.endswith('Rower Swivel Power'))[0]
			swivel_pow = aperiodic_data.iloc[:,swivel_pow].dropna()
			swivel_pow_crop = power_data[aperiodic_onset:aperiodic_offset]
			swivel_pow_crop = swivel_pow_crop.iloc[:,1].astype(float)
			
			

			#st.line_chart(boat_dist[boat_dist_onset:boat_dist_offset] - boat_dist[boat_dist_onset])
			
			

			col3, col4, col5, col6 = st.columns(4)
			with col3: 
				st.metric('Section Time', (float(time_off)-float(time_on))/1000)
				st.metric('Average Seat Power', round(swivel_pow_crop.mean(),2))
			with col4:
				st.metric('Average Boat Power', round(np.mean(power_data_crop),2))
			with col5: 
				avg_speed = np.mean(boat_speed[section[0]:section[-1]])
				avg_speed = round(avg_speed,3)
				st.metric('Avg Boat Speed', avg_speed)
			with col6: 
				st.metric('Piece Distance (m)', abs(round(boat_dist[boat_dist_offset]-boat_dist[boat_dist_onset])))

			angle_data = np.where(periodic_data.iloc[0].str.endswith('GateAngle'))[0]
			angle_data = periodic_data.iloc[:,angle_data]
			angle_data_crop = angle_data[periodic_onset:periodic_offset]
			
			forceX_data = np.where(periodic_data.iloc[0].str.endswith('GateForceX'))[0]
			forceX_data = periodic_data.iloc[:,forceX_data]
			forceX_data_crop = forceX_data[periodic_onset:periodic_offset]

			catch_slip = np.where(aperiodic_data.iloc[0].str.endswith('CatchSlip'))[0]
			catch_slip = aperiodic_data.iloc[:,catch_slip]
			catch_slip_crop = catch_slip[aperiodic_onset:aperiodic_offset]

			finish_slip = np.where(aperiodic_data.iloc[0].str.endswith('FinishSlip'))[0]
			finish_slip = aperiodic_data.iloc[:,finish_slip]
			finish_slip_crop = finish_slip[aperiodic_onset:aperiodic_offset]

			min_angle = np.where(aperiodic_data.iloc[0].str.endswith('MinAngle'))[0]
			if len(min_angle) < seats: 
				min_angle = np.where(aperiodic_data.iloc[0].str.endswith('Min Angle'))[0]
			
			min_angle = aperiodic_data.iloc[:,min_angle]
			min_angle_crop = min_angle[aperiodic_onset:aperiodic_offset]


			max_angle = np.where(aperiodic_data.iloc[0].str.endswith('MaxAngle'))[0]
			max_angle = aperiodic_data.iloc[:,max_angle]
			max_angle_crop = max_angle[aperiodic_onset:aperiodic_offset]

			gate_vel = np.where(periodic_data.iloc[0].str.endswith('GateAngleVel'))[0]
			gate_vel = gate_vel[:seats]
			
			gate_vel = periodic_data.iloc[:,gate_vel]
			gate_vel = gate_vel[periodic_onset:periodic_offset]


			#Analyze gate data

			angle_data = np.where(periodic_data.iloc[0].str.endswith('GateAngle'))[0]
			angle_data = periodic_data.iloc[:,angle_data]
			angle_data_crop = angle_data[periodic_onset:periodic_offset]
			
			forceX_data = np.where(periodic_data.iloc[0].str.endswith('GateForceX'))[0]
			forceX_data = periodic_data.iloc[:,forceX_data]
			forceX_data_crop = forceX_data[periodic_onset:periodic_offset]

			catch_slip = np.where(aperiodic_data.iloc[0].str.endswith('CatchSlip'))[0]
			catch_slip = aperiodic_data.iloc[:,catch_slip]
			catch_slip_crop = catch_slip[aperiodic_onset:aperiodic_offset]

			finish_slip = np.where(aperiodic_data.iloc[0].str.endswith('FinishSlip'))[0]
			finish_slip = aperiodic_data.iloc[:,finish_slip]
			finish_slip_crop = finish_slip[aperiodic_onset:aperiodic_offset]

			min_angle = np.where(aperiodic_data.iloc[0].str.endswith('MinAngle'))[0]
			if len(min_angle) < seats: 
				min_angle = np.where(aperiodic_data.iloc[0].str.endswith('Min Angle'))[0]
			
			min_angle = aperiodic_data.iloc[:,min_angle]
			min_angle_crop = min_angle[aperiodic_onset:aperiodic_offset]
	

			max_angle = np.where(aperiodic_data.iloc[0].str.endswith('MaxAngle'))[0]
			max_angle = aperiodic_data.iloc[:,max_angle]
			max_angle_crop = max_angle[aperiodic_onset:aperiodic_offset]



			athlete_select = np.where(pd.Series(name_list).str.contains(name_select))[0][0]+1



			#Angle Data
			seat_angle = np.where(pd.to_numeric(angle_data.iloc[1], errors='coerce') == athlete_select)[0]
			seat_angle_data = angle_data_crop.iloc[:,seat_angle].reset_index(drop=True)
			seat_angle_data = seat_angle_data[2:]
			

			#Force Data
			seat_forceX = np.where(pd.to_numeric(forceX_data.iloc[1], errors='coerce')== athlete_select)[0]
			seat_forceX_data = forceX_data_crop.iloc[:,seat_forceX].reset_index(drop=True)
			seat_forceX_data = seat_forceX_data[2:]

			extremes = list(np.where(abs(seat_angle_data.astype(float))>200)[0])
			force_extremes = list(np.where(abs(seat_forceX_data.astype(float))>200)[0])
			
			
			#Removing extreme angle data	
			if len(extremes)>0 or len(force_extremes)>0:
				seat_angle_data = seat_angle_data.drop(extremes)
				seat_forceX_data = seat_forceX_data.drop(extremes)
				
			

			#Slip Data	
			seat_cslip = np.where(pd.to_numeric(catch_slip.iloc[1], errors='coerce')== athlete_select)[0]
			seat_cslip_data = catch_slip_crop.iloc[:,seat_cslip].reset_index(drop=True)
			#seat_cslip_data = seat_cslip_data[2:]
			
			seat_fslip = np.where(pd.to_numeric(finish_slip.iloc[1], errors='coerce')== athlete_select)[0]
			seat_fslip_data = finish_slip_crop.iloc[:,seat_fslip].reset_index(drop=True)
			#seat_fslip_data = seat_fslip_data[2:]

			seat_min = np.where(pd.to_numeric(min_angle.iloc[1], errors='coerce')== athlete_select)[0]
			seat_min_data = min_angle_crop.iloc[:,seat_min].reset_index(drop=True)
			#seat_min_data = seat_min_data[2:]


			seat_max = np.where(pd.to_numeric(max_angle.iloc[1], errors='coerce')== athlete_select)[0]
			seat_max_data = max_angle_crop.iloc[:,seat_max].reset_index(drop=True)
			#seat_max_data = seat_max_data[2:]

			#SeatPower
			

			fig = go.Figure()
			fig.update_layout(xaxis_range=[-70,50])



			for gate in range(len(seat_forceX)):
				
				front_slip = seat_min_data.iloc[:,gate].astype(float) + seat_cslip_data.iloc[:,gate].astype(float)
				end_slip = seat_max_data.iloc[:,gate].astype(float) - seat_fslip_data.iloc[:,gate].astype(float)
				min_mean = np.mean(seat_min_data.iloc[:,gate].astype(float))
				max_mean = np.mean(seat_max_data.iloc[:,gate].astype(float))
				front_res = np.mean(front_slip) - min_mean
				end_res = max_mean - np.mean(end_slip)

				fig.add_trace(go.Scatter(x=seat_angle_data.iloc[:,gate], y=seat_forceX_data.iloc[:,gate],
			    	fill=None,
			    	mode='lines',
			    	#line_color = 'red',
			    	name = 'Angle Vs. Force'))
				fig.add_vline(x=np.mean(front_slip), line_width=3, line_dash="dash", line_color=  f'#2ca0{gate*2}c' , annotation_text= f'Average Catch Slip:  <b>{round(front_res)}<b>', annotation_textangle = 270, annotation_position="top left")
				fig.add_vline(x=np.mean(end_slip), line_width=3, line_dash="dash", line_color=  f'#5ca0{gate*2}c' , annotation_text= f'Average Finish Slip:  <b>{round(end_res)}<b>', annotation_textangle = 270, annotation_position="top left")
				fig.update_layout(title = f"<b>Force Vs. Gate Angle:<b> {name_select} Seat {athlete_select} Piece {count}", 
									xaxis_title = '<b>Gate Angle<b> (Degrees)', 
									yaxis_title = '<b>Gate Force<b> (N)')
			st.plotly_chart(fig)

	


	
	export_data = pd.Series([])

	#stroke_impulse = integrate.cumtrapz(, dx = 1)

	#fig2 = go.Figure()

	




		
	













