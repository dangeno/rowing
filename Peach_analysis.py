'''

Peach Analysis streamlit app

#POWER VS DISTANCE PER STROKE
Effective work per stroke


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
from scipy.signal import savgol_filter

import scipy.integrate as integrate


st.image('rowing_canada.png', width = 150)
st.title("Rowing Canada Peach Analysis")

def find_folder(root_path, folder_name):
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            if fnmatch.fnmatch(dir_name, folder_name):
                folder_path = os.path.join(root, dir_name)
                return folder_path

def lowpass(signal, highcut): 
	
	order = 4 

	nyq = 0.5*frequency
	highcut = highcut/nyq

	b,a = sp.signal.butter(order, [highcut], 'lowpass', analog=False)
	y = sp.signal.filtfilt(b,a, signal, axis = 0)
	return(y)

os_name = platform.system()

uploaded_data = st.file_uploader('select file for analysis')

frequency = 50

#for i in range(len(file_paths)): 
if uploaded_data is None:
	st.header("Select Session")
	st.stop()

#if boat_select == 'Select Session':
#	st.header("Select Session")
#	st.stop()
else:
	#data = pd.read_csv(f'{og_path}{delimiter}{session}{delimiter}{boat_select}.csv')
	file_type = uploaded_data.name.split('.')[-1]
	
	data = pd.read_csv(uploaded_data)
	data_array = data.values
	data_list = data_array.tolist()
	aperiodic = []
	for row_idx, row in enumerate(data_list):
		if 'AvgBoatSpeed' in row:
			col_idx = row.index('AvgBoatSpeed')
			aperiodic.append(row_idx)			
		if 'Rig' in row:
			col_idx_rig = row.index('Rig')
			row_idx_rig = np.where(data.iloc[:,col_idx_rig]=='Rig')[0][0]	
		if 'Abbreviation' in row:
			col_idx_names = row.index('Abbreviation')
			row_idx_names = np.where(data.iloc[:,col_idx_names]=='Abbreviation')[0][0]	

	
	
	#Periodic Data
	periodic = np.where(data['File Info'] == 'Periodic')[0][0]
	periodic_data = data.iloc[periodic+1:].reset_index(drop=True)
	cutttoff = np.where(periodic_data.iloc[1] == 'Boat')[0][0]
	seats = int(periodic_data.iloc[1][1:cutttoff-1].max())
	names = data.iloc[row_idx_names+1:row_idx_names+seats+1,col_idx_names]
	rig = data.iloc[row_idx_rig+1, col_idx_rig]
	
	
	
	if all(x for x in names):
		names = list(names)
	else: 
		st.write('names incomplete')
		
	

	#Aperiodic Data
	aperiodic_data = data.iloc[aperiodic[0]:].reset_index(drop=True)
	AP_cutttoff = np.where(aperiodic_data.iloc[0] == 'AvgBoatSpeed')[0][0]

	
	boat_speed = aperiodic_data.iloc[:,AP_cutttoff][3:np.where(aperiodic_data['File Info'] == 'Aperiodic')[0][0]].reset_index(drop=True)
	boat_speed = np.array(boat_speed.astype(float))
	boat_speed = boat_speed[np.logical_not(np.isnan(boat_speed))]

	vel_threshold = boat_speed.max()-1


	threshold = st.number_input('Detect Velcoity Above:', value=round(vel_threshold,1))

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

	plot_show = st.checkbox('Show Detailed Plots')

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

	if rig == 'sculling': 
		if plot_show == True:
			port_star_sel = st.selectbox('Select Gate', ['Port', 'Starboard'])

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
			

			boat_dist = np.where(periodic_data.iloc[0,:]=='Distance')[0]
			boat_dist = periodic_data.iloc[2:,boat_dist]


			boat_dist = (boat_dist.iloc[periodic_onset:periodic_offset,:].astype(float) - boat_dist.iloc[periodic_onset,:].astype(float))
			power_data = np.where(aperiodic_data.iloc[0].str.endswith('Average Power'))[0]	
			power_data = aperiodic_data.iloc[:,power_data].dropna()
			power_data_crop = power_data[aperiodic_onset:aperiodic_offset]
			power_data_crop = power_data_crop.iloc[3:,1].astype(float)
			#power_data_crop = np.array(power_data_crop)
			
			swivel_pow = np.where(aperiodic_data.iloc[0].str.endswith('Rower Swivel Power'))[0]
			swivel_pow = aperiodic_data.iloc[:,swivel_pow]
			swivel_pow_crop = swivel_pow.iloc[aperiodic_onset:aperiodic_offset,:]
			swivel_pow_avg = swivel_pow_crop.iloc[:, 1:].astype(float)
			swivel_pow_avg = swivel_pow_avg.mean()
			swivel_pow_avg = swivel_pow_avg.mean()


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
			angle_data_crop = angle_data[periodic_onset:periodic_offset].astype(float)
			avg_angle = angle_data_crop.mean(axis=1)
			
			
			accel_data = np.where(periodic_data.iloc[0].str.endswith('Accel'))[0]
			accel_data = periodic_data.iloc[:,accel_data]
			accel_data_crop = accel_data[periodic_onset:periodic_offset].iloc[:,0]
			plot_accel = lowpass(accel_data_crop,5)
			


			
			

			fig2 = go.Figure()

			fig2.add_trace(go.Scatter(x=avg_angle, y=plot_accel,
		    	fill=None,
		    	mode='lines',
		    	line_color = 'blue',
		    	opacity=.6,
		    	name = 'Acceleration'))
			

			round_angle = avg_angle.round()
			postive_pairs = []
			negative_pairs = []
			positive_y_values = []
			negative_y_values = []
			previous_x = None

			for x, y in zip(round_angle, accel_data_crop):
				if previous_x is not None:
					if x > previous_x:
						positive_y_values.append(y)
						postive_pairs.append((x, y))
					elif x < previous_x:
						negative_y_values.append(y)
						negative_pairs.append((x, y))
				
				previous_x = x
			
			pos_trace_data = pd.DataFrame(postive_pairs).astype(float)
			pos_trace_data.columns = ['angles', 'accel']
			pos_trace_data = pos_trace_data.groupby(['angles']).mean()
			pos_trace_data = pos_trace_data.reset_index()

			neg_trace_data = pd.DataFrame(negative_pairs).astype(float)
			neg_trace_data.columns = ['angles', 'accel']
			neg_trace_data = neg_trace_data.groupby(['angles']).mean()
			neg_trace_data = neg_trace_data.reset_index()

			trace_data = pos_trace_data.append(neg_trace_data, ignore_index=True)
			trace_data.columns = ['angles', 'accel']
			    
			fig2.add_trace(go.Scatter(x=trace_data['angles'], y=trace_data['accel'],
			
		    	fill=None,
		    	mode='markers',
		    	line_color = 'red',
		    	name = 'Average Acceleration'))

			fig2.add_hline(y=0)
			
			fig2.update_layout(title = f"<b>Boat Acceleration Vs. Gate Average Angle", 
								xaxis_title = '<b>Average Gate Angle<b> (Degrees)', 
								yaxis_title = '<b>Boat Acceleration<b> (m/s2)')
			if plot_show==True:
				st.plotly_chart(fig2)


			
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


			angle_vel = np.where(periodic_data.iloc[0].str.endswith('GateAngleVel'))[0]
			angle_vel = periodic_data.iloc[:,angle_vel]
			angle_vel_crop = angle_vel[periodic_onset:periodic_offset]

			forceX_data_crop.replace("", float('NaN'), inplace=True)
			forceX_data_crop.dropna(how='all', axis = 1, inplace=True)
			forceX_data_crop = forceX_data_crop.astype(float)
			mean_force = forceX_data_crop.mean(axis = 1).reset_index(drop = True)
			mean_gate_angle = angle_data_crop.mean(axis = 1).reset_index(drop = True)

			fig3 = go.Figure()
			fig3.update_layout(title = f"<b>Angular Velocity Vs. Gate Angle:<b> Piece {count}", 
									xaxis_title = '<b>Gate Angle<b> (Deg)', 
									yaxis_title = '<b>Gate Velocity<b> (Deg/s)')

			if plot_show == True:

				if rig =='sculling': 
					fig4 = go.Figure()
					
					for seat in range(0,seats):
						if port_star_sel == 'Port':
							fig3.add_trace(go.Scatter(x=angle_data_crop.iloc[:,seat], y=angle_vel_crop.iloc[:,seat],
						    	fill=None,
						    	mode='lines',
						    	#line_color = 'red',
						    	name = f'Port Angle Velocity Vs. Angle Seat {seat+1}'))
						if port_star_sel == 'Starboard':
							fig3.add_trace(go.Scatter(x=angle_data_crop.iloc[:,(seat+seats)], y=angle_vel_crop.iloc[:,(seat+seats)],
						    	fill=None,
						    	mode='lines',
						    	#line_color = 'red',
						    	name = f'Sartboard Angle Velocity Vs. Angle Seat {seat+1}'))
					
					st.plotly_chart(fig3)
					
				else: 
					for seat in range(0,seats):
						fig3.add_trace(go.Scatter(x=angle_data_crop.iloc[:,seat], y=angle_vel_crop.iloc[:,seat],
						    	fill=None,
						    	mode='lines',
						    	#line_color = 'red',
						    	name = f'Angle Velocity Vs. Angle Seat {seat+1}'))

					st.plotly_chart(fig3)

		




			col4, col5, col6 = st.columns(3)
				
			with col4:
				st.metric('Average Boat Power', round(np.mean(power_data_crop),2))
			with col5: 
				avg_speed = np.mean(boat_speed[section[0]:section[-1]])
				avg_speed = round(avg_speed,3)
				st.metric('Avg Boat Speed', avg_speed)
				
			with col6: 
				final_dist = float(boat_dist.max())
				st.metric('Piece Distance (m)', round(final_dist))


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

			#Power Data
			seat_power = np.where(pd.to_numeric(swivel_pow.iloc[1], errors='coerce') == athlete_select)[0]
			seat_power_data = swivel_pow_crop.iloc[:,seat_power].astype(float)



			#length Data
			seat_min = np.where(pd.to_numeric(min_angle.iloc[1], errors='coerce') == athlete_select)[0]
			seat_min_data = min_angle_crop.iloc[:,seat_min].astype(float)
			seat_max = np.where(pd.to_numeric(max_angle.iloc[1], errors='coerce') == athlete_select)[0]
			seat_max_data = max_angle_crop.iloc[:,seat_max].astype(float)

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
			seat_min_data = seat_min_data.astype(float)
			#seat_min_data = seat_min_data[2:]


			seat_max = np.where(pd.to_numeric(max_angle.iloc[1], errors='coerce')== athlete_select)[0]
			seat_max_data = max_angle_crop.iloc[:,seat_max].reset_index(drop=True)
			seat_max_data = seat_max_data.astype(float)
			#seat_max_data = seat_max_data[2:]
			
		

			
			if len(seat_max_data.columns)>1:
				for col in range(len(seat_max_data.columns)): 
					length = seat_max_data.iloc[:,col] - seat_min_data.iloc[:,col]
					length = length.dropna()
					length = np.mean(length)
					eff_length = length - seat_cslip_data.iloc[:,col].dropna().astype(float).mean() - seat_fslip_data.iloc[:,col].dropna().astype(float).mean()
					st.write(eff_length)
					
					if col == 1:
						with col4:
							st.metric("P Effective Length (deg)", round(eff_length,1))
					else:
						with col5: 
							st.metric("S Effective Length (deg)", round(eff_length,1))
				with col6:
					st.metric('Average Seat Power', round(seat_power_data.mean(),2))
			
			else: 
				with col4: 
					length = np.array(seat_max_data.dropna()) - np.array(seat_min_data.dropna())
					length = np.mean(length)
					
					eff_length = length - seat_cslip_data.dropna().astype(float).mean()[0] - seat_fslip_data.dropna().astype(float).mean()[0]
					st.metric("Effective Length (deg)", round(eff_length,1))
				with col5:
					st.metric('Average Seat Power', round(seat_power_data.mean(),2), delta= float(seat_power_data.mean() - swivel_pow_avg))


			fig = go.Figure()
			fig.update_layout(xaxis_range=[-70,50])
			st.header('Slips')
			col7, col8 = st.columns(2)

			gate_count = 0

			fig.add_trace(go.Scatter(x=mean_gate_angle, y=mean_force,
						    	fill=None,
						    	mode='lines',
						    	opacity=0.5, 
						    	name = 'Boat Average Force Trace'))
			for gate in range(len(seat_forceX)):
				gate_count += 1
				
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
				fig.add_vline(x=np.mean(front_slip), line_width=3, line_dash="dash", line_color=  f'#{gate}ca0{gate*5}c' , annotation_text= f'Catch Slip {gate_count}:  <b>{round(front_res)}<b>', annotation_textangle = 270, annotation_position="top left")
				fig.add_vline(x=np.mean(end_slip), line_width=3, line_dash="dash", line_color=  f'#{gate}ca0{gate*5}c' , annotation_text= f'Finish Slip {gate_count}:  <b>{round(end_res)}<b>', annotation_textangle = 270, annotation_position="top left")
				fig.update_layout(title = f"<b>Force Vs. Gate Angle:<b> {name_select} Seat {athlete_select} Piece {count}", 
									xaxis_title = '<b>Gate Angle<b> (Degrees)', 
									yaxis_title = '<b>Gate Force<b> (N)')
				with col7: 
					st.metric('Catch Slip', round(np.mean(front_res),2))
					st.metric('Catch Length', round(np.mean(seat_min_data.iloc[:,gate].astype(float)),2))
				with col8: 
					st.metric('Finish Slip', round(np.mean(end_res),2))
					st.metric('Finish Length', round(np.mean(seat_max_data.iloc[:,gate].astype(float)),2))
			st.plotly_chart(fig)

			
			if len(seat_power_data)>41: 
				smoothed_power =  savgol_filter(seat_power_data.iloc[:, 0], window_length=40, polyorder=2)
			else: 
				smoothed_power =  savgol_filter(seat_power_data.iloc[:, 0], window_length=(len(seat_power_data) - 5), polyorder=2)

			


			fig5 = go.Figure()
			fig5.update_layout(title = f"<b>Power Per Stroke<b> {name_select} Seat {athlete_select} Piece {count}", 
									xaxis_title = '<b>Stroke (number)', 
									yaxis_title = '<b>Swivel Power (Watts)')
			fig5.add_trace(go.Scatter(y=seat_power_data.iloc[:,0],
			    	fill=None,
			    	mode='lines',
			    	line_color = 'red',
			    	name = 'Swivel Power'))
			fig5.add_trace(go.Scatter(
				    y=smoothed_power,
				    fill=None,
				    mode='lines',
				    line_color='blue',
				    name='Smoothed Power'))
			st.plotly_chart(fig5)

			for name in name_list: 
				export_data = pd.Series([name, count, ])
				
				

	


	
	
	#stroke_impulse = integrate.cumtrapz(, dx = 1)

	#fig2 = go.Figure()

	




		
	













