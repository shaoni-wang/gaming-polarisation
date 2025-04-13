import csv
import os
import numpy as np
import pandas as pd 
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import truncnorm
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# # # Load the dataset
# df1 = pd.read_csv("20250329SustainabilityS1.csv")
# df = pd.read_csv("20250329SustainabilityS3.csv")
df = pd.read_csv("20250329SustainabilitySensitivity.csv")

# #Figure 1 总数
# # stack area plots, Filter the DataFrame based on given conditions
# # Filter DataFrame based on given conditions
# filt1_df = df.loc[
#     (df['ThreAss'] == 0.5) &
#     (df['ThreCon'] == 0.5) &
#     (df['step'] >= 1) &
#     (df['step'] <= 100) &
#     (df['intervention'] == 0)
# ]

# # Calculate mean for each step for specific columns
# y0_mean = filt1_df.groupby('step')['ABsupp'].mean()
# y1_mean = filt1_df.groupby('step')['Asupp'].mean()
# y2_mean = filt1_df.groupby('step')['Bsupp'].mean()

# # Extract step indices
# x = y0_mean.index

# # Create figure and axis
# fig, ax = plt.subplots(figsize=(7, 5))

# # Stacked Area Plot with Brighter Colors
# ax.stackplot(x, y0_mean, y1_mean, y2_mean, 
#              labels=['Neutrals', 'Opponents', 'Supporters'],  
#              colors = ['darkgray', '#E69F00', '#009E73'],
#              alpha=0.8)

# # Formatting
# ax.set_xlim(0.2, 100.3)
# ax.set_ylim(-0.2, max(y0_mean.max(), y1_mean.max(), y2_mean.max())) ## + 5
# ax.set_title("Opponents, Neutrals, or Supporters?", fontsize=14)
# ax.set_xlabel("Time/Ticks", fontsize=14)
# ax.set_ylabel("Number of Individuals", fontsize=14)

# # Move legend completely outside
# ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12, frameon=False)

# # Increase axis tick label size
# ax.tick_params(axis='both', labelsize=12)

# # Adjust layout for legend to fit
# plt.tight_layout()

# # Show the plot
# plt.show()


###Figure 1.折线图，总数按照满意度分
# ###add line to the background
# filt1_df = df1.loc[
#     (df1['ThreAss'] == 0.5) &
#     (df1['ThreCon'] == 0.5) &
#     (df1['step'] >= 0) &
#     (df1['step'] <= 100) &
#     (df1['intervention'] == 0)
# ]

# # Define colorblind-friendly colors
# opponent_colors = ['#E69F00', '#D55E00', '#CC79A7']  # Orange, Dark Orange, Reddish Pink
# supporter_colors = ['#009E73', '#0072B2', '#56B4E9']  # Teal, Deep Blue, Sky Blue
# neutral_colors = ['black', 'dimgray', 'lightgray']  # Keeping neutral shades

# # Calculate mean values for each step
# y0_mean = filt1_df.groupby('step')['Apos'].mean()
# y1_mean = filt1_df.groupby('step')['Aneg'].mean()
# y2_mean = filt1_df.groupby('step')['A'].mean()
# y3_mean = filt1_df.groupby('step')['Bpos'].mean()
# y4_mean = filt1_df.groupby('step')['Bneg'].mean()
# y5_mean = filt1_df.groupby('step')['B'].mean()
# y6_mean = filt1_df.groupby('step')['ABpos'].mean()
# y7_mean = filt1_df.groupby('step')['ABneg'].mean()
# y8_mean = filt1_df.groupby('step')['AB'].mean()

# # Extract step indices
# x = y1_mean.index

# # Store y-values in a dictionary for easy iteration
# y_values = {
#     "Opponents+": (y0_mean, opponent_colors[0]),
#     "Opponents-": (y1_mean, opponent_colors[1]),
#     "Opponents": (y2_mean, opponent_colors[2]),
#     "Supporters+": (y3_mean, supporter_colors[0]),
#     "Supporters-": (y4_mean, supporter_colors[1]),
#     "Supporters": (y5_mean, supporter_colors[2]),
#     "Neutrals+": (y6_mean, neutral_colors[0]),
#     "Neutrals-": (y7_mean, neutral_colors[1]),
#     "Neutrals": (y8_mean, neutral_colors[2]),
# }

# # Create plot
# fig, ax = plt.subplots(figsize=(8, 5))

# # Add a more vibrant and obvious background with alternating solid color bands
# y_min, y_max = -1, 30  ## 25  31, 30
# num_bands = 3
# #band_colors = ['#D55E00', '#CC79A7', '#0072B2','#56B4E9']
# #band_colors = ['#9370DB', '#FF7F50', '#87CEFA', '#32CD32']
# band_colors = ['#FF9E7F', '#87CEFA', '#32CD32']  # New color choices


# # Create solid horizontal bands for a more obvious background
# for i in range(num_bands):
#     ax.axhspan(y_min + (i * (y_max - y_min) / num_bands),
#                 y_min + ((i + 1) * (y_max - y_min) / num_bands),
#                 facecolor=band_colors[i % len(band_colors)],
#                 alpha=0.25)  # Increase opacity for a stronger effect

# # Generate a smooth x-axis range
# x_smooth = np.linspace(x.min(), x.max(), 500)

# # Plot each category with a smoothed curve
# for label, (y_data, color) in y_values.items():
#     cubic_interp = interp1d(x, y_data, kind="cubic")  
#     y_smooth = cubic_interp(x_smooth)  
#     ax.plot(x_smooth, y_smooth, '-', label=label, color=color, lw=2)  

# # Formatting
# ax.set_xlim(0, 100.3)
# ax.set_ylim(y_min, y_max)
# ax.set_title("Satisfied(+), Neutral (), or Unsatisfied (-)", fontsize=14)
# ax.set_xlabel("Time/Ticks", fontsize=14)
# ax.set_ylabel("Numbers of individuals", fontsize=14)
# ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0., fontsize=11)
# ax.tick_params(axis='both', labelsize=12)

# # Improve layout and show plot
# plt.tight_layout()
# plt.show()


# ##Figure 1.折线图，总数按照满意度分
# ###平滑曲线图plot smooth curve
# # # Filter the DataFrame based on given conditions
# filt1_df = df1.loc[
#     (df1['ThreAss'] == 0.5) &
#     (df1['ThreCon'] == 0.5) &
#     (df1['step'] >= 1) &
#     (df1['step'] <= 100) &
#     (df1['intervention'] == 0)
# ]

# # Define colors from the same color family for each category
# # opponent_colors = ['darkred', 'brown', 'firebrick']
# # supporter_colors = ['darkgreen', 'seagreen', 'mediumseagreen']
# # neutral_colors = ['black', 'dimgray', 'darkgray']

# opponent_colors = ['#E69F00', '#D55E00', '#CC79A7']  # Orange, Dark Orange, Reddish Pink
# supporter_colors = ['#009E73', '#0072B2', '#56B4E9']  # Teal, Deep Blue, Sky Blue
# neutral_colors = ['black', 'dimgray', 'lightgray']  # Keeping neutral shades

# y0_mean = filt1_df.groupby('step')['Apos'].mean()
# y1_mean = filt1_df.groupby('step')['Aneg'].mean()
# y2_mean = filt1_df.groupby('step')['A'].mean()
# y3_mean = filt1_df.groupby('step')['Bpos'].mean()
# y4_mean = filt1_df.groupby('step')['Bneg'].mean()
# y5_mean = filt1_df.groupby('step')['B'].mean()
# y6_mean = filt1_df.groupby('step')['ABpos'].mean()
# y7_mean = filt1_df.groupby('step')['ABneg'].mean()
# y8_mean = filt1_df.groupby('step')['AB'].mean()

# # Extract step indices
# x = y1_mean.index

# # Store your y-values in a dictionary for easy iteration
# y_values = {
#     "Opponents+": (y0_mean, opponent_colors[0]),
#     "Opponents-": (y1_mean, opponent_colors[1]),
#     "Opponents": (y2_mean, opponent_colors[2]),
#     "Supporters+": (y3_mean, supporter_colors[0]),
#     "Supporters-": (y4_mean, supporter_colors[1]),
#     "Supporters": (y5_mean, supporter_colors[2]),
#     "Neutrals+": (y6_mean, neutral_colors[0]),
#     "Neutrals-": (y7_mean, neutral_colors[1]),
#     "Neutrals": (y8_mean, neutral_colors[2]),
# }

# # Create plot
# fig, ax = plt.subplots(figsize=(8, 5))

# # Generate a smooth x-axis range
# x_smooth = np.linspace(x.min(), x.max(), 500)

# # Loop through each category and plot smoothed curves
# for label, (y_data, color) in y_values.items():
#     cubic_interp = interp1d(x, y_data, kind="cubic")  # Create interpolation function
#     y_smooth = cubic_interp(x_smooth)  # Get smooth y values
#     ax.plot(x_smooth, y_smooth, '-', label=label, color=color, lw=2)  # Plot smooth line

# # Format subplot
# ax.set_xlim(0.7, 100.3)
# ax.set_ylim(-1, 25)
# ax.set_title("Satisfied(+), Neutral (), or Unsatisfied (-)", fontsize=14)
# ax.set_xlabel("Time/Ticks", fontsize=14)
# ax.set_ylabel("Numbers of individuals", fontsize=14)
# ax.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=11)
# #ax.legend(loc='best', fontsize=10, ncol=3)
# ax.tick_params(axis='both', labelsize=12)  # Change tick label size

# plt.tight_layout()
# plt.show()


# ####Figure 3: Comparison of the Satisfaction Rate
# # Filter the DataFrame based on given conditions (for two dataframes: df1 and df)
# filt1_df = df1.loc[
#     (df1['ThreAss'] == 0.5) &
#     (df1['ThreCon'] == 0.5) &
#     (df1['step'] == 100)
# ]
# filt3_df = df.loc[
#     (df['ThreAss'] == 0.5) &
#     (df['ThreCon'] == 0.5) &
#     (df['step'] == 100)
# ]

# # Calculate the mean support rates for different intervention conditions, 'support-rate', 'satisfaction-rate'
# y1_mean = filt1_df.loc[filt1_df['intervention'] == 0, 'satisfaction-rate'].mean() * 100  # No intervention (from df1)
# y2_mean = filt3_df.loc[filt3_df['intervention'] == 0, 'satisfaction-rate'].mean() * 100  # No intervention (from df)
# y3_mean = filt3_df.loc[filt3_df['intervention'] == 1, 'satisfaction-rate'].mean() * 100  # With intervention (from df)

# # Define bar labels and values
# categories = ['$Scenario$ $1$', '$Scenario$ $2$', '$Scenario$ $3$']
# values = [y1_mean, y2_mean, y3_mean]

# # Define x positions for bars
# x_positions = [0.2, 0.6, 1]

# # Create the bar plot
# fig, ax = plt.subplots(figsize=(6.5, 4))
# bars = ax.bar(x_positions, values, color=['#66B2FF', '#FF6F61', '#2E8B57'], alpha=0.7, width=0.1)

# # Add numbers on top of each bar
# for bar in bars:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2, height + 2, f'{height:.1f}%', ha='center', fontsize=12)

# # Add a line connecting the bars
# ax.plot(x_positions, values, linestyle='--', color='grey', marker='o', markersize=6, label="$Support$ $Rate$")

# # Set the labels for the bars in the legend
# bars[0].set_label('$Scenario$ $1$')
# bars[1].set_label('$Scenario$ $2$')
# bars[2].set_label('$Scenario$ $3$')

# # Format subplot
# ax.set_xticks(x_positions)
# ax.set_xticklabels(categories)
# ax.set_xlim(0, 1.2)
# ax.set_ylim(0, 100) # (0, 110)
# ax.set_ylabel("$Satisfaction$ $Rate$ (%)", fontsize=13)# "$Support$ $Rate$ (%)", "$Satisfaction$ $Rate$ (%)"
# ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=11)
# ax.tick_params(axis='both', labelsize=12)

# # Adjust layout to fit everything
# plt.tight_layout()

# # Show the plot
# plt.show()




# #11 柱状图 Filter the DataFrame based on given conditions
# filt1_df = df.loc[
#     (df['ThreAss'] == 0.5) &
#     (df['ThreCon'] == 0.5) &
#     (df['step'] >= 1) &
#     (df['step'] == 100) &
#     (df['intervention'] == 0)
# ]

# # Calculate mean for each category
# y0_mean = filt1_df['Asupp'].mean()
# y1_mean = filt1_df['Bsupp'].mean()
# y2_mean = filt1_df['ABsupp'].mean()

# # Bar chart setup
# labels = ['Opponents', 'Supporters', 'Neutrals']
# values = [y0_mean, y1_mean, y2_mean]
# colors = ['red', 'blue', 'gray']

# fig, ax = plt.subplots()
# ax.bar(labels, values, color=colors, alpha=0.75)

# # Format subplot
# ax.set_ylim(0, max(values) * 1.2)  # Add some padding to the y-axis
# ax.set_title("Opponents, Neutrals or Supporters")
# ax.set_ylabel("Number of Individuals")

# # Show the plot
# plt.show()


# # Filter the DataFrame based on given conditions
# filt1_df = df.loc[
#     (df['ThreAss'] == 0.5) &
#     (df['ThreCon'] == 0.5) &
#     (df['step'] >= 1) &
#     (df['step'] <= 100) &
#     (df['intervention'] == 1)
# ]

# # Calculate mean for each step for specific columns
# y0_mean = filt1_df.groupby('step')['Asupp'].mean()
# y1_mean = filt1_df.groupby('step')['Bsupp'].mean()
# y2_mean = filt1_df.groupby('step')['ABsupp'].mean()

# # Extract step indices
# x = y1_mean.index

# # Create plot
# fig, ax = plt.subplots(figsize=(6, 5))

# ax.plot(x, y0_mean, '-', label='Opponents', lw=2, color='Maroon')  #'x-',
# ax.plot(x, y1_mean, '-', label='Supporters', lw=2, color='darkgreen', markersize=2)  #'o-'
# ax.plot(x, y2_mean, '-', label='Neutral', lw=2, color='gray')

# # Format subplot
# ax.set_xlim(0.7, 100.3)
# ax.set_ylim(-1, 41)
# ax.set_title("Opponents, Neutrals or Supporters?", fontsize=14)
# ax.set_xlabel("Time/Ticks", fontsize=14)
# ax.set_ylabel("Numbers of individuals", fontsize=14)
# ax.legend(loc='right', fontsize=12)

# # **Increase axis tick label size**
# ax.tick_params(axis='both', labelsize=12)  # Change tick label size

# # Show the plot
# plt.show()


###Figure 1.折线图，总数按照满意度分
# # # Filter the DataFrame based on given conditions
# filt1_df = df.loc[
#     (df['ThreAss'] == 0.5) &
#     (df['ThreCon'] == 0.5) &
#     (df['step'] >= 1) &
#     (df['step'] <= 100) &
#     (df['intervention'] == 0)
# ]

# # Calculate mean for each step for specific columns
# # "Asupp","Bsupp","ABsupp","Apos","Aneg","A","Bpos","Bneg","B","ABpos","ABneg","AB"
# y0_mean = filt1_df.groupby('step')['Apos'].mean()
# y1_mean = filt1_df.groupby('step')['Aneg'].mean()
# y2_mean = filt1_df.groupby('step')['A'].mean()
# y3_mean = filt1_df.groupby('step')['Bpos'].mean()
# y4_mean = filt1_df.groupby('step')['Bneg'].mean()
# y5_mean = filt1_df.groupby('step')['B'].mean()
# y6_mean = filt1_df.groupby('step')['ABpos'].mean()
# y7_mean = filt1_df.groupby('step')['ABneg'].mean()
# y8_mean = filt1_df.groupby('step')['AB'].mean()

# # Extract step indices
# x = y1_mean.index

# # Define colors from the same color family for each category
# opponent_colors = ['darkred', 'brown', 'firebrick']
# supporter_colors = ['darkgreen', 'seagreen', 'mediumseagreen']
# neutral_colors = ['black', 'dimgray', 'darkgray']

# # Create plot
# fig, ax = plt.subplots(figsize=(8, 5), sharey=True)

# # #####Plot lines with distinct colors for each group, different lines
# # ax.plot(x, y0_mean, 's-', label='Opponents+', color=opponent_colors[0], markersize=2)
# # ax.plot(x, y1_mean, '-.', label='Opponents-', color=opponent_colors[1], markersize=4)
# # ax.plot(x, y2_mean, '-', label='Opponents', color=opponent_colors[2], markersize=4)
# # ax.plot(x, y3_mean, 'o-', label='Supporters+', color=supporter_colors[0], markersize=2)
# # ax.plot(x, y4_mean, '--', label='Supporters-', color=supporter_colors[1], markersize=4)
# # ax.plot(x, y5_mean, '-', label='Supporters', color=supporter_colors[2], markersize=4)
# # ax.plot(x, y6_mean, 'x-', label='Neutrals+', color=neutral_colors[0], markersize=4)
# # ax.plot(x, y7_mean, ':', label='Neutrals-', color=neutral_colors[1], markersize=4)
# # ax.plot(x, y8_mean, '-', label='Neutrals', color=neutral_colors[2], markersize=4)

# # same lines
# ax.plot(x, y0_mean, '-', label='Opponents+', color=opponent_colors[0], lw=2, markersize=2)
# ax.plot(x, y1_mean, '-', label='Opponents-', color=opponent_colors[1], lw=2, markersize=4)
# ax.plot(x, y2_mean, '-', label='Opponents', color=opponent_colors[2], lw=2, markersize=4)
# ax.plot(x, y3_mean, '-', label='Supporters+', color=supporter_colors[0], lw=2,  markersize=2)
# ax.plot(x, y4_mean, '-', label='Supporters-', color=supporter_colors[1], lw=2,  markersize=4)
# ax.plot(x, y5_mean, '-', label='Supporters', color=supporter_colors[2], lw=2,  markersize=4)
# ax.plot(x, y6_mean, '-', label='Neutrals+', color=neutral_colors[0], lw=2,  markersize=4)
# ax.plot(x, y7_mean, '-', label='Neutrals-', color=neutral_colors[1], lw=2,  markersize=4)
# ax.plot(x, y8_mean, '-', label='Neutrals', color=neutral_colors[2], lw=2,  markersize=4)

# # Format subplot
# ax.set_xlim(0.7, 100.3)
# ax.set_ylim(-1, 41)  #30
# ax.set_title("Satisfied(+), Neutral (), or Unsatisfied (-)", fontsize=14)
# ax.set_xlabel("Time/Ticks", fontsize=14)
# ax.set_ylabel("Numbers of individuals", fontsize=14)
# ax.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=10)
# #ax.legend(loc='upper left', fontsize=10, ncol=3) #'right', 'upper left'

# plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to prevent clipping

# #ax.xaxis.set_major_locator(MultipleLocator(10))
# ax.tick_params(axis='both', labelsize=12)  # Change tick label size

# # Show the plot
# plt.show()


#### 满意率，支持率的演化
# #### Satisfaction Rate and Support Rate Evolution
# ### Add background lines for better visual clarity
# # Filter DataFrame based on conditions
# filt1_df = df.loc[
#     (df['ThreAss'] == 0.5) &
#     (df['ThreCon'] == 0.5) &
#     (df['step'] >= 1) &
#     (df['step'] <= 100)
# ]

# # Calculate the average support rate based on the intervention condition
# y0_mean = filt1_df.loc[filt1_df['intervention'] == 0, 'support-rate'].mean()
# y1_mean = filt1_df.loc[filt1_df['intervention'] == 1, 'support-rate'].mean()

# # Extract step indices from the data (adjusting for correct assignment)
# x = filt1_df['step'].unique()

# # Create plot
# fig, ax = plt.subplots(figsize=(8, 5))

# # # Define vibrant colors for the background bands
# # band_colors = ['#FF9E7F', '#87CEFA', '#32CD32']  # Peach, Sky Blue, Lime Green

# # # Add background with alternating color bands
# y_min, y_max = 0, 1  # Set Y-axis limits
# # num_bands = len(band_colors)

# # # Create solid horizontal bands for a more obvious background
# # for i in range(num_bands):
# #     ax.axhspan(
# #         y_min + (i * (y_max - y_min) / num_bands),
# #         y_min + ((i + 1) * (y_max - y_min) / num_bands),
# #         facecolor=band_colors[i % len(band_colors)],
# #         alpha=0.4  # Slight transparency to ensure the background doesn't overpower the plot
# #     )

# # Generate smooth x-axis range for interpolation
# x_smooth = np.linspace(x.min(), x.max(), 500)

# # Plot smoothed curves for each category (intervention 0 and intervention 1)
# ax.plot(x_smooth, np.full_like(x_smooth, y0_mean), label='No Intervention (Support Rate)', color='#FF7F50', lw=2)
# ax.plot(x_smooth, np.full_like(x_smooth, y1_mean), label='Intervention (Support Rate)', color='#32CD32', lw=2)

# # Formatting and labels
# ax.set_xlim(0.7, 100.3)
# ax.set_ylim(y_min, y_max)
# ax.set_title("Satisfaction Rate and Support Rate Evolution", fontsize=14)
# ax.set_xlabel("Time/Ticks", fontsize=14)
# ax.set_ylabel("Satisfaction Rate", fontsize=14)
# ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0., fontsize=11)
# ax.tick_params(axis='both', labelsize=12)

# # Improve layout and show plot
# plt.tight_layout()
# plt.show()


# ###柱状分布图### in one chart, 100 repititions
# # Filter data
# filt1_df = df.loc[
#     (df['ThreAss'] == 0.5) &
#     (df['ThreCon'] == 0.5) &
#     (df['step'] == 100) &
#     (df['intervention'] == 0)
# ]

# # Define bin edges
# bin_edges = np.arange(0.1, 1.1, 0.05)

# # Create figure and axes for subplots
# fig, ax = plt.subplots(figsize=(8, 6))

# # Extract data for histogram
# data = filt1_df['support-rate']  # Assuming 'support-rate' is a column in the CSV

# # Normalize data
# normalized_data = data

# # Calculate mean and standard deviation
# mean_value = normalized_data.mean()
# std_deviation = normalized_data.std()

# # Plot histogram
# sns.histplot(normalized_data, color='blue', bins=bin_edges, kde=True, alpha=0.6, ax=ax)

# # Annotate mean and standard deviation
# ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_value:.2f}')
# ax.axvline(mean_value + std_deviation, color='green', linestyle='dotted', linewidth=1.5, label=f'SD: {std_deviation:.2f}')
# ax.axvline(mean_value - std_deviation, color='green', linestyle='dotted', linewidth=1.5)

# # Add title and labels
# ax.legend(fontsize=10)
# ax.set_xlabel('Support', fontsize=10)
# ax.set_ylabel('Frequency', fontsize=10)
# ax.set_xlim(0, 1.1)
# ax.set_ylim(0, 100)
# ax.xaxis.set_tick_params(pad=10)

# # Adjust layout and display
# plt.tight_layout()
# plt.show()


################################
#### SENSENTIVITY ANALYSIS######
################################
#4. 3D surface for sensitivity analysis
# x_values = np.arange(0.2, 1.1, 0.2)  # Values for ThreAss
# y_values = np.arange(0.2, 1.1, 0.2)  # Values for ThreCon

# # Create a meshgrid from the x and y values
# X, Y = np.meshgrid(x_values, y_values)

# # Filter the DataFrame based on the current strategy and other predefined conditions
# filt1_df = df.loc[
#     (df['step'] == 100) &
#     (df['intervention'] == 0)
# ]

# # Sort the filtered DataFrame by ThreAss and ThreCon
# sorted_df = filt1_df.sort_values(by=["ThreAss", "ThreCon"], ascending=True)

# # Calculate the mean of 'Bsupp' (support) for each combination of ThreAss and ThreCon
# Z_values = sorted_df.groupby(['ThreAss', 'ThreCon'])['satisfaction-rate'].mean().values * 100 #satisfaction-rate ,'support-rate'

# # Reshape the Z values to match the meshgrid
# Z = Z_values.reshape(len(y_values), len(x_values))

# # Create the 3D surface plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the surface with the chosen colormap and styling
# surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8, vmin=65, vmax=75) # vmin=40, vmax=100

# # Add labels and title
# #ax.set_title('Influence of $Thre_{ass}$ & $Thre_{con}$ on Support', fontsize=14)
# ax.set_xlabel('$Thre_{ass}$', fontsize=14)
# ax.set_ylabel('$Thre_{con}$', fontsize=14)
# ax.set_zlabel('$Satisfaction$ $Rate$ (%)', fontsize=14) #"$Support$ $Rate$ (%)", "$Satisfaction$ $Rate$ (%)"
# ax.set_xticks(np.arange(0.1, 1.1, 0.2))  # X-axis increments of 0.2
# ax.set_yticks(np.arange(0.1, 1.1, 0.2))

# # Set the Z-axis limits based on expected range of support values (Bsupp)
# ax.set_zlim(55, 80)    # satifaction (55, 80) , support(0, 104) 

# # Set the view angle for better visualization
# ax.view_init(25, -55) # satisfaction (25, -50), support 

# # Add a color bar to indicate the range of support values
# cbar = fig.colorbar(surface, shrink=0.6, aspect=10, pad=0.05,location='left') #location='left', 'right'

# cbar.set_label('$Satisfaction$ $Rate$ (%)', fontsize=14) #"$Support$ $Rate$ (%)", "$Satisfaction$ $Rate$ (%)"
# cbar.ax.tick_params(labelsize=12)  # Set font size for color bar label
# ax.tick_params(axis='both', labelsize=12)  # Change tick label size

# # Show the plot
# plt.tight_layout()
# plt.show()



##for corelations
# # Filter data for step 100 and no intervention
# filtered_df = df[(df['step'] == 100) & (df['intervention'] == 0)]

# # Compute mean support and satisfaction rates for each (ThreAss, ThreCon) combination
# grouped_df = filtered_df.groupby(['ThreAss', 'ThreCon'])[['support-rate', 'satisfaction-rate']].mean().reset_index()

# # Compute overall correlation
# correlation = grouped_df[['support-rate', 'satisfaction-rate']].corr().iloc[0, 1]

# # Print correlation value
# print(f"Overall Correlation: {correlation:.2f}")

# # Scatter plot to visualize relationship
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=grouped_df['support-rate'], y=grouped_df['satisfaction-rate'], alpha=0.7)

# plt.xlabel('Support Rate (%)', fontsize=12)
# plt.ylabel('Satisfaction Rate (%)', fontsize=12)
# plt.title('Scatter Plot: Support vs. Satisfaction', fontsize=14)

# # Add correlation annotation
# plt.annotate(f'Correlation: {correlation:.2f}', xy=(0.7, 0.1), xycoords='axes fraction', fontsize=12)

# plt.show()


###heatmap for the sensitivity
# Compute mean satisfaction rate for each (ThreAss, ThreCon) combination # support-rate, satisfaction-rate
grouped_df = df.groupby(['ThreAss', 'ThreCon'], as_index=False)['satisfaction-rate'].mean()

# Create pivot table for heatmap
pivot_satisfaction = grouped_df.pivot(index='ThreCon', columns='ThreAss', values='satisfaction-rate')

# Plot heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(pivot_satisfaction, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'size': 12},
                      cbar_kws={'label': 'Satisfaction Rate', 'shrink': 0.8}) # Satisfaction Rate, 'Support Rate'

# Adjust colorbar label and tick font size
colorbar = heatmap.collections[0].colorbar

colorbar.set_label("$Satisfaction$ $Rate$", fontsize=14) #"$Satisfaction$ $Rate$" , $Support$ $Rate$"
colorbar.ax.tick_params(labelsize=13)

# Set axis labels and tick font sizes
plt.xlabel('$Thre_{Con}$', fontsize=14)
plt.ylabel('$Thre_{Ass}$', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# Improve layout
plt.tight_layout()
plt.show()

# # Plot overall correlation heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(), annot=True, linewidth=0.5, cmap='coolwarm')
# plt.title('Heatmap: Correlation Matrix', fontsize=14)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()
# plt.show()



## 2D heatmap for sensitivity analysis
# ##Define the values for ThreAss and ThreCon (input parameters for sensitivity analysis)
# x_values = np.arange(0.1, 1.01, 0.2)  # Values for ThreAss
# y_values = np.arange(0.1, 1.01, 0.2)  # Values for ThreCon

# # Create a meshgrid for ThreAss and ThreCon
# X, Y = np.meshgrid(x_values, y_values)

# # Filter the DataFrame based on specific conditions (step = 100, intervention = 0)
# filt1_df = df.loc[
#     (df['step'] == 100) &
#     (df['intervention'] == 0)
# ]

# # Sort the filtered DataFrame by ThreAss and ThreCon
# sorted_df = filt1_df.sort_values(by=["ThreAss", "ThreCon"], ascending=True)

# # Calculate the mean of 'Bsupp' (support) for each combination of ThreAss and ThreCon
# Z_values = sorted_df.groupby(['ThreAss', 'ThreCon'])['support-rate'].mean().values #satisfaction-rate,'support-rate'
# Z1_values = sorted_df.groupby(['ThreAss', 'ThreCon'])['satisfaction-rate'].mean().values #satisfaction-rate,'support-rate'

# # Reshape the Z values to match the meshgrid
# Z = Z_values.reshape(len(y_values), len(x_values))

# # Create the 2D heatmap plot
# fig, ax = plt.subplots(figsize=(10, 8))

# # Plot the heatmap with the chosen colormap and styling
# cax = ax.pcolormesh(X, Y, Z, cmap='viridis', shading='auto', vmin=0.8, vmax=1)  # satisfaction 0.55, 0.85

# # Add labels and title
# ax.set_title('Influence of $Thre_{ass}$ & $Thre_{con}$ on Support', fontsize=14) # satisfaction, Support
# ax.set_xlabel('$Thre_{ass}$', fontsize=14)
# ax.set_ylabel('$Thre_{con}$', fontsize=14)

# # Set the colorbar to indicate the range of support values
# cbar = fig.colorbar(cax, shrink=0.5, aspect=10)
# cbar.set_label('$Support$', fontsize=14)

# # Show the plot
# plt.tight_layout()
# plt.show()