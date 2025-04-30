import csv
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

# df1 = pd.read_csv("SustainabilityS1.csv")
# df = pd.read_csv("SustainabilityS3.csv")
df = pd.read_csv("SustainabilitySensitivity.csv")

#Figure 5, 7 9, use data df1 = pd.read_csv("SustainabilityS1.csv") and df = pd.read_csv("SustainabilityS3.csv")
##add line to the background
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


# ####Figure 10: Comparison of the Satisfaction and support Rate
# # df1 = pd.read_csv("SustainabilityS1.csv") and df = pd.read_csv("SustainabilityS3.csv")
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

# # Add a line connecting the bars           #label="$Support$ $Rate$"
# ax.plot(x_positions, values, linestyle='--', color='grey', marker='o', markersize=6, label="$Satisfaction$ $Rate$")

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


##for Figure 11,sensitivity analysis use data df = pd.read_csv("SustainabilitySensitivity.csv")
###heatmap for the sensitivity
# Compute mean satisfaction rate for each (ThreAss, ThreCon) combination # support-rate, satisfaction-rate
df = df[(df['intervention'] == 1) ]  
grouped_df = df.groupby(['ThreAss', 'ThreCon'], as_index=False)['satisfaction-rate'].mean()

# Create pivot table for heatmap
pivot_satisfaction = grouped_df.pivot(index='ThreCon', columns='ThreAss', values='satisfaction-rate')

# Plot heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(pivot_satisfaction, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'size': 12},
                      cbar_kws={'label': 'Support Rate', 'shrink': 0.8}) # Satisfaction Rate, 'Support Rate'

# Adjust colorbar label and tick font size
colorbar = heatmap.collections[0].colorbar

colorbar.set_label("$Satisfaction$ $Rate$", fontsize=14) #"$Satisfaction$ $Rate$" , $Support$ $Rate$"
colorbar.ax.tick_params(labelsize=13)

# Set axis labels and tick font sizes
plt.xlabel('$\mu(SI)$', fontsize=14)
plt.ylabel('$\mu(SA)$', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# Improve layout
plt.tight_layout()
plt.show()
