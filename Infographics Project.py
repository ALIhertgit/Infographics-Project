#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv("C:/Users/ali10/OneDrive/Desktop/Task-02/country_population.csv")

# Select specific countries
selected_countries = df[df['Country Name'].isin(['Russia', 'Germany', 'United Kingdom', 'France', 'Italy', 'Spain', 'Ukraine'])]

# Compute population growth for the selected years
selected_years = ['1965', '1975', '1985', '1995', '2005', '2015']
population_growth_data = selected_countries[selected_years].pct_change(axis=1) * 100

# Convert population to millions
selected_countries['2015 (Millions)'] = selected_countries['2015'] / 1e6  # For Plot 2
selected_countries['1965 (Millions)'] = selected_countries['1965'] / 1e6  # For Plot 4

# Set background color
background_color = '#f5f5f5'  # Light background color

# Create the figure with specific dimensions and background color
plt.figure(figsize=(12, 21), facecolor=background_color)
plt.subplots_adjust(hspace=0.5)  # Decreased vertical space

# Plot 1: Population Distribution in 2015 (Pie Chart)
plt.subplot(5, 1, 1)
colors_pie = sns.color_palette("dark", n_colors=len(selected_countries))
explode = tuple(0.1 for _ in range(len(selected_countries)))
plt.pie(selected_countries['2015'], labels=selected_countries['Country Name'], autopct='%1.1f%%', colors=colors_pie, startangle=140, explode=explode, textprops=dict(color="w"))
plt.title('Population Distribution in 2015', fontsize=14, color='black')
plt.legend(selected_countries['Country Name'], title="Country", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.annotate("This pie chart illustrates the distribution of population in selected countries for the year 2015.",
             xy=(1, -1.2), xytext=(0, -10), textcoords='offset points',
             ha='center', va='center', fontsize=12, color='black')

# Plot 2: Population Distribution in 1965 and 2015 (Vertical Bar Plot)
plt.subplot(5, 1, 2)
colors_bar = sns.color_palette("dark", n_colors=len(selected_countries))
population_1965_2015_bar = selected_countries[['Country Name', '1965 (Millions)', '2015 (Millions)']]
population_1965_2015_bar = population_1965_2015_bar.melt(id_vars='Country Name', var_name='Year', value_name='Population (Millions)')
sns.barplot(x='Country Name', y='Population (Millions)', hue='Year', data=population_1965_2015_bar, palette=['darkblue', 'darkgreen'])
plt.title('Population Distribution in 1965 and 2015', color='black')
plt.xlabel('Country')
plt.ylabel('Population (Millions)')
plt.annotate("This vertical bar plot represents the population distribution in selected countries for the years 1965 and 2015.",
             xy=(1, 1), xytext=(260, -50), textcoords='offset points',
             ha='center', va='center', fontsize=12, color='black')

# Plot 3: Line Chart for Population Growth in 1965, 1975, 1985, 1995, 2005, and 2015
plt.subplot(5, 1, 3)
plt.plot(selected_years, population_growth_data.transpose(), marker='o')
plt.title('Population Growth Rate (1965-2015) - Line Chart', color='black')
plt.xlabel('Year')
plt.ylabel('Population Growth Rate (%)')
plt.legend(selected_countries['Country Name'], title="Country", bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 4: Horizontal Bar Graph for Population in 1965 and 2015
plt.subplot(5, 1, 4)
bar_width = 0.35
bar_positions = np.arange(len(selected_countries))
plt.barh(bar_positions - bar_width/2, selected_countries['1965 (Millions)'], bar_width, label='1965', color='darkblue')
plt.barh(bar_positions + bar_width/2, selected_countries['2015 (Millions)'], bar_width, label='2015', color='darkgreen')
plt.yticks(bar_positions, selected_countries['Country Name'])
plt.title('Population in 1965 and 2015', color='black')
plt.xlabel('Population (Millions)')
plt.ylabel('Country')
plt.legend()

# Overall title
plt.suptitle('Population Analysis Infographics\nName: M. Siddiqui\nStudent ID: 22082227', fontsize=16, color='black')

# Summary text
summary_text = """
The report contains data for the six European countries throughout the years. 
It can be seen that Germany was the most populous country in 2015, followed by France and the UK. 
The population growth graph shows interesting insights into the population growth trend. 
It can be seen that in the year 1965 Spain has the fastest-growing rate. 
Also, it can be noted that in the later years, Ukrainian population growth was negative, which means that the population was decreasing.
Lastly, it can be seen that the countries with the highest deviation in the population in 1965 and 2015 are Spain and France, which was evident from the growth rate.
"""

# Add summary text below the plots
plt.figtext(0.1, -0.3, summary_text, ha='left', va='center', fontsize=12, color='black')  # Adjusted vertical space

# Save the visualisation as PNG
plt.savefig("22082227_selected_countries_with_line_chart_and_bar_graph_summary.png", dpi=300, facecolor=background_color)

# Show the plots
plt.show()


# In[ ]:




