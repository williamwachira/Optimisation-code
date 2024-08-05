from scipy. optimize import curve_fit
from scipy. stats import linregress
from sympy import symbols, log, integrate, I, re
import scipy. stats as stats
import numpy as np
import matplotlib. pyplot as plt
import seaborn as sns
from reportlab. pdfgen import canvas
import math
import datetime
from reportlab.lib import pdfencrypt
# saving to pdf
#**********************pdf generation********************************
# Creating a PDF file
pdf_file_path = "output_report.pdf"
#enc=pdfencrypt. StandardEncryption ("william”, canPrint=0)
pdf_canvas = canvas. Canvas(pdf_file_path)
# Setting the page size
page_width, page_height = pdf_canvas. Pagesize
# Maximum number of lines to print on each page
max_lines_per_page = 120
# Initializing variables for tracking space on the current page
current_line = 0
current_page = 1
# Custom print function to write to the PDF canvas
def custom_print (*args, **kwargs):
    global current_line, current_page
    """
    the map function below iterates through the various 
    items(elements) of args and applies the str function 
    to individual elements.
    """
    text = ''. join (map (str, args))

    # Check if there's enough space on the current page
    if current_line >= max_lines_per_page:
        # Start a new page
        pdf_canvas. showPage ()
        current_line = 0
        current_page += 1
# Write the text to the PDF canvas
    pdf_canvas. drawString (5, page_height - 30 - current_line * 7, text)
    current_line += 1
# Replacing the default print function with the custom one
original_print = print
print = custom_print
#**************************end of pdf generation****************************
# A variable holding a scalar value for concatenating the string 
spacing=150
def logarithim_curve (x, a, b, c):
    """
    function takes in values of x and constants as parameters which then 
    computes the corresponding dependent variables (fitted data) to help 
    in plotting the fitted curve
    """
    # expression to be masked
    x_offset=x+b
    #print ("x_offset values: --> “, x_offset)
    #All elements of the list less than 0 are ignored during log computation
    x_offset [x_offset <= 0] = np.nan
    return a*np.log10(x_offset) +c
try:
    #curve fit method returns constants a, b, c and a covariance matrix to assist in computing the variance and standard deviation
    # it takes in the fitted curve and the x and y data set
    params, covariance= curve_fit (logarithim_curve, time_series_data, gear_ratio_data)
except RuntimeError as e:
    #Print the values of parameters causing the warning
    print ("Values causing the Runtime Warning:")
    raise e # Re-raise the warning to get the full traceback
#Print the parameters (a, b, c)
print ("Optimised Parameters (a, b, c):", params)
if any (params <= 0):    
    print ("Invalid parameters in the fitted curve.")
# creates an iteration of all fitted data
fitted_data=logarithim_curve (time_series_data, *params)
#****************end of basic log curve fitting******************************
"""
#---------polynomial with logarithm term function block--------------------
def polynomial_with_log_term (x, a, b, c):
    return a*(np. power (x, b) *np.log(x+c))

params, covariance=
curve_fit (polynomial_with_log_term, time_series_data, gear_ratio_data)
#printing the parameters (a, b, c, d)
print ("parameters: a, b, c:", params)
fitted_data=polynomial_with_log_term (time_series_data, *params)
#*******************end****************************************************
"""
"""
#-----------exponential with logarithm term function block--------------
def exponential_with_log_term (x, a, b, c):
    return a*(np.exp (np.log (x + b)) + c)
params, covariance= curve_fit (exponential_with_log_term, time_series_data, gear_ratio_data)
#printing the parameters (a, b, c, d)
print ("parameters: a, b, c:", params)
fitted_data=exponential_with_log_term (time_series_data, *params)
#****************end*****************************************
"""
#*********end of calculating the R squared********
#***********calculating correlation*****************************
# Calculate the correlation coefficients
# to check how the variables are sensitive to each other
correlation_coefficient = np. corrcoef (Pinion_Angular_displacement, gear_ratio_data) [0, 1]
correlation_coefficient_of_RackDisplacement_and_angular_displacement=np.corrcoef (Pinion_Angular_displacement, Rack_displacement_1) [0, 1]
correlation_coefficient_of_gear_ratio_and_rack_displacement = np. corrcoef (gear_ratio_data, Rack_displacement_1) [0, 1]
print (f"Correlation Coefficient of gear ratio and angular displacement: {correlation_coefficient:.4f}")
print (f"Correlation Coefficient of rack displacement and angular displacement: {correlation_coefficient_of_RackDisplacement_and_angular_displacement:.4f}")
print (f"Correlation Coefficient of rack displacement and gear ratio: {correlation_coefficient_of_gear_ratio_and_rack_displacement:.4f}")
#**********************end****************************************
#******calculating the R squared of the curve fitting*****
residuals = gear_ratio_data - fitted_data
ss_res = np.sum(residuals**2)
ss_tot = np.sum ((gear_ratio_data - np. mean(gear_ratio_data)) **2)
r_squared = 1 - (ss_res / ss_tot)
# Print the R-squared value
print ("R-squared for the fitted curve is:", r_squared)
if np. isnan(fitted_data). any () or np. isinf(fitted_data). any ():
    print ("Invalid values in the fitted curve.")
# Print standard deviations (square root of diagonal elements of covariance matrix)
std_dev = np. sqrt (np. diag(covariance))
#printing the covariance of the matrix
print ("The covariance of the parameters a, b, c are: ")
print (f"Covariance of a: {np. diag(covariance) [0]} ")
print (f"Covariance of b: {np. diag(covariance) [1]}")
print (f"Covariance of c: {np. diag(covariance) [2]} ")
print ("Standard Deviations of a, b, c respectively:", std_dev)
# Calculate standard errors from the diagonal of the covariance matrix
parameter_std_dev = np. sqrt (np. diag(covariance))
# Calculate t-scores for each parameter
t_scores = params / parameter_std_dev
# Calculate degrees of freedom (df) for the t-distribution
# For curve fitting, df is usually equal to the number of data points minus the number of parameters
df = len(time_series_data) - len(params)
# Calculate two-tailed p-values for each parameter using the t-distribution
p_values = 2 * (1 - stats.t.cdf(np.abs(t_scores), df))
print (f"p_values: {p_values}")
# Output parameter estimates, standard errors, t-scores, and p-values
for i, param_name in enumerate (['a', 'b', 'c']):
    print(f"{param_name}: Estimate={params[i]:.4f}, SE={parameter_std_dev[i]:.4f}, t={t_scores[i]:.4f}, \
           p={p_values[i]: e}")
#calculating the average gear ratio with the help of python integrate function
a, b, c =params
# Define the symbol and the curve fit equation
t = symbols('t')
equation = a * log (t + b) + c # Replace with your curve fit coefficients
# Define the limits of integration
t_lower_limit = 0.0
t_upper_limit = 0.8
# Calculate the average value
average_value = (1 / (t_upper_limit - t_lower_limit)) * integrate (equation, (t, t_lower_limit, t_upper_limit))
# Print the result
print (f'Average gear ratio in the range {t_lower_limit}s to {t_upper_limit}s is: {average_value}')
#***defining the parameters for calculating the constant c of the dimensional equation******
G=re(average_value). evalf ()
average_radius=12/2 #total displacement is 12 but considering displacement of 1 finger equal to radius
maximum_plant_strain=0.48 # from literature
radius_of_pinion=34.5/2 # design pitch diameter of the pinion is 34.5mm
total_cross_section_area=round(math.pi*average_radius*average_radius,2)
# Get the line properties
#line = plt.gca ()
slope, intercept = np. polyfit (Pinion_Angular_displacement_1, Rack_displacement, 1) # Using np. polyfit to get coefficients
# Print coefficients
print (f'The value of y Intercept: {intercept:.4f}')
print (f'The value of Slope: {slope:.4f}')
#*******************end*************************************
def constant_c (s, g, strain, area, pinion_radius):
    """
    this function takes in the slope, gear ratio, maximum strain, 
    area and pinion radius and returns the constant c
    """
    return(s/((g*strain*area)/pinion_radius))
constant= (constant_c (slope, G, maximum_plant_strain, total_cross_section_area, radius_of_pinion))
print (f"Average radius: {average_radius}")
print (f"Maximum tomato strain: {maximum_plant_strain}")
print (f"Total cross section area: {total_cross_section_area}")
print (f'The constant c is: {constant}')
#*************end of the calculation*****************
#*********validating the model*************
#predicted rack displacement values
#pinion angular displacement
predicted_displacement=(float(constant)*float(G)*maximum_plant_strain*total_cross_section_area*Pinion_Angular_displacement_1)/radius_of_pinion
#simulated rack displacement values
validation_correlation=np. corrcoef (predicted_displacement, Rack_displacement) [0,1]
print (f"validation correlation value: {validation_correlation:.10f}")
#calculating the R squared to validate the model
# Calculate correlation coefficient
correlation_matrix = np. corrcoef (predicted_displacement, Rack_displacement)
correlation_coefficient = correlation_matrix [0, 1]
# Calculate R squared evaluate prediction model
r_squared = correlation_coefficient**2
print ("Validation of prediction model with R squared using corrcoef ():", r_squared)
#*******or by using the linregress function*****************
# Calculate linear regression
slope, intercept, r_value, p_value, std_err = linregress (Rack_displacement, predicted_displacement)
# Calculate R squared
r_squared = r_value**2
print ("Validation of prediction model with R squared using linregress (): ", r_squared)
print ("Validation of prediction model with standard error using linregress (): ", std_err)
print ("slope: ", slope)
print ("intercept", intercept)
#***************end of model validation***********************
print = original_print
# Save and close the PDF file
pdf_canvas. save ()
#PLOTS
# Create subplots
    #*************plot of curve fit*********************************
"""
fig, axes = plt. subplots (nrows=1, ncols=3, figsize= (20, 4))
axes [0]. scatter (time_series_data, gear_ratio_data, label='Simulation data', color='red')
axes [0]. plot (time_series_data, fitted_data, label="Logarithmic Fit", color='green')
#********************plot of correlation********************************
axes [1]. scatter (Pinion_Angular_displacement_1, Rack_displacement)
axes [1]. plot (Pinion_Angular_displacement_1, Rack_displacement) 
"""
#************************heatmap*****************************************
#**********restoring original print*******************
# Restoring the original print function
#print = original_print
# Save and close the PDF file
#pdf_canvas. save ()
print (f"validation correlation: {validation_correlation:.10f}")
# Visualizing the covariance matrix using seaborn
"""
sns.set(style="whitegrid”) # Set background style
axes [2] = sns. heatmap (covariance, annot=True, cmap="coolwarm", fmt=".10f", linewidths=.5)
# Set labels and title using seaborn
axes [2]. set_title ('Covariance Matrix Heatmap')
axes [2]. set_xticklabels (['a', 'b', 'c'])
axes [2]. set_yticklabels (['a', 'b', 'c'])
axes [2]. set_xlabel('constants')
axes [2]. set_ylabel('constants')
"""
"""
#*****************end****************************************
# Display the equation as text at the center of the plot
equation_text = f'$y = {params[0]:.2f} \cdot \log (t {params[1]:.2f}) + {params[2]:.2f} $'
param_R_squared_text=f"R Squared: {r_squared}"
axes [0].text (np. mean(time_series_data), np. mean(fitted_data)-0.18, equation_text,
         fontsize=10, color='blue', ha='center', va='center')
axes [0].text (np. mean(time_series_data), np. mean(fitted_data)-0.25, param_R_squared_text,
         fontsize=10, color='blue', ha='center', va='center')
axes [0]. set_title ("A GRAPH OF LOGARITHIMIC CURVE-FIT")
axes [0]. set_xlabel ("Time (seconds) ")
axes [0]. set_ylabel ("Gear ratio")
axes [0]. grid (True)
axes [0]. legend ()
axes [1]. set_title ("A GRAPH OF RACK DISPLACEMENT AHAINST PINION ANGULAR DISPLACEMENT")
axes [1]. set_xlabel ("PINION ANGULAR DISPLACEMENT (radians)")
axes [1]. set_ylabel ("RACK DISPLACEMENT (mm)")
axes [1]. grid (True)
"""
"""
#plotting individual graphs
# Display the equation as text at the center of the plot
equation_text = f'$y = {params[0]:.2f} \cdot \log (t {params[1]:.2f}) + {params[2]:.2f} $'
param_R_squared_text=f"R Squared: {r_squared}"
plt. scatter (time_series_data, gear_ratio_data, label='Simulation data', color='red')
plt. plot (time_series_data, fitted_data, label="Logarithmic Fit", color='green')
plt.text (np. mean(time_series_data), np. mean(fitted_data)-0.15, equation_text,
         fontsize=20, color='blue', ha='center', va='center')
plt.text (np. mean(time_series_data), np. mean(fitted_data)-0.2, param_R_squared_text,
         fontsize=20, color='blue', ha='center', va='center')
plt. title ("A GRAPH OF LOGARITHIMIC CURVE-FIT", fontsize=20)
plt. xlabel ("Time (seconds) ", fontsize=20)
plt. ylabel ("Gear ratio", fontsize=20)
plt. grid (True)
plt. legend ()
"""
#********************plot of correlation********************************
"""
plt. scatter (Pinion_Angular_displacement_1, Rack_displacement)
plt. plot (Pinion_Angular_displacement_1, Rack_displacement) 
plt. title ("A GRAPH OF RACK DISPLACEMENT AGAINST PINION ANGULAR DISPLACEMENT”, fontsize=20)
plt. xlabel ("PINION ANGULAR DISPLACEMENT (radians)”, fontsize=20)
plt. ylabel ("RACK DISPLACEMENT (mm)”, fontsize=20)
plt. grid (True)
# Visualizing the covariance matrix using seaborn
"""
sns.set(style="whitegrid”) # Set background style")
sea_born = sns. heatmap (covariance, annot=True, cmap="coolwarm", fmt=".2e", linewidths=.5, annot_kws= {"fontsize": 20})
# Set labels and title using seaborn
sea_born.set_title ('Covariance Matrix Heatmap', fontsize=20)
sea_born.set_xticklabels (['a', 'b', 'c'], fontsize=20)
sea_born.set_yticklabels (['a', 'b', 'c'], fontsize=20)
sea_born.set_xlabel ('constants', fontsize=20)
sea_born.set_ylabel ('constants', fontsize=20)
#end of plotting individual graphs
plt. tight_layout ()
# Show the plots
plt. show ()
