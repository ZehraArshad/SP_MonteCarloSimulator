# backend/simulation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Define the profit function
def getprofit(VS, VM, VL, PS, PM, PL, VCake, VChoc, PCake, PChoc, Max):
    profit = VS * PS + VM * PM + VL * PL + VCake * PCake + VChoc * PChoc
    if VS + VL + VM > Max:
        # profit is reduced to 90% of the original profit
        profit *= 0.9
    return profit

def run_simulation(n_days):
    # Define parameters
    VS, VM, VL = 20, 15, 10 #average order
    XCake, XChoc = 0.05, 0.10 #additional % to show that 5% could be the chance of them being cake
    # Simulate values for quantities
    xVS = np.random.poisson(VS, n_days) # poisson for a 
    xVM = np.random.poisson(VM, n_days)
    xVL = np.random.poisson(VL, n_days)
    xVCake = np.random.binomial(xVS + xVM + xVL, XCake, n_days) #cake or not
    xVChoc = np.random.binomial(xVS + xVM + xVL, XChoc, n_days) # choc or not

    # Create DataFrame
    df = pd.DataFrame({
        'Qty_Small': xVS,
        'Qty_Medium': xVM,
        'Qty_Large': xVL,
        'Price_Small': 5,
        'Price_Medium': 7,
        'Price_Large': 10,
        'Qty_Cake': xVCake,
        'Qty_Choc': xVChoc,
        'Price_Cake': 10,
        'Price_Choc': 3,
        'Qty_penalty': 50
    })

    # Define profit calculation function
    def getprofit1(row):
        VS = row['Qty_Small']
        VM = row['Qty_Medium']
        VL = row['Qty_Large']
        PS = row['Price_Small']
        PM = row['Price_Medium']
        PL = row['Price_Large']
        VCake = row['Qty_Cake']
        VChoc = row['Qty_Choc']
        PCake = row['Price_Cake']
        PChoc = row['Price_Choc']
        Max = row['Qty_penalty']

#row represent each day
        profit = VS * PS + VM * PM + VL * PL + VCake * PCake + VChoc * PChoc
        if VS + VL + VM > Max:
            profit *= 0.9
        return profit

    # Apply the function and get results for profits (avg, total)
    df['Profit'] = df.apply(getprofit1, axis=1)
    total_profit = df['Profit'].sum()
    average_profit = df['Profit'].mean()

    # Generate plot for histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Profit'], bins=30, kde=False, color='#e3a346')
    mu = df['Profit'].mean()
    sigma = df['Profit'].std()
    x = np.linspace(df['Profit'].min(), df['Profit'].max(), 100)
    plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) ), color='#de7354')
    plt.axvline(mu, color='black', linestyle='dashed', linewidth=2)
    plt.title('Histogram of Simulated Profit')
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plt.legend(['Gaussian Curve', 'Mean'])
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # Sensitivity analysis
    Xvals = {
        'QtySmall': 20,
        'QtyMedium': 15,
        'QtyLarge': 10,
        'PriceSmall': 5,
        'PriceMedium': 7,
        'PriceLarge': 10,
        'QtyCake': 0.05 * (20 + 15 + 10),
        'QtyChoc': 0.10 * (20 + 15 + 10),
        'PriceCake': 10,
        'PriceChoc': 3,
        'QtyPentalty': 50
    }

    S = sensitivity_analysis(getprofit, Xvals)

    # Convert the sensitivity analysis DataFrame to a dictionary
    sensitivity_results = S.to_dict(orient='list')

    # Return results
    return {
        'total_profit': total_profit,
        'average_profit': average_profit,
        'daily_profits': df['Profit'].tolist(),
        'plot_image': image_base64,
        'sensitivity': sensitivity_results,
    }

# Sensitivity analysis function
def sensitivity_analysis(FUN, X):
    n = len(X)
    xnames = list(X.keys())
    xvals = list(X.values())

    y = FUN(*xvals)
    sens = np.zeros(n)
    for i in range(n):
        new_xvals = xvals.copy()
        new_xvals[i] *= 1.01  # Perturb each price and quantity by 1%
        new_y = FUN(*new_xvals)
        sens[i] = 100 * (new_y - y) / (y * 0.01)  # Sensitivity calculation as percentage
    index = np.argsort(np.abs(sens))
    xnames_sorted = [xnames[idx] for idx in reversed(index)]
    sens_sorted = [sens[idx] for idx in reversed(index)]
    
    df = pd.DataFrame({'Variable': xnames_sorted, 'PercentSens': sens_sorted})
    
    return df

