from flask import Flask, render_template, request
import pandas_datareader as pdr
import pandas as pd
from datetime import date, datetime, timedelta
import yfinance as yf
print("Running yfinance version:", yf.__version__)
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    var = ""
    quarterly_chart_html = ""
    annual_chart_html = ""
    nonlinearmodel = ""
    linearmodel = ""
    price_text_11 = ""
    price_text_12 = ""
    price_text_13 = ""
    price_text_14 = ""
    price_text_15 = ""
    price_text_21 = ""
    price_text_22 = ""
    price_text_23 = ""
    price_text_24 = ""
    price_text_25 = ""
    price_text_26 = ""
    nonlinear_chart_html = ""
    linear_chart_html = ""
    processed_text = None  # Renamed for clarity
    if request.method == "POST":
        # Get the ticker symbol from the form
        text_input = request.form.get("ticker")

        # Process the text (for example, convert it to uppercase)
        ticker = text_input.upper()
        print("")  # Debugging: Print the processed text to the console
        print(ticker)

        API_KEY = "11"

        url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        print("-----DATA-----")
        print(data)
        print("--------------")

        def get_quarterly_freecashflows(data):
            cashflow_quaterly = []
            date_quaterly = []
            quarterly_cashflows = data.get("quarterlyReports", [])
            for report in quarterly_cashflows:
                if report.get('operatingCashflow', 0) == 'None':
                    operating_cashflow = 0
                else:
                    operating_cashflow = float(report.get('operatingCashflow', 0))
                if report.get('capitalExpenditures', 0) == 'None':
                    capital_expenditures = 0
                else:
                    capital_expenditures = float(report.get('capitalExpenditures', 0))
                fcf = operating_cashflow - capital_expenditures
                cashflow_quaterly.append(fcf)
                date_quaterly.append(report['fiscalDateEnding'])
            cashflow_quaterly.reverse()
            date_quaterly.reverse()
            return cashflow_quaterly, date_quaterly

        def get_annual_freecashflows(data):
            annual_cashflows = data.get("annualReports", [])
            cashflow_annual = []
            date_annual = []
            for report in annual_cashflows:
                if report.get('operatingCashflow', 0) == 'None':
                    continue
                else:
                    operating_cashflow = float(report.get('operatingCashflow', 0))
                if report.get('capitalExpenditures', 0) == 'None':
                    continue
                else:
                    capital_expenditures = float(report.get('capitalExpenditures', 0))
                fcf = operating_cashflow - capital_expenditures
                cashflow_annual.append(fcf)
                date_annual.append(report['fiscalDateEnding'][:4])
            cashflow_annual.reverse()
            date_annual.reverse()
            date_annual_str = []
            for element in date_annual:
                date_annual_str.append(int(element))
            return cashflow_annual, date_annual_str

        def get_annual_freecashflows_yfinance(ticker):
            stock = yf.Ticker(ticker)

            cashflow = stock.cashflow

            fcf = cashflow.loc["Free Cash Flow"]

            fcf_dates = [date.year for date in fcf.index]
            fcf_values = fcf.values.tolist()
            fcf_dates.reverse()
            fcf_values.reverse()
            return fcf_dates, fcf_values

        def add_cashflows(years_list, cashflows_list, yfinance_years_list, yfinance_cashflows_list):
            for i in range(len(yfinance_years_list)):
                if yfinance_years_list[i] not in years_list:
                    years_list.append(yfinance_years_list[i])
                    cashflows_list.append(yfinance_cashflows_list[i])
            return years_list, cashflows_list

        def add_chart(x, y, color, name, title):
            fig = go.Figure()
            for i in range(len(y)):
                fig.add_traces(go.Scatter(x=x[i], y=y[i], mode='lines', line=dict(color=color[i], width=3), name=name[i]))
            fig.update_layout(title=title)
            return fig.to_html(full_html=False)

        quarterly_freecashflow = get_quarterly_freecashflows(data)
        annual_freecashflow = get_annual_freecashflows(data)

        annual_freecashflows_yfinance = get_annual_freecashflows_yfinance(ticker)

        print(annual_freecashflows_yfinance[0])
        print(annual_freecashflows_yfinance[1])

        updated_cashflows_and_dates = add_cashflows(annual_freecashflow[1], annual_freecashflow[0], annual_freecashflows_yfinance[0], annual_freecashflows_yfinance[1])
        print(updated_cashflows_and_dates[0])
        print(updated_cashflows_and_dates[1])

        quarterly_chart_html = add_chart([quarterly_freecashflow[1]], [quarterly_freecashflow[0]], ['red'], ['FCF'], 'Free Cash Flow Growth Rate Over Quarters')
        annual_chart_html = add_chart([updated_cashflows_and_dates[0]], [updated_cashflows_and_dates[1]], ['blue'], ['FCF'], 'Free Cash Flow Growth Rate Over Years')

        def get_cost_of_equity(rf, rm, beta):
            return (rf + beta*(rm-rf))-1

        def get_risk_free_rate():
            syms = ['DGS10']
            ir_data = pdr.DataReader(syms, 'fred', datetime(2006, 12,1),datetime.today())
            riskfreeratechart = ir_data.iloc[[-1]]
            dictionary = riskfreeratechart.to_dict(orient='records')
            riskfreerate = dictionary[0]["DGS10"]
            riskfreerate = riskfreerate/100 +1

            return riskfreerate 

        def get_stock_beta(ticker):
            stock = yf.Ticker(ticker)
            beta = stock.info.get("beta", None)
            if beta is None:
                raise ValueError("Beta not available for this stock.")
            return beta

        def get_stock_market_cap(ticker):
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get("marketCap", None)
            if market_cap is None:
                raise ValueError("Market capitalization not available for this stock.")
            return market_cap

        def get_total_debt(ticker):
            stock = yf.Ticker(ticker)
            balance_sheet = stock.quarterly_balance_sheet
            latest_quarter = balance_sheet.columns[0]
            total_debt = balance_sheet.loc["Total Debt", latest_quarter]
            if total_debt is None:
                raise ValueError("Total debt not available for this stock.")
            return total_debt

        def get_tax_rate(ticker):
            stock = yf.Ticker(ticker)
            income_statement = stock.financials
            tax_rate = None
            income_before_tax = income_statement.loc["Pretax Income"].iloc[0]
            income_tax_expense = income_statement.loc["Tax Provision"].iloc[0]
            if income_before_tax and income_tax_expense:
                tax_rate = income_tax_expense / income_before_tax
            if tax_rate is None:
                raise ValueError("Tax rate not available for this stock.")
            return tax_rate

        def get_cost_of_debt(ticker, market_return):
            stock = yf.Ticker(ticker)
            income_statement = stock.financials
            interest_expense = income_statement.loc["Interest Expense"].iloc[0]
            total_debt = get_total_debt(ticker)
            cost_of_debt = abs(interest_expense) / total_debt if interest_expense else None
            if pd.isna(cost_of_debt):
                cost_of_debt = (get_cost_of_equity(get_risk_free_rate(), market_return, get_stock_beta(ticker)) + (get_risk_free_rate()-1))/2
            return cost_of_debt

        def calculate_wacc(ticker, market_return):
            cost_of_equity = get_cost_of_equity(get_risk_free_rate(), market_return, get_stock_beta(ticker))
            wacc = ((get_stock_market_cap(ticker) / (get_total_debt(ticker) + get_stock_market_cap(ticker))) * cost_of_equity) + ((get_total_debt(ticker) / (get_total_debt(ticker) + get_stock_market_cap(ticker))) * get_cost_of_debt(ticker, market_return) * (1 - get_tax_rate(ticker)))
            return wacc

        def get_total_cash(ticker):
            stock = yf.Ticker(ticker)
            balance_sheet = stock.quarterly_balance_sheet
            latest_quarter = balance_sheet.columns[0]
            cash = balance_sheet.loc["Cash And Cash Equivalents", latest_quarter]
            return cash

        def get_shares_issued(ticker):
            stock = yf.Ticker(ticker)
            balance_sheet = stock.quarterly_balance_sheet
            latest_quarter = balance_sheet.columns[0]
            shares = balance_sheet.loc["Share Issued", latest_quarter]
            return shares

        market_return = 1.08
        wacc = calculate_wacc(ticker, market_return)
        print(f"Risk Free Rate {get_risk_free_rate()}")
        print(f"coe {get_cost_of_equity(get_risk_free_rate(), market_return, get_stock_beta(ticker))}")
        print(f"Market CAP {get_stock_market_cap(ticker)}")
        print(f"TAX {get_tax_rate(ticker)}")
        print(f"Debt {get_total_debt(ticker)}")
        print(f"Cost of debt {get_cost_of_debt(ticker, market_return)}")
        print(f"Cash & Cash equivalents {get_total_cash(ticker)}")
        print(f"Shares issued {get_shares_issued(ticker)}")

        print(f"The WACC for {ticker} is: {wacc:.4f} or {wacc * 100:.2f}%")
        var = f"The WACC for {ticker} is: {wacc:.4f} or {wacc * 100:.2f}%"

        def nth_root_of_negative(number, n):
            if number < 0:
                if n%2==0:
                    return (((number*-1) ** (1/n))*-1)
                else:
                    return ((number* -1) ** (1/n)) * -1
            else:
                return number ** (1/n)
            
        def geometric_average_percantage_return(endingvalue, initialvalue, years):
            cal = (nth_root_of_negative((endingvalue/initialvalue), (years)) - 1)
            growth_rate_list = [cal*0.5, cal*0.75, cal, cal*1.25, cal*1.5]
            return growth_rate_list
            #return (((endingvalue/initialvalue)**(1/years)) - 1)*100

        def get_average_growth_rate(endingvalue, initialvalue, years):
            average_growth_rate = ((endingvalue-initialvalue)/abs(initialvalue))/years
            average_growth_rate_list = [0, average_growth_rate * 0.5, average_growth_rate, average_growth_rate * 1.5, average_growth_rate * 2, average_growth_rate * -0.5]
            return average_growth_rate_list

        #def line_of_best_fit(x, y):
        #    x = np.array(x)
        #    y = np.array(y)
        #    n = len(x)
        #    m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2) * 3
        #    b = (np.sum(y) - m * np.sum(x)) / n
        #    y_output = []
        #    x_output = []
        #    for i in x:
        #        y_output.append((i*m) + b)
        #        x_output.append(i)
        #    return x_output, y_output, m

        def line_of_best_fit(x1, y1):
            x = np.array(x1)
            y = np.array(y1)

            # Calculate the line of best fit
            m, b = np.polyfit(x, y, 1)  # 1 = degree of the polynomial (linear)

            # Generate y values for the fitted line
            #y_fit = m * x + b
            y_output = []
            x_output = []
            for i in x:
                y_output.append((i*m) + b)
                x_output.append(i)
            return x_output, y_output, m

        gs = line_of_best_fit(updated_cashflows_and_dates[0], updated_cashflows_and_dates[1])
        print(updated_cashflows_and_dates[0], updated_cashflows_and_dates[1])
        print(gs)

        def calculate_present_value_and_future_fcf(fcf, k, g, tg, yearstoterminal, yearstopeak):
            future_fcf_list = []
            m = (tg-g)/(yearstoterminal-yearstopeak)
            b = g - (m * yearstopeak)
            present_value = 0.0

            # Calculate the present value of each FCF
            for t in range(1, yearstopeak+1):
                future_fcf_list.append(fcf*((g+1)**t))
                present_value += (fcf*((g+1)**t))/((1 + k) ** t)

            lastcashflowbeforepeak = future_fcf_list[-1]

            for t in range(yearstopeak + 1, ((yearstoterminal)+1)):
                gi = m*(t) + b
                present_value += ((fcf*((g+1)**yearstopeak))*(gi+1)**(t-yearstopeak))/((1 + k) ** t)
                future_fcf_list.append((fcf*((g+1)**yearstopeak))*(gi+1)**(t-yearstopeak))

            lastcashflowbeforeterminal = future_fcf_list[-1]   
            terminal_value = (lastcashflowbeforeterminal * (1 + tg)) / ((k - tg) * ((1 + k) ** yearstoterminal))
            future_fcf_list.append(lastcashflowbeforeterminal * (1 + tg))
            present_value += terminal_value
            return present_value, future_fcf_list

        def calculate_present_value_and_future_fcf_linear(fcf, k, g, tg, yearstoterminal, yearstopeak, m, average_growth_rate):
            future_fcf_list = []
            m1 = (tg-g)/(yearstoterminal-yearstopeak)
            b1 = g - (m1 * yearstopeak)
            present_value = 0.0
            g = (g/average_growth_rate)
            tg1 = (tg/average_growth_rate)
            # Calculate the present value of each FCF
            for t in range(1, yearstopeak+1):
                future_fcf_list.append((fcf+(m*(g)*t)))
                present_value += ((fcf+(m*(g)*t)))/((1 + k) ** t)
            lastcashflowbeforepeak = future_fcf_list[-1]

            for t in range(yearstopeak + 1, ((yearstoterminal)+1)):
                gi = ((tg+g)/2)
                present_value += ((fcf+(m*(g)*yearstopeak))+(m*(gi)*(t-yearstopeak)))/((1 + k) ** t)
                future_fcf_list.append((fcf+(m*(g)*yearstopeak))+(m*(gi)*(t-yearstopeak)))

            lastcashflowbeforeterminal = future_fcf_list[-1] 
            terminal_value = (lastcashflowbeforeterminal+(m*(tg1))) / ((k - tg) * ((1 + k) ** (yearstoterminal)))
            future_fcf_list.append((lastcashflowbeforeterminal+(m*(tg))))
            present_value += terminal_value
            return present_value, future_fcf_list

        growthrate = geometric_average_percantage_return(gs[1][-1], abs(gs[1][0]), (gs[0][-1]-gs[0][0])+1)
        print(f"{growthrate[2]*100:.2f} %")

        average_growth_rate = get_average_growth_rate(gs[1][-1], gs[1][0], (gs[0][-1]-gs[0][0])+1)
        print(f"{average_growth_rate[2]*100:.2f} %")

        #average_growth_rate = get_average_growth_rate(gs[1][-1], gs[1][0], 1)
        #print(f"{average_growth_rate:.2f} %")

        #print(updated_cashflows_and_dates[1], updated_cashflows_and_dates[1][-1], (updated_cashflows_and_dates[0][-1]-updated_cashflows_and_dates[0][0])+1)

        #growthrate = geometric_average_percantage_return(updated_cashflows_and_dates[1][0], updated_cashflows_and_dates[1][-1], (updated_cashflows_and_dates[0][0]-updated_cashflows_and_dates[0][-1])+1)
        #print(f"{growthrate:.2f} %")

        def get_stock_value_and_update_future_fcf(last_freecashflow, wacc, growthrate, terminalgrowthrate, yearstoterminal, yearstopeak, ticker):
            V0 = calculate_present_value_and_future_fcf(last_freecashflow, wacc, growthrate, terminalgrowthrate, yearstoterminal, yearstopeak)[0]
            P0 = (V0 + get_total_cash(ticker) - get_total_debt(ticker))/get_shares_issued(ticker)
            updated_future_fcf = updated_cashflows_and_dates[1] + calculate_present_value_and_future_fcf(last_freecashflow, wacc, growthrate, terminalgrowthrate, yearstoterminal, yearstopeak)[1]
            new_dates = []
            for i in range(1, yearstoterminal+2):
                new_dates.append(updated_cashflows_and_dates[0][-1] + i)
            updated_dates = updated_cashflows_and_dates[0] + new_dates
            return (P0, updated_future_fcf, updated_dates)

        def get_stock_value_and_update_future_fcf_linear(last_freecashflow, wacc, growthrate, terminalgrowthrate, yearstoterminal, yearstopeak, ticker, m, average_growth_rate):
            V0 = calculate_present_value_and_future_fcf_linear(last_freecashflow, wacc, growthrate, terminalgrowthrate, yearstoterminal, yearstopeak, m, average_growth_rate)[0]
            P0 = (V0 + get_total_cash(ticker) - get_total_debt(ticker))/get_shares_issued(ticker)
            updated_future_fcf = updated_cashflows_and_dates[1] + calculate_present_value_and_future_fcf_linear(last_freecashflow, wacc, growthrate, terminalgrowthrate, yearstoterminal, yearstopeak, m, average_growth_rate)[1]
            new_dates = []
            for i in range(1, yearstoterminal+2):
                new_dates.append(updated_cashflows_and_dates[0][-1] + i)
            updated_dates = updated_cashflows_and_dates[0] + new_dates
            return (P0, updated_future_fcf, updated_dates)

        print("")
        r03 = get_stock_value_and_update_future_fcf(gs[1][-1], wacc, (growthrate[0]), 0.0, 20, 10, ticker)
        print(f"Price ${r03[0]:.2f} Growth Rate: {growthrate[0]*100:.2f}%")
        price_text_11 = f"Price ${r03[0]:.2f} Growth Rate: {growthrate[0]*100:.2f}%"
        r02 = get_stock_value_and_update_future_fcf(gs[1][-1], wacc, (growthrate[1]), 0.0, 20, 10, ticker)
        print(f"Price ${r02[0]:.2f} Growth Rate: {growthrate[1]*100:.2f}%")
        price_text_12 = f"Price ${r02[0]:.2f} Growth Rate: {growthrate[1]*100:.2f}%"
        r01 = get_stock_value_and_update_future_fcf(gs[1][-1], wacc, (growthrate[2]), 0.0, 20, 10, ticker)
        print(f"Price ${r01[0]:.2f} Growth Rate: {growthrate[2]*100:.2f}%")
        price_text_13 = f"Price ${r01[0]:.2f} Growth Rate: {growthrate[2]*100:.2f}%"
        r0 = get_stock_value_and_update_future_fcf(gs[1][-1], wacc, (growthrate[3]), 0.0, 20, 10, ticker)
        print(f"Price ${r0[0]:.2f} Growth Rate: {growthrate[3]*100:.2f}%")
        price_text_14 = f"Price ${r0[0]:.2f} Growth Rate: {growthrate[3]*100:.2f}%"
        r04 = get_stock_value_and_update_future_fcf(gs[1][-1], wacc, (growthrate[4]), 0.0, 20, 10, ticker)
        print(f"Price ${r04[0]:.2f} Growth Rate: {growthrate[4]*100:.2f}%")
        price_text_15 = f"Price ${r04[0]:.2f} Growth Rate: {growthrate[4]*100:.2f}%"
        print("")
        l05 = get_stock_value_and_update_future_fcf_linear(gs[1][-1], wacc, (average_growth_rate[5]), 0.0, 20, 10, ticker, gs[2], average_growth_rate[2])
        print(f"Price ${l05[0]:.2f} Growth Rate: {(average_growth_rate[5]*100):.2f}%")
        price_text_21 = f"Price ${l05[0]:.2f} Growth Rate: {(average_growth_rate[5]*100):.2f}%"
        l03 = get_stock_value_and_update_future_fcf_linear(gs[1][-1], wacc, (average_growth_rate[0]), 0.0, 20, 10, ticker, gs[2], average_growth_rate[2])
        print(f"Price ${l03[0]:.2f} Growth Rate: {(average_growth_rate[0]*100):.2f}%")
        price_text_22 = f"Price ${l03[0]:.2f} Growth Rate: {(average_growth_rate[0]*100):.2f}%"
        l02 = get_stock_value_and_update_future_fcf_linear(gs[1][-1], wacc, (average_growth_rate[1]), 0.0, 20, 10, ticker, gs[2], average_growth_rate[2])
        print(f"Price ${l02[0]:.2f} Growth Rate: {(average_growth_rate[1]*100):.2f}%")
        price_text_23 = f"Price ${l02[0]:.2f} Growth Rate: {(average_growth_rate[1]*100):.2f}%"
        l01 = get_stock_value_and_update_future_fcf_linear(gs[1][-1], wacc, (average_growth_rate[2]), 0.0, 20, 10, ticker, gs[2], average_growth_rate[2])
        print(f"Price ${l01[0]:.2f} Growth Rate: {(average_growth_rate[2]*100):.2f}%")
        price_text_24 = f"Price ${l01[0]:.2f} Growth Rate: {(average_growth_rate[2]*100):.2f}%"
        l0 = get_stock_value_and_update_future_fcf_linear(gs[1][-1], wacc, (average_growth_rate[3]), 0.0, 20, 10, ticker, gs[2], average_growth_rate[2])
        print(f"Price ${l0[0]:.2f} Growth Rate: {(average_growth_rate[3]*100):.2f}%")
        price_text_25 = f"Price ${l0[0]:.2f} Growth Rate: {(average_growth_rate[3]*100):.2f}%"
        l04 = get_stock_value_and_update_future_fcf_linear(gs[1][-1], wacc, (average_growth_rate[4]), 0.0, 20, 10, ticker, gs[2], average_growth_rate[2])
        print(f"Price ${l04[0]:.2f} Growth Rate: {(average_growth_rate[4]*100):.2f}%")
        price_text_26 = f"Price ${l04[0]:.2f} Growth Rate: {(average_growth_rate[4]*100):.2f}%"

        t03 = r03[1]
        t02 = r02[1]
        t01 = r01[1]
        t0 = r0[1]
        t04 = r04[1]

        y04 = l04[1]
        y03 = l03[1]
        y02 = l02[1]
        y01 = l01[1]
        y0 = l0[1]

        x2 = l03[2]
        nonlinearmodel = add_chart([x2, x2, x2, x2, x2, gs[0], updated_cashflows_and_dates[0]], [t04, t03, t02, t01, t0, gs[1], updated_cashflows_and_dates[1]], ['purple', 'blue', 'red', 'black', 'yellow', 'green', 'orange'], [f'Growth Rate  {(growthrate[4]*100):.2f}%', f'Growth Rate  {(growthrate[0]*100):.2f}%', f'Growth Rate  {(growthrate[1]*100):.2f}%', f'Growth Rate  {(growthrate[2]*100):.2f}%', f'Growth Rate  {(growthrate[3]*100):.2f}%', 'Line of Best Fit', 'Actualized FCF'], 'Projected Annual Free Cash Flow')
        linearmodel = add_chart([x2, x2, x2, x2, x2, x2, gs[0], updated_cashflows_and_dates[0]], [l05[1], y04, y03, y02, y01, y0, gs[1], updated_cashflows_and_dates[1]], ['pink','purple', 'blue', 'red', 'black', 'yellow', 'green', 'orange'], [f'Growth Rate: {(average_growth_rate[5]*100):.2f}%', f'Growth Rate: {(average_growth_rate[4]*100):.2f}%', f'Growth Rate: {(average_growth_rate[0]*100):.2f}%', f'Growth Rate {(average_growth_rate[1]*100):.2f}%', f'Growth Rate {(average_growth_rate[2]*100):.2f}%', f'Growth Rate {(average_growth_rate[3]*100):.2f}%', 'Line of Best Fit', 'Actualized FCF'], 'Projected Annual Free Cash Flow Linear Model')
    return render_template("index.html", processed_text=var, quarterly_chart_html=quarterly_chart_html, annual_chart_html=annual_chart_html, nonlinear_chart_html=nonlinearmodel, linear_chart_html=linearmodel, price_text_11=price_text_11, price_text_12=price_text_12, price_text_13=price_text_13, price_text_14=price_text_14, price_text_15=price_text_15, price_text_21=price_text_21, price_text_22=price_text_22, price_text_23=price_text_23, price_text_24=price_text_24, price_text_25=price_text_25, price_text_26=price_text_26)
