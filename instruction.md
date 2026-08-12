We need two datasets: a daily ASEAN market dataset for the connectedness model and a global-shock dataset for explaining changes in connectedness.

1. ASEAN equity-market data
Target period: January 1, 2010–July 31, 2026, subject to common availability.

Country	Index	Required currency
Indonesia	Jakarta Composite Index	IDR
Malaysia	FTSE Bursa Malaysia KLCI	MYR
Philippines	PSE Composite Index	PHP
Singapore	Straits Times Index	SGD
Thailand	SET Index	THB
Vietnam	VN-Index	VND
For every trading day and market, download:
date
country
index_name
ticker
open
high
low
close
adjusted_close
volume
currency
source

Only date, high, low, and close are essential. If high and low are unavailable, closing prices alone are sufficient for the fallback volatility measure.

Do not merge or forward-fill missing national trading days in the raw files.

2. Global-shock and control data
Essential variables
Variable	Frequency	Suggested series	Use
VIX	Daily	FRED VIXCLS	Global risk aversion
Geopolitical risk	Daily/monthly	Caldara–Iacoviello GPR	Geopolitical shocks
Brent crude oil	Daily	FRED DCOILBRENTEU	Commodity shock
US 2-year Treasury yield	Daily	FRED DGS2	Monetary-policy expectations
Broad US dollar index	Daily	FRED DTWEXBGS	Global dollar conditions
S&P 500	Daily	FRED SP500 or equivalent	External equity-market conditions
VIX is a daily measure of expected near-term equity volatility; Brent is reported daily in dollars per barrel; the US two-year yield and broad dollar index are also available daily. (fred.stlouisfed.org)

The conventional geopolitical-risk dataset provides monthly data and a recent daily series. A newer AI-GPR daily series currently covers data through July 31, 2026, but it should initially be treated as a robustness alternative because it uses a different measurement method. (matteoiacoviello.com)

Optional monthly variables
Global Economic Policy Uncertainty index
US Economic Policy Uncertainty index
US Monetary Policy Uncertainty index
Country-specific GPR indices
These should be used in a monthly model. We should not forward-fill monthly EPU observations and pretend they are independent daily observations. Official EPU data include global, US, monetary-policy, and country-level series. (policyuncertainty.com)

3. Exchange-rate data
For the currency robustness test, collect daily local-currency-per-US-dollar rates:
date
country
local_currency
local_currency_per_usd

Then:

P
i
,
t
U
S
D
=
P
i
,
t
L
C
U
F
X
i
,
t
,
P 
i,t
USD
​
 = 
FX 
i,t
​
 
P 
i,t
LCU
​
 
​
 ,
and

r
i
,
t
U
S
D
=
100
Δ
ln
⁡
(
P
i
,
t
U
S
D
)
=
r
i
,
t
L
C
U
−
100
Δ
ln
⁡
(
F
X
i
,
t
)
.
r 
i,t
USD
​
 =100Δln(P 
i,t
USD
​
 )=r 
i,t
LCU
​
 −100Δln(FX 
i,t
​
 ).
The baseline results will use local currencies; USD returns will test whether the conclusions are driven by exchange-rate movements.

4. Data-cleaning calculations
For every series:

Convert dates to a common format.
Sort chronologically.
Remove duplicate dates.
Convert price columns to numeric values.
Identify missing, zero, and negative prices.
Check that 
H
t
≥
L
t
>
0
H 
t
​
 ≥L 
t
​
 >0.
Calculate the number of observations per market.
Document the first and last valid date.
Retain the raw national trading calendars.
Create a synchronized common-date dataset separately.
We should compare at least two synchronization rules:

Intersection of trading dates across all six markets.
Weekly aggregation, which reduces nonsynchronous-trading problems.
5. Return calculations
Daily continuously compounded returns:

r
i
,
t
=
100
[
ln
⁡
(
P
i
,
t
)
−
ln
⁡
(
P
i
,
t
−
1
)
]
.
r 
i,t
​
 =100[ln(P 
i,t
​
 )−ln(P 
i,t−1
​
 )].
Calculate:

Local-currency returns
USD returns
S&P 500 returns
Brent oil returns
Broad-dollar-index changes
For price or index variables:

Δ
x
t
=
100
[
ln
⁡
(
x
t
)
−
ln
⁡
(
x
t
−
1
)
]
.
Δx 
t
​
 =100[ln(x 
t
​
 )−ln(x 
t−1
​
 )].
For interest rates:

Δ
y
t
=
100
(
y
t
−
y
t
−
1
)
,
Δy 
t
​
 =100(y 
t
​
 −y 
t−1
​
 ),
which expresses the change in basis points when 
y
t
y 
t
​
  is recorded in percentage points.

6. Volatility calculations
Preferred: Parkinson range volatility
When daily high and low prices are available:

v
i
,
t
P
=
v 
i,t
P
​
 =
\frac{1}{4\ln(2)}

\left[

\ln\left(\frac{H_{i,t}}{L_{i,t}}\right)

\right]^2.

For estimation, we may use:

x
i
,
t
=
ln
⁡
(
v
i
,
t
P
+
ϵ
)
,
x 
i,t
​
 =ln(v 
i,t
P
​
 +ϵ),
where 
ϵ
ϵ is a small constant preventing the logarithm of zero.

Fallback: squared returns
v
i
,
t
S
R
=
r
i
,
t
2
.
v 
i,t
SR
​
 =r 
i,t
2
​
 .
Additional robustness measure
v
i
,
t
A
R
=
∣
r
i
,
t
∣
.
v 
i,t
AR
​
 =∣r 
i,t
​
 ∣.
The final paper should report Parkinson volatility as the baseline if coverage is reliable and squared returns as the principal robustness test.

7. Descriptive calculations
For returns and volatility:

Number of observations
Mean
Median
Standard deviation
Minimum and maximum
Skewness
Excess kurtosis
Jarque–Bera normality test
Augmented Dickey–Fuller test
Phillips–Perron or KPSS test
Correlation matrices
Autocorrelation functions
Volatility and return plots
These produce Tables 3–4 and Figure 1 in the manuscript.

8. VAR calculations
Construct:

x
t
=
(
v
Indonesia
,
t
,
…
,
v
Vietnam
,
t
)
′
.
x 
t
​
 =(v 
Indonesia,t
​
 ,…,v 
Vietnam,t
​
 ) 
′
 .
Estimate:

x
t
=
∑
k
=
1
p
Φ
k
x
t
−
k
+
ε
t
.
x 
t
​
 = 
k=1
∑
p
​
 Φ 
k
​
 x 
t−k
​
 +ε 
t
​
 .
Required calculations:

Select lag order using AIC, BIC, and HQIC.
Check VAR stability.
Test residual autocorrelation.
Estimate the moving-average representation.
Calculate generalized forecast-error variance decompositions.
Normalize every variance-decomposition row.
Baseline starting specification:

VAR lag: selected by BIC, maximum 10
Forecast horizon: 10 trading days
Rolling window: 250 trading days

9. Connectedness calculations
From the normalized variance decomposition, calculate:

Total Connectedness Index
Directional connectedness received “FROM” others
Directional connectedness transmitted “TO” others
Net directional connectedness
Pairwise directional connectedness
Net pairwise connectedness
For each market:

N
e
t
i
=
T
O
i
−
F
R
O
M
i
.
Net 
i
​
 =TO 
i
​
 −FROM 
i
​
 .
Interpretation:

N
e
t
i
>
0
Net 
i
​
 >0: net volatility transmitter
N
e
t
i
<
0
Net 
i
​
 <0: net volatility receiver
Repeat these calculations over every 250-day rolling window.

10. Shock and contagion calculations
For each shock, prepare:
event_name
event_date
shock_start
shock_end
tranquil_start
tranquil_end
source_for_dates
Then calculate:

Mean TCI during the shock
Mean TCI during the tranquil benchmark
Difference in means
Median difference
Bootstrap confidence interval
HAC-standard-error regression
Changes in each market’s net position
A possible regression is:

T
C
I
t
=
α
+
β
1
G
P
R
t
+
β
2
V
I
X
t
+
β
3
Δ
O
i
l
t
+
β
4
Δ
D
G
S
2
t
+
β
5
Δ
D
o
l
l
a
r
t
+
ε
t
.
TCI 
t
​
 =α+β 
1
​
 GPR 
t
​
 +β 
2
​
 VIX 
t
​
 +β 
3
​
 ΔOil 
t
​
 +β 
4
​
 ΔDGS2 
t
​
 +β 
5
​
 ΔDollar 
t
​
 +ε 
t
​
 .
Evidence of contagion requires a significant increase in connectedness, not merely high volatility.

11. Robustness calculations
We will repeat the model using:

200-, 250-, and 300-day windows
5-, 10-, and 20-day forecast horizons
Alternative VAR lag orders
Parkinson volatility and squared returns
Local-currency and USD returns
Common trading days and weekly data
Alternative definitions of shock windows
Conventional GPR and AI-GPR
DCC-GARCH conditional correlations, if feasible
12. Files to provide
The cleanest handoff would be:

asean_indices_raw.csv
exchange_rates_raw.csv
global_daily_raw.csv
global_monthly_raw.csv
event_windows.csv

The ASEAN file is enough to begin the primary analysis. External variables and exchange rates can be added afterward.