**Script**
**0:00-0:40 Title**
Hi everyone, my project is about forecasting hourly bike-sharing demand in Washington, DC.

The main question is: what temporal structure characterizes hourly bike-sharing demand, and how well can we forecast short-term demand using historical usage, calendar variables, and weather conditions?

In this project, I focus on the total number of bike rentals in each hour, and the goal is not only to get a good forecast, but also to understand what kind of time dependence is driving the forecasts.

**0:40-1:40 Motivation**
Let me first introduce the background and motivation behind this project. Bike-sharing is a suitable setting for time series analysis because demand changes a lot within a day. A station can be busy during commute hours, quiet overnight, and then busy again the next morning.

This matters operationally. Shared micromobility systems recorded about 157 million trips in the U.S. and Canada in 2023. At this scale, short-term allocation matters. For a station-based system like Capital Bikeshare, large hourly swings affect bike availability. If demand is underestimated, stations can run out of bikes. If it is overestimated, bikes may be placed where they are not needed.

So the forecasting problem is tied to rebalancing and short-term resource allocation.

**1:40-2:20 Roadmap**
In this project presentation, I will first describe the data and preprocessing. Then I will use exploratory analysis to identify the main temporal patterns. After that, I compare three families of models: classical time-series models, time-lagged regression models, and neural sequence models. For each family, I use a baseline method and an extension method. I will end with model comparison, limitations, and next steps.

**2:20-3:30 Data**
The data come from the Bike Sharing Dataset. In this project, I use `hour.csv`, the file at hourly resolution, which contains Capital Bikeshare rentals in Washington, DC from January 1, 2011 through December 31, 2012.

The response variable is `cnt`, the total number of bike rentals in each hour. The dataset also includes calendar variables, such as hour, weekday, holiday, working day, month, and season, plus weather variables including temperature, humidity, wind speed, and weather condition.

I do not use `casual` or `registered` as predictors, because they sum directly to `cnt`, so they are redundant.

The raw file has 17,379 observed rows. But the full hourly range contains 17,544 hours, so 165 timestamps are missing. So here, I rebuild the complete hourly index. I interpolate continuous variables, fill the weather category, and reconstruct calendar variables from the timestamp.

After this step, I have a complete hourly modeling frame with no missing values.

**3:30-4:30 Demand Scale**
Before fitting models, I first look at the response itself.

As we can see from the histogram, the demand distribution is strongly right-skewed. Many hours have low or moderate rentals, while peak periods can be much higher. The maximum hourly count is close to 1,000, while the median is much lower.

The moving average plot also shows slow movement across the two years. Demand changes by season and by year, not just hour to hour.

This motivates two choices later: first, I use log transformation in the regression and neural models to stabilize the scale; second, I include calendar information because demand is clearly not constant through time.

**4:30-5:50 Calendar Patterns**
After introducing the response variable, let's move on to the predictor variables, the calendar patterns and weather patterns.

The calendar plots show the clearest structure in the project.

Monthly demand is higher in warmer months. The weekday-hour heatmap shows repeated daily structure. And the working-day versus non-working-day plot shows a very different shape.

On working days, there are clear commute peaks, especially in the morning and late afternoon. On non-working days, the peak shifts toward the middle of the day.

This is important because it tells us that the model should not only know what happened recently. It should also know what hour it is, whether it is a working day, and what happened at the same hour yesterday or last week.

That is why later models use 24-hour and 168-hour history.

**5:50-6:40 Weather Patterns**
Weather also matters, although it is not the whole story.

Four conclusions can be drawn from the IQR plots. First, demand tends to rise with temperature over much of the observed range. Second, high humidity is associated with lower demand. Third, wind speed has a weaker and noisier relationship. And finally, worse weather conditions, especially rain or snow, reduce the median demand.

So weather variables are useful predictors, especially for regression and SARIMAX.

**6:40-7:25 Mean-Variance and Log Scale**
After some basic EDA, let's explore some detailed issues before modeling. Here I check the mean-variance relationship.

On the raw count scale, periods with larger average demand also tend to have larger variance. That means the noise level grows when demand is high. This phenomenon also suggests using a log transformation to stabilize variance.

After applying the log transformation, this relationship becomes weaker. So for the lagged regression, RNN, and LSTM, I train on log demand, then convert predictions back to counts before diagnostics and further analysis.

**7:25-8:55 Autocorrelation and Frequency**
Another detailed issue is the time dependence. The autocorrelation results are the main bridge from EDA to modeling.

On the log scale, the autocorrelation is about 0.90 at lag 1, about 0.87 at lag 24, and about 0.91 at lag 168. So demand is strongly related to the previous hour, the same hour yesterday, and the same hour last week.

First differencing reduces short-run persistence, but it does not remove the daily pattern.

Another differencing strategy is the 24-hour difference. As we can see from the ACF plot, it removes more of the daily cycle, which makes it a better choice for the ARIMA benchmark.

The frequency-domain plot says the same thing in another way. The daily frequency is the dominant peak in the original log demand series. After 24-hour differencing, that daily peak is reduced much more clearly.

To summarize, the EDA part does have some implications for modeling choice. For example, in the later analysis, ARIMA uses 24-hour differencing; regression uses lags 1, 2, 3, 24, and 168; and the neural models should test history windows of 24, 72, and 168 hours.

**8:55-9:40 Modeling Setup**
Then let's move on to the Modeling part. For the modeling setup, I use a time-based split instead of a random split, because future demand should not be used to predict the past.

The split is 60 percent training, 20 percent validation, and 20 percent test. I also leave a 168-hour gap, or one week, before the validation and test periods. This is mainly to reduce leakage from the lagged features, especially the 168-hour lag.

Validation is used for model selection, such as ARIMA order and sequence window length. The test set is used for the final comparison.

All models use the same data split configuration and random seed. And they are evaluated on the original count scale using MAE and RMSE.

**9:40-11:25 ARIMA**
The first model is ARIMA, which I use as the classical benchmark.

This model only uses past demand. Based on the earlier ACF and PACF analysis, I apply 24-hour differencing first, so the model is fitted to the differenced data. The choice of candidate order comes from the previous EDA. As we can see, the ACF is still visible at the first few lags, so I include MA terms up to order 2. Also, the PACF remains noticeable through about lag 3, so I include AR terms up to order 3.

The best validation RMSE comes from ARIMA(2,0,2), although several nearby orders perform similarly. On the validation set, ARIMA has an RMSE around 83.5, and on the test set, around 84.2.

For this first model, I want to explain the diagnostics in more detail, because I reuse the same diagnostic structure later.

The residuals-over-time plot checks whether the errors have drift, clusters, or long periods of bias. The residuals-versus-fitted plot checks whether the model misses high-demand periods or has changing error spread. The QQ plot checks whether the residuals have heavy tails or strong non-normality. And the residual ACF checks whether there is still time dependence left in the errors.

For ARIMA, the average bias is small, so the model is not simply overpredicting or underpredicting on average. But the residual ACF still shows weekly dependence. In particular, the residual autocorrelation at lag 168 is about 0.44.

That means 24-hour differencing helps with the daily cycle, but the model still misses part of the weekly structure and covariate effects.

**11:25-12:10 SARIMAX**
To address part of this issue, I use SARIMAX as an extension. It extends ARIMA by adding observed predictors.

Here, I include weather variables, working day, holiday, year, and cyclic encodings for hour, weekday, and month. The selected model uses a seasonal structure with period 24.

As we can see from the results, SARIMAX improves slightly over ARIMA. The validation RMSE drops from about 83.5 to 80.9, and the test RMSE is about 81.3.

The diagnostics tell a similar story. Bias remains close to zero, and the daily residual autocorrelation is smaller than in ARIMA. But weekly residual dependence is still large. So adding covariates helps, but SARIMAX is still not capturing the repeated weekly pattern well enough to be the main model.

**12:10-13:20 Lagged Regression**
To capture the calendar pattern more directly, I use time-lagged regression as the next model.

This model is more explicit about the patterns found in EDA. It uses log demand as the target, with lagged log demand at 1, 2, 3, 24, and 168 hours. It also includes calendar and weather variables.

This gives a large improvement over ARIMA and SARIMAX. The validation RMSE is about 65.7, and the test RMSE is about 59.4.

Moreover, since time-lagged regression is an interpretable model, the coefficient results are also useful. The results mostly line up with the EDA. Recent demand, daily lag, and weekly lag are positive and significant. Temperature has a positive effect, while humidity and wind speed are negative. Bad weather conditions also reduce demand.

The diagnostics improve compared with ARIMA, but they are not clean. Residual autocorrelation remains at short, daily, and weekly lags, and the residual spread is still larger at some fitted values. So the linear form is interpretable and useful, but not flexible enough to capture all peak-hour behavior.

**13:20-14:05 Poisson Lagged Regression**
As an extension, since the response is a count, I also fit a Poisson lagged regression.

It uses the same basic predictor set as the lagged regression, but models the count directly with a log link and Poisson likelihood. This gives a small RMSE improvement over ordinary lagged regression: validation RMSE is about 62.1, and test RMSE is about 57.8.

The diagnostic result is mixed. The forecast accuracy improves a bit, but the dispersion statistic is about 7.14, far above 1. Since the Poisson model assumes the conditional mean and variance are equal, this assumption is too tight here.

So as a conclusion, the Poisson model is useful as a count-model extension, but the overdispersion points toward a negative binomial model as a better future direction.

**14:05-15:00 RNN**
To increase the flexibility of the model, I use the neural sequence models as the last model family. The RNN is the first neural sequence model.

Instead of using only fixed lag columns, it reads a rolling history window and combines that history with target-time calendar and weather features. I test windows of 24, 72, and 168 hours. The model architecture and training configuration are listed on the slide.

The best RNN uses a 72-hour window. That window covers several daily cycles without being as long as a full week. The validation RMSE is about 55.4, and the test RMSE is about 55.2, so it improves over both regression models.

The training curve is stable. The residual diagnostics still show autocorrelation, especially around recurring calendar patterns, but the error level is lower than in the lagged regression models.

So the RNN benefits from reading a sequence rather than a small fixed set of lags.

**15:00-16:00 LSTM**
The final model is the LSTM.

It uses the same general setup as the RNN, but replaces the simple recurrent layer with gated LSTM layers. The idea is that the model can learn what information from the recent history to keep or forget. The model architecture and training configuration are also listed on the slide.

Again, the selected validation window is 72 hours. The LSTM has the best validation result among the main models, with validation RMSE about 43.5. On the test set, the RMSE is about 43.9.

The diagnostics are also the strongest among the models. The residual spread is smaller, and the residual autocorrelation at lag 24 and lag 168 is lower than in the earlier models. It does not disappear completely, but it is reduced.

This suggests that the gated sequence model is better at capturing the nonlinear hourly pattern, especially the repeated daily behavior and peak periods.

**16:00-17:05 Model Comparison**

Finally, we can make a comparison between the models mentioned before.

Putting the models side by side, the ranking is clear.

ARIMA and SARIMAX are useful baselines, but their test RMSE stays around 81 to 84. The lagged regression models improve a lot, with test RMSE around 58 to 59. The RNN improves further, with test RMSE around 55. And the LSTM performs best, with test RMSE around 44.

So the main result is that models do better when they directly represent temporal history. Weather and calendar variables help, but the biggest gain comes from using recent, daily, and weekly demand patterns.

As a conclusion, the LSTM is the strongest forecaster, while the lagged regression remains useful because it is easier to interpret.

**17:05-17:55 Forecast Plot**
To better understand where the errors happen. I also provide static and interactive forecast plots. You can select the model and the time period you want to visualize.

As we can see from the plot, all models capture the broad daily cycle. They know demand is low overnight and higher during active hours. The harder part is peak timing and peak magnitude.

The classical models are smoother and miss more of the sharp changes. The regression models track the pattern better because they use explicit lags. The RNN and LSTM are closer to the actual series, especially around repeated daily peaks.

**17:55-19:00 Conclusion**

As the very last step, we can draw some conclusions.

To answer the research question: hourly bike-sharing demand has strong temporal structure. It is shaped by short-run persistence, daily cycles, weekly repetition, working-day differences, seasonality, and weather.

For forecasting, the best model in this project is the LSTM with a 72-hour window. But the simpler models are still informative. ARIMA shows that daily differencing matters. Lagged regression shows that lag 1, lag 24, and lag 168 are important. Poisson regression shows that count modeling is reasonable, but the data are overdispersed.

The main limitation is that residual autocorrelation remains, even for the best model. That means some repeated structure is still not fully captured. Also, the project uses aggregate system-level demand, not station-level demand, so it does not model spatial imbalance.

With more time, I would try a negative binomial model, probabilistic forecasts instead of point forecasts, and station-level models that account for location and rebalancing needs.

That's all for my presentation. Thank you so much.