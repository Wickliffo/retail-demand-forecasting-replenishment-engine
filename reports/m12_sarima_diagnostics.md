# M12: SARIMA Residual Diagnostics

Model: SARIMA_111_011_7

Order: SARIMA(1,1,1)(0,1,1,7)

Missing calendar sales observations were treated as zero.

This is an experimental assumption requiring business verification.

## Overall out-of-fold metrics

           model  observations        MAE       RMSE  mean_error  median_error  std_error
SARIMA_111_011_7           150 318.607551 521.100558     56.2226     43.566389 519.794239

## Residual statistics

 fold residual_type  count       mean     median        std      variance          min         max  mean_absolute_residual  skewness  kurtosis_excess
    1     in_sample   1508   9.251390 -14.100362 475.088421 225709.007586 -3086.210244 5816.815732              280.438719  0.788533        21.743734
    1    validation     30 -54.531129 -24.475379 658.562677 433704.799232 -1999.017837 2058.100170              369.961889  0.706528         5.863372
    2     in_sample   1538   9.534956 -11.997985 478.345066 228814.001742 -3084.741531 5820.573489              281.693469  0.771248        21.050393
    2    validation     30   4.163095 -38.705658 550.648485 303213.753826 -1061.416846 2235.766998              314.323487  2.123270         9.247680
    3     in_sample   1568   9.562270  -9.995866 479.625196 230040.328460 -3079.806530 5828.752469              282.294680  0.696717        20.809958
    3    validation     30 201.618649 210.664515 500.788240 250788.861612 -1216.465698 1944.335088              367.241505  0.552695         6.099634
    4     in_sample   1598   7.236257 -13.688942 481.490792 231833.382607 -3074.072605 5840.385868              282.721117  0.724218        20.505138
    4    validation     30  56.858945  14.484241 464.595965 215849.410662 -1151.275700 1713.174132              289.846026  1.091339         5.867216
    5     in_sample   1628   6.754089 -13.185209 480.938661 231301.996057 -3070.045313 5846.618296              282.607314  0.716994        20.359794
    5    validation     30  73.003441 142.399064 383.388514 146986.752730 -1599.905932  591.339848              251.664850 -3.087277        12.607425

## Ljung-Box results

 lag  ljung_box_statistic  p_value  fold residual_type
   7            12.628153 0.081704     1     in_sample
  14            17.766979 0.217601     1     in_sample
  21            26.632473 0.183352     1     in_sample
  28            31.667567 0.288249     1     in_sample
   7            10.027760 0.187006     2     in_sample
  14            15.297220 0.358144     2     in_sample
  21            23.581176 0.313791     2     in_sample
  28            29.349573 0.394961     2     in_sample
   7             8.816988 0.266064     3     in_sample
  14            13.801769 0.464582     3     in_sample
  21            22.890828 0.349827     3     in_sample
  28            29.289515 0.397955     3     in_sample
   7             9.026379 0.250768     4     in_sample
  14            13.406934 0.494761     4     in_sample
  21            24.430662 0.272663     4     in_sample
  28            31.722524 0.285949     4     in_sample
   7             9.560373 0.214885     5     in_sample
  14            13.770677 0.466932     5     in_sample
  21            24.930247 0.250198     5     in_sample
  28            34.053827 0.199107     5     in_sample

## Model fitting information

 fold            model  training_observations training_start training_end validation_start validation_end  converged  fit_seconds  warning_count warnings
    1 SARIMA_111_011_7                   1508     2013-01-01   2017-02-16       2017-02-17     2017-03-18       True     2.856250              0
    2 SARIMA_111_011_7                   1538     2013-01-01   2017-03-18       2017-03-19     2017-04-17       True     3.698470              0
    3 SARIMA_111_011_7                   1568     2013-01-01   2017-04-17       2017-04-18     2017-05-17       True     4.739807              0
    4 SARIMA_111_011_7                   1598     2013-01-01   2017-05-17       2017-05-18     2017-06-16       True     4.679729              0
    5 SARIMA_111_011_7                   1628     2013-01-01   2017-06-16       2017-06-17     2017-07-16       True     4.605485              0

## Interpretation guidance

- Ljung-Box p-values below 0.05 provide evidence against the null of no autocorrelation up to the tested lag.
- Ljung-Box p-values above 0.05 do not prove that residuals are independent.
- Residual normality is not a strict requirement for accurate point forecasting.
- Residual variance patterns should be examined visually.
- Validation forecast errors are different from in-sample fitted residuals.
- Diagnostic results must be considered across folds, not from one fold alone.
- This model is not yet an inventory optimization model.
