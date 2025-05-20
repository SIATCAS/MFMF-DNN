# MFMF-DNN
multi-view feature map fusion-based deep neural network
Instructions for Use:

1. The data format for each subject is N¡Á3, where N represents the number of CGM (Continuous Glucose Monitoring) data points. CGM readings are collected every 5 minutes. For example, if a subject provides 3 days of data, then N = 288 ¡Á 3 = 864.

2. The first column contains the subject ID, which is not used in the code.

3. The second column contains the CGM values, measured in mmol/L.

4. The third column represents the time of day for each reading. For ease of analysis, the time has been preprocessed as follows: 00:00 is marked as 0, 00:05 as 5, and so on, with 23:59 represented as 1439.

5. For other datasets, as long as the data is formatted according to the above structure, this code can be applied directly without modification.
