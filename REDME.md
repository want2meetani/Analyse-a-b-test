
# Analyze-AB-test-Results

***Summary***

This is an A/B testing assignment completed for Udacity's Data Analyst Nano Degree Program. The project consisted of understanding the results of an A/B test run by an e-commerce website and helping the company understand through statistical conclusions, if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.

***Objectives***

For this project, I worked to understand the results of an A/B test run by an e-commerce website. The company has developed a new web page in order to try and increase the number of users. My goal was to help the company understand if they should implement this new page, keep the old page, or perhaps run the experiment longer to make their decision.

***Software Needed***

1)Python (Numpy, Pandas, Matplotlib, StatsModels, Scipy)
2)Jupyter Notebook

***Part I - Probability***

Statistics were computed to find out the probabilities of converting regardless of page. These were used to analyze if one page or the other led to more conversions.

***Part II - A/B Test***

Next, hypothesis testing was conducted assuming the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%.

The data was bootstrapped and sampling distributions were determined for both pages. Conclusions were drawn on conversions for both pages by calculating p-values.

***Data Wrangling:***

1)remove duplicates or records with missing or mismatched values
2)handle the rows where the landing_page and group columns don't align

***Data Analytics:***

1)Compute probabilities of converting:
      a)regardless of page.
    b)Given that an individual received the treatment
    c)Given that an individual received the control page
    
2)Perform Hypothesis Testing and calculate p-values

3)Conduct Logistic Regression

***Part III - Regression***

Logistic regression was then performed to confirm results of the previous steps. Null and alternative hypotheses associated with this regression model were stated and verified using statsmodel.

Next, along with testing if the conversion rate changes for different pages, I added an effect based on which country a user lives. Statistical output using logistic regression was provided to check if country had an impact on conversion.

***Conclusions***

1)There was no evidence suggesting that those who explore either page will neccessary lead to more conversions

2)The country of the user did not impact the rate of conversion between the two pages
