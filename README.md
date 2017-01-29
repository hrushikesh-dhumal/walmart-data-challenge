# walmart-data-challenge

## Introduction  
An attempt to solve- [Walmart data challenge](https://www.hackerrank.com/companies/walmart) using python notebooks.  
I have cleaned the data but the model building is still incomplete. 

## [Data Cleaning](/01walmart-Data Cleaning.ipynb) 
It involves the following steps-
* Substitute the missing value - By analyzing the data I was able to identify patterns in it and came up with rules to fill put the missing data-  
  1. If data is missing for both 'Item Class ID' and 'Product Long Description' i will assign it a tag value of 4483.  
  2. If the data is only missing for 'Product Long Description', I will assign it a tag value of 4537.  
  3. If the data is only missing for 'Item Class ID', I will assign it a tag value of 4537.  
  4. There is missing text data in 'Product Long Description' so I decided to subsitute the Nan values with text 'No data' so that my training algorithm will work onthose cases as well.
* Some of the features were very skewed.
* I saw html tags and symbols such as ¢, â which I do not want. So I decided to clean the data in 2 steps -  
  1. Remove the HTML TAGS and weird text such as ¢.  
  2. Remove the unnecesarry content such as-  
    a. Stop words  
    b. Punctuation marks
* New nulls introduced because of cleaning with beautiful soup, so the entire records were dropped. This happened because though there were Html tags, there was no product description. So I ***combined product description and name*** to ensure there is some text to descrive the product.  
* After removingthe Html tags NLP Pre processing steps involved - 
  1. Perform following NLP task on data-  
  2. Lower case  
  3. Remove punctuation  
  4. Remove numbers  
  5. Remove stop words  
  6. Stem the words using PorterStemmer




#Author
hrushikes.dhumal@gmail.com
