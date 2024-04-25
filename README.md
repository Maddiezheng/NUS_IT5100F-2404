<img width="760" alt="截屏2024-04-25 下午8 12 16" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/ca678ed9-35c4-41c9-ab4f-d35579f65a54"># IT5100F Project: Predicting HDB Resale Price

## Project Group 3

## Content
- [Task 1: Problem Definition](#task-1-problem-definition)
- [Task 2: Data Collection/Curation](#task-2-data-collectioncuration)
- [Task 3: Data Preparation](#task-3-data-preparation)
- [Task 4: Exploratory Data Analysis and Visualization](#task-4-exploratory-data-analysis-and-visualization)
- [Task 5: Modeling and Visualization](#task-5-modeling-and-visualization)
- [Task 6: Report Insights and Conclusions](#task-6-report-insights-and-conclusions)

---

### Task 1: Problem Definition
#### Objective & Problem Statement
In Singapore, public housing, or HDB (Housing Development Board) flats, are a primary residential choice for many young individuals and couples due to their affordability. The decision-making process for potential buyers involves multiple factors, such as location preference, proximity to amenities like MRT stations and shopping centers, and the selection of flat types. To aid in this decision-making, it's essential to predict the resale value of HDB flats accurately. This value is influenced by various factors, including location, nearby amenities, environmental surroundings, future urban development, and government policies.

Our approach involves building a comprehensive data pipeline that uses historical transaction data and relevant property attributes. We will develop predictive models using three different techniques: Multiple Linear Regression, LightGBM, and Random Forest. Each model's performance will be evaluated using metrics such as Mean Squared Error (MSE) and R-squared (R²) to determine their effectiveness in predicting HDB flat resale prices.

This data pipeline is designed to serve multiple use cases, including reporting, analytics, and machine learning applications. It will facilitate the extraction of key insights from the data, enable the generation of reports for stakeholders, and support the development of predictive models for machine learning purposes.

By incorporating these models and evaluating their performance, we aim to provide young buyers with a clearer understanding of the key factors impacting HDB resale prices, thereby simplifying the purchasing process. This analysis will not only assist buyers in making more informed decisions but will also enhance our understanding of the housing market dynamics in Singapore.

#### Data Information
1. Utilizing the latest dataset of HDB Resale Prices released on April 12, 2024, by [DATA.GOV.SG](http://data.gov.sg/), which includes HDB resale transaction data from January 1990 to April 2024(https://beta.data.gov.sg/collections/189/datasets/d_ebc5ab87086db484f88045b47411ebc5/view):
   -  HDB Resale  Prices approved between January 1990 and December 1999 (287,196 records)
   -  HDB Resale  Prices approved between January 2000 and February 2012 (369,651 records)
   -  HDB Resale  Prices based on registration dates from March 2012 to December 2014 (52,203 records)
   -  HDB Resale  Prices based on registration dates from January 2015 to December 2016 (37,153 records)
   -  HDB Resale  Prices based on registration dates from January 2017 to April 2024 (176,976 records)

   A total of 923,179 records.

| Title                | Column Name          | Data Type                    | Unit of Measure |
|----------------------|----------------------|------------------------------|-----------------|
| Month                | month                | Datetime (Month) "YYYY-MM"   | -               |
| Town                 | town                 | Text (General)               | -               |
| Flat Type            | flat_type            | Text (General)               | -               |
| Block                | block                | Text (General)               | -               |
| Street Name          | street_name          | Text (General)               | -               |
| Storey Range         | storey_range         | Text (General)               | -               |
| Floor Area           | floor_area_sqm       | Numeric (General)            | Sqm             |
| Flat Model           | flat_model           | Text (General)               | -               |
| Lease Commence Date  | lease_commence_date  | Datetime (Year) "YYYY"       | -               |
| Resale Price         | resale_price         | Numeric (General)            | S$              |


2. Using the public dataset mrtsg.csv available on Github (https://github.com/hxchua/datadoubleconfirm/blob/master/datasets/mrtsg.csv):
   - This dataset includes the names and the latitude and longitude of all MRT stations in Singapore.
| Title       | Column Name | Data Type       | Unit of Measure |
|-------------|-------------|-----------------|-----------------|
| Object ID   | OBJECTID    | Numeric         | -               |
| Station Name| STN_NAME    | Text            | -               |
| Station No  | STN_NO      | Text            | -               |
| X Coordinate| X           | Numeric         | -               |
| Y Coordinate| Y           | Numeric         | -               |
| Latitude    | Latitude    | Numeric         | Degrees         |
| Longitude   | Longitude   | Numeric         | Degrees         |
| Color       | COLOR       | Categorical Text| -               |

3. Using the public dataset shopping_mall_coordinates.csv available on Kaggle (https://www.kaggle.com/datasets/karthikgangula/shopping-mall-coordinates):
   - These datasets include the names and the latitude and longitude of all shopping malls in Singapore.

| Title        | Column Name | Data Type       | Unit of Measure |
|--------------|-------------|-----------------|-----------------|
| Mall Name    | Mall Name   | Text            | -               |
| Latitude     | LATITUDE    | Numeric         | Degrees         |
| Longitude    | LONGITUDE   | Numeric         | Degrees         |


---

### Task 2: Data Collection/Curation
In this part, we loaded the datasets and merged them.

---

### Task 3: Data Preparation
In this part, in addition to using the HDB datasets originally downloaded on data.gov.sg, we also downloaded mrtsg.csv (https://github.com/hxchua/datadoubleconfirm/blob/master/datasets/mrtsg.csv) and shopping_mall_coordinates.csv (https://www.kaggle.com/datasets/karthikgangula/shopping-mall-coordinates) these two datasets. These two datasets include all MRT Stations and Shopping Malls in Singapore and their latitude and longitude respectively.

#### 1. Data Conversion:

Converted 'month' and 'lease_commence_date' columns to datetime data type for better time series analysis.
#### 2. Handling Missing Values:

Addressed missing values in the 'remaining_lease' column by recalculating it using the formula: remain_lease = lease_commence_date + 99 - resale_year. This ensures uniformity across datasets with differing formats for 'remaining_lease'.
#### 3. Categorical Data Transformation:

Standardized the case for 'flat_model' and 'flat_type' columns to ensure consistency.
Transformed 'storey_range' into a numerical average to refine the granularity of storey level data, enhancing accuracy in modeling.
#### 4. Geospatial Data Integration and Proximity Analysis:

Integrated OneMap APIs to fetch geospatial coordinates (latitude and longitude) for each HDB flat by merging 'Block' and 'Street Name'.
Computed proximity to the nearest MRT stations and shopping malls using the geospatial coordinates and calculated walking distances.
#### 5. Addressing Incomplete Geospatial Data:

For missing geospatial data, opted to remove records where addresses could not be located, given the dataset's large size and the method's complexity.
#### 6. Feature Engineering:

Developed features indicating the walking time to the nearest MRT station and shopping mall based on the calculated distances.
Calculated town and flat model premiums by comparing median resale prices against overall median prices, providing insights into the cost implications of different HDB locations and models.

---

### Task 4: Exploratory Data Analysis and Visualization
#### Overall Analytics
1. Median Resale Price Trend of HDB Flats (1990 - 2024)
It can be seen that the Median Resale Price of HDB Flats is fluctuating and rising. The HDB resale price has 2 peaks. The first peak occurred in 1997, followed by a sharp decline, mainly due to the Asian financial crisis. Since then, the resale price of flats has risen again since 2006 and reached a new high in 2013. Subsequently, the price of resale of flats fell again, which coincided with a series of cooling measures in the public housing market. From 2014 to 2019, from the median point of view, the overall resale price of HDB is quite stable. However, since 2019, the price of HDB resale has risen again, which may have been affected by Covid-19 and inflation.
<img width="930" alt="截屏2024-04-25 下午8 09 43" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/9823651e-e62d-4bdd-8896-4bf09bb3cbcf">

2. Trend of HDB Transactions (1990 - 2024)
From 1990 to 1996, the transaction volume of HDB resale increased year by year. In the range from 1997 to 1999, the Resale volume of HDB increased significantly, with more than 50,000 transactions in 1999. There was a sharp decline in 2010-2013. The sharp decline in 2023-2024 is not because the number of HDB resale transactions has really decreased, but because the dataset is only counted until March 2024.
<img width="932" alt="截屏2024-04-25 下午8 10 14" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/22790e27-f19e-4984-8c33-547f112329ba">


3. Median Resale Price Trend of HDB Flats by Town (1990 - 2024)
It is obvious that the HDB resale price of Central Area rose sharply in 2013, and the increase was greater than that of any town. For other central regions such as Bukit Timah, Bishan, Bukit Merah, Queenstown, the resale price is also relatively high.
<img width="738" alt="截屏2024-04-25 下午8 10 54" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/e0ab887f-2f50-4235-a83f-c0573d5bf2f5">

4. Trend of Town Premiums for HDB Flats (1990 - 2024)
In comparison to other towns, Bukit Timah, Bishan, Bukit Merah, Central Area, Kallang, Queenstown, Pasir Ris, Serangoon, Tampines, Punggol, and Sengkang exhibit a positive average town premium. This indicates that purchasing HDBs (Housing Development Board flats) in these areas requires a higher financial outlay compared to other regions. Among them, the town premiums in Bukit Timah, Bishan, and Bukit Merah are notably higher. Conversely, towns like Yishun, Jurong West, Jurong East, and Ang Mo Kio show a negative average town premium, suggesting that buying HDBs in these areas could be less costly compared to other regions.
<img width="736" alt="截屏2024-04-25 下午8 11 47" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/e4590e7b-8d7a-49a6-b71b-69765a593293">

5. Median Resale Price Trend of HDB Flats by Flat Model (1990 - 2024)
It can be seen that the resale prices of Type S1, Type S2, PREMIUM APARTMENT LOFT, MULTI-GENERATION are relatively high, while the resale prices of Simplified, New Generation, 2 Room, Standard are relatively low.
<img width="739" alt="截屏2024-04-25 下午8 12 31" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/7eb16b59-7486-42a2-af51-81ed6009aad1">

6. Trend of Flat Model Premiums for HDB Flats (1990 - 2024)
The HDB Model Premiums for Type S2, Type S1, Premium Apartment Loft, Multi-Generation, Model A Maisonette, DBSS (Design, Build and Sell Scheme), Maisonette, Adjoined Flat, Apartment, Terrace, 3 Gen, and Premium Apartment are relatively high. In contrast, the HDB Model Premiums for Simplified, New Generation, 2 Room, and Standard types generally show negative figures.
<img width="743" alt="截屏2024-04-25 下午8 13 00" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/c28cd3ea-5d75-4bac-bf89-e945513ea788">















#### Statistical Analysis
展示数据的统计分析，如平均值、标准偏差等。




---

### Task 5: Modeling and Visualization
#### Model Selection
介绍选择的模型和原因。

#### Model Building
描述模型建立过程。

#### Visualization of Results
展示模型结果的图表或图形。

---

### Task 6: Report Insights and Conclusions
#### Insights
分享从数据中获得的洞察。

#### Conclusions
基于分析得出的结论，包括对策和建议。

---

## Authors
介绍参与项目的团队成员。

## Acknowledgements
感谢对项目有贡献的个人或组织。

## License
说明项目的许可证信息。

