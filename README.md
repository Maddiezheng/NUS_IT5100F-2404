# IT5100F Final Project: Predicting HDB Resale Price
<img width="1200" alt="截屏2024-04-25 下午8 54 01" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/881df37a-00a5-4d2f-a6e0-edc390d6afd5">
(The picture is of our team member, Fan's future home: SkyTerrace @Dawson)


## Course Information
Course Title: IT5100F - Industry Readiness: Data Analytics and AI in Practice

Professor：Akshay Narayan

Semester: 2023/2024 Semester 2

Credits: 4

## Project Group 3

## Content
- [Task 1: Problem Definition](#task-1-problem-definition)
  - [Objective & Problem Statement](#objective--problem-statement)
  - [Data Information](#data-information)
- [Task 2: Data Collection/Curation](#task-2-data-collectioncuration)
- [Task 3: Data Preparation](#task-3-data-preparation)
- [Task 4: Exploratory Data Analysis and Visualization](#task-4-exploratory-data-analysis-and-visualization)
  - [Overall Analytics](#overall-analytics)
  - [By Flat Type](#by-flat-type)
  - [By Town](#by-town)
  - [By Flat Model](#by-flat-model)
  - [By Storeys](#by-storeys)
  - [By Floor Area](#by-floor-area)
  - [By Lease Commence Date](#by-lease-commence-date)
  - [By Remaining Lease Year](#by-remaining-lease-year)
  - [By Distance to Nearest Amenities](#by-distance-to-nearest-amenities)
  - [By Travel Time](#by-travel-time)
  - [By Location](#by-location)
  - [HeatMap](#heatmap)
  - [Statistical Analysis](#statistical-analysis)
- [Task 5: Modeling and Visualization](#task-5-modeling-and-visualization)
  - [Multi Linear Regression](#multi-linear-regression)
  - [LightGBM](#lightgbm)
  - [Random Forest](#random-forest)
- [Task 6: Report Insights and Conclusions](#task-6-report-insights-and-conclusions)
---

## Task 1: Problem Definition
### Objective & Problem Statement
In Singapore, public housing, or HDB (Housing Development Board) flats, are a primary residential choice for many young individuals and couples due to their affordability. The decision-making process for potential buyers involves multiple factors, such as location preference, proximity to amenities like MRT stations and shopping centers, and the selection of flat types. To aid in this decision-making, it's essential to predict the resale value of HDB flats accurately. This value is influenced by various factors, including location, nearby amenities, environmental surroundings, future urban development, and government policies.

Our approach involves building a comprehensive data pipeline that uses historical transaction data and relevant property attributes. We will develop predictive models using three different techniques: Multiple Linear Regression, LightGBM, and Random Forest. Each model's performance will be evaluated using metrics such as Mean Squared Error (MSE) and R-squared (R²) to determine their effectiveness in predicting HDB flat resale prices.

This data pipeline is designed to serve multiple use cases, including reporting, analytics, and machine learning applications. It will facilitate the extraction of key insights from the data, enable the generation of reports for stakeholders, and support the development of predictive models for machine learning purposes.

By incorporating these models and evaluating their performance, we aim to provide young buyers with a clearer understanding of the key factors impacting HDB resale prices, thereby simplifying the purchasing process. This analysis will not only assist buyers in making more informed decisions but will also enhance our understanding of the housing market dynamics in Singapore.

### Data Information
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

## Task 2: Data Collection/Curation
In this part, we loaded the datasets and merged them.

---

## Task 3: Data Preparation
In this part, in addition to using the HDB datasets originally downloaded on data.gov.sg, we also downloaded mrtsg.csv (https://github.com/hxchua/datadoubleconfirm/blob/master/datasets/mrtsg.csv) and shopping_mall_coordinates.csv (https://www.kaggle.com/datasets/karthikgangula/shopping-mall-coordinates) these two datasets. These two datasets include all MRT Stations and Shopping Malls in Singapore and their latitude and longitude respectively.

### 1. Data Conversion:

Converted 'month' and 'lease_commence_date' columns to datetime data type for better time series analysis.
### 2. Handling Missing Values:

Addressed missing values in the 'remaining_lease' column by recalculating it using the formula: remain_lease = lease_commence_date + 99 - resale_year. This ensures uniformity across datasets with differing formats for 'remaining_lease'.
### 3. Categorical Data Transformation:

Standardized the case for 'flat_model' and 'flat_type' columns to ensure consistency.
Transformed 'storey_range' into a numerical average to refine the granularity of storey level data, enhancing accuracy in modeling.
### 4. Geospatial Data Integration and Proximity Analysis:

Integrated OneMap APIs to fetch geospatial coordinates (latitude and longitude) for each HDB flat by merging 'Block' and 'Street Name'.
Computed proximity to the nearest MRT stations and shopping malls using the geospatial coordinates and calculated walking distances.
### 5. Addressing Incomplete Geospatial Data:

For missing geospatial data, opted to remove records where addresses could not be located, given the dataset's large size and the method's complexity.
### 6. Feature Engineering:

Developed features indicating the walking time to the nearest MRT station and shopping mall based on the calculated distances.
Calculated town and flat model premiums by comparing median resale prices against overall median prices, providing insights into the cost implications of different HDB locations and models.

---

## Task 4: Exploratory Data Analysis and Visualization
### Overall Analytics
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


### By Flat Type
1. Number of HDB Flats by Flat Type (1990 - 2024)
4 ROOM has the largest number of resales. Between 1990 and 2024, there were as many as 350,000 resale transactions, followed by 3 ROOM and 5 ROOM. 1 The number of ROOM and Multi-Generation sales transactions is very limited.
<img width="743" alt="截屏2024-04-25 下午8 15 44" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/2d6a247e-4cd4-4607-9114-2dba6a9ba7be">


2. Yearly Counts of Flats by HDB Flat Type (1990 - 2024)
From 1990 to 1996, 3 ROOM had the largest resale transaction volume, but since 1997, 4 ROOM has gradually become the Flat Type with the largest resale transaction volume. 1998 and 1999 were the two years with the largest number of resale transactions.

PS: The number of resale transactions in 2024 is low because datasets only counted until March 2024, but it can still be seen that the current type of 4 ROOM in 2024 is still the mainstream of resale transactions.
<img width="521" alt="截屏2024-04-25 下午8 16 14" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/d64b8446-61d8-467a-af59-3e05fb8d5fa1">


3. Percentage of HDB Flats by Flat Type (1990 - 2024)
4 ROOM accounts for the highest proportion of all resale transactions, accounting for 38.4%, followed by 3 ROOM, accounting for 31.4%; the least is 1 ROOM and MULTI-GENERATION, only 0.1%
<img width="651" alt="截屏2024-04-25 下午8 16 38" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/85f55307-5c95-4747-988c-19a489a15a0f">


4. Sales Count by Flat Type (1990-2000), Sales Count by Flat Type (2001-2011), Sales Count by Flat Type (2012-2024)
In 2001-2011 and 2012-2024, 4 ROOM was the Flat Type with the highest resale transaction volume, but in 1990-2000, 3 ROOM was the Flat Type with the highest resale transaction volume. From the picture, it can be seen that 3 ROOM, 4 ROOM, 5 ROOM, EXECUTIVE are all mainstream resale Flat types.
<img width="724" alt="截屏2024-04-25 下午8 17 42" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/2190623f-b256-4684-9bb1-bfc919ffcec6">
<img width="728" alt="截屏2024-04-25 下午8 18 03" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/910fadfb-ed4e-4029-8412-ab7bd21048ff">
<img width="732" alt="截屏2024-04-25 下午8 18 16" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/1259172e-9684-405c-aef0-38aed7cb6955">


5. Resale Price Trend in 1990-2024 by All Flat Type
The trend of the median resale price of these 7 Flat Models is roughly the same, rising year by year from 1990 to 1995. Prices were relatively stable in 1996-2007. It began to rise slowly after 2007. Among them, the price of MULTI-GENERATION and EXECUTIVE is relatively high.
<img width="752" alt="截屏2024-04-25 下午8 18 38" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/cfe8943b-4103-41d6-8bc8-45f3a3f847f3">


### By Town
1. HDB Flat Type Resale Count by Town (1990-2024)
Tampines, Bedok, Jurong West, Yishun's HDB resale volume is relatively large, and Lim Chu Kang, Bukit Timah, Central Area's HDB resale volume is relatively small. . Ang Mo Kio and Bedok, clementi, Geylang, Kallang, Toa Payoh mainly concentrated in the transactions of 3 ROOM. The rest of the town mainly focuses on the transactions of 4 ROOM.
<img width="731" alt="截屏2024-04-25 下午8 19 12" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/e0c1ffde-8efb-4e92-9cfa-dc99bab383cd">


2. Median Resale Price Trend of 4-Room HDB Flats by Town (1990 - 2024)
Since 2014, the resale of 4 ROOM has surged, and the transaction volume has begun to decline again in 2018. It was not until 2020 that the resale of 4 ROOM began to surge again. The three regions of Queestown, Bukit Merah and Central Area have always been relatively high in the resale of 4 ROOM.
<img width="732" alt="截屏2024-04-25 下午8 19 33" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/3b1d421d-5127-46cc-8b3d-4b59976ed1cd">


3. Town Premium Trend By Bukit Merah, Central Area and Queenstown(1990 - 2024)
Now from an analysis of the three regions mentioned above, it can be seen that in 1990-2005, the town premium of BUKIT MERAH and CENTRAL AREA was negative, and the town premium began to rise year by year in 2005. Queensland didn't start to be positive until 2014. CENTRAL AREA surged in town premium in 2015, and has returned to normal since then.
<img width="742" alt="截屏2024-04-25 下午8 19 57" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/2b1ac8d4-b45c-415e-afec-0416e41fdd38">


4. Median Price of HDB Flats from 1990 - 2000 & 2001-2011 & 2012-2024
In 1990-2000, the middle price of HDB flats of Pasir Ris was the highest, and Lim Chu Kang was the lowest. For the most common 4 ROOM, the price of Bishan is the highest, and Lim Chu Kang is still the lowest.

In 2001-2011, Punggol's meadin price of HDB flats was the highest, and Yishun was the lowest. For the most common 4 ROOM, Queenstown has the highest price, and Yishun is still the lowest.

In 2012-2024, Bukit Timah's meadin price of HDB flats is the highest, up to S$710,000. Ang Mo Kio is the lowest. For the most common 4 ROOM, the price of Central Area is the highest, and Woodlands is the lowest.
<img width="730" alt="截屏2024-04-25 下午8 20 24" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/e83bd62b-2e78-4c57-8c8e-1ff94e16ef1a">
<img width="726" alt="截屏2024-04-25 下午8 20 39" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/e0b65a37-114b-40d9-9faa-6eeec7fbd128">
<img width="744" alt="截屏2024-04-25 下午8 20 52" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/f2a4049a-5dc4-404f-80e4-7a9f0be24e32">


### By Flat Model
1. HDB Flat Models vs. Resale Price (1990-2024)
Each violin plot represents the distribution of resale prices for that specific model over the period from 1990 to 2024. The thinnest part of a violin indicates fewer sales at that price point, while the widest part indicates a price point with more transactions. For example, if a violin is particularly wide at the bottom, this means there are many flats sold at lower prices. Conversely, if it is wide at the top, there are many at higher prices.

Comparisons and Trends:

Price Range: Different models have different price ranges; for example, some models like the 'DBSS' have a very wide range, indicating a large variation in resale prices.
Median Prices: The white dot in the center of the colored area represents the median price for that model. Type S2, Type S1 and BDSS models tend to be more expensive.
Outliers: The lines extending from the main body of the violins suggest there are outliers in the data, with some flats selling for much higher or lower prices than the typical range. Like Improved, Model A and Adjioned Flat.

<img width="739" alt="截屏2024-04-25 下午8 21 33" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/1c83e940-9f60-4b67-9b95-b734b0bc14ad">


2. Trends of HDB Resale Prices on Each Flat Type (1990-2024)
The data distribution of 2 Room, IMPROVED MAISONETTE, PREMIUM MAISONETTE, PREMIUM APARTMENT LOFT and 3 GEN are relatively sparse and not carried out much in many years. Transaction. DBSS, Type S1, Type S2, ADJOINEd FLAT, 3 Gen's HDB Resale Prices have a higher starting point and a larger price range.
<img width="751" alt="截屏2024-04-25 下午8 21 57" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/ca69ec92-5d0a-4ca3-b605-b23696a7ed95">
<img width="748" alt="截屏2024-04-25 下午8 22 08" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/af8c66e3-6831-40d6-bff4-5b35b68da7a1">
<img width="764" alt="截屏2024-04-25 下午8 22 39" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/0b7b716c-d14d-4caf-a30c-9b7065409cf0">
<img width="764" alt="截屏2024-04-25 下午8 22 56" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/c4514ede-07aa-4483-9dba-dbca7fa58f68">
<img width="794" alt="截屏2024-04-25 下午8 23 11" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/62a0bf08-6e3f-4723-8384-9655b1b89ee1">
<img width="784" alt="截屏2024-04-25 下午8 23 28" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/1895a642-d57c-4a62-8b8f-eb39063bc6cc">
<img width="766" alt="截屏2024-04-25 下午8 23 44" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/52edbcee-f8dd-41c4-bc51-dfbf19b3fb5d">
<img width="766" alt="截屏2024-04-25 下午8 24 01" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/d844b4c9-48e7-47c1-9c6a-410e8b2e885e">
<img width="788" alt="截屏2024-04-25 下午8 24 15" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/b3d22628-2ef0-48b0-979a-183e016d6a82">
<img width="783" alt="截屏2024-04-25 下午8 24 37" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/b66f5e44-5ac9-4654-8f71-123b69ac3d90">
<img width="773" alt="截屏2024-04-25 下午8 24 50" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/9b4e0e0a-23d4-4afb-aed2-e7e26333e8b1">
<img width="754" alt="截屏2024-04-25 下午8 25 00" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/bba387be-f6ee-4fbb-8bb3-8d450f324a5b">


3. Trends of HDB Resale Prices for Each Flat Type
There are more flat models in the flat types of 2 ROOM, 3 ROOM, 4 ROOM and 5 ROOM. Among them, the price of Terrace model is relatively high, the price of Type S1 model of 4 ROOM is relatively high, and the price of DBSS and Type S2 of 5 ROOM is relatively high. And most flat types have increased the range of price range since 2010.
<img width="766" alt="截屏2024-04-25 下午8 25 27" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/30e6ef9e-75fc-43e6-8157-428c8f06ec91">
<img width="752" alt="截屏2024-04-25 下午8 25 44" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/2fa3d4e3-4360-4c31-aea1-18aec25d02fb">
<img width="757" alt="截屏2024-04-25 下午8 25 58" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/344ed7bc-e56c-4ea6-a1d3-efd0da2f8876">
<img width="759" alt="截屏2024-04-25 下午8 26 10" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/499679b5-05a2-4b28-a3e9-67b262fd5eaa">

### By storeys
1. Scatter Plot of Resale Prices by Storey Range (1990 - 2024)
It can be seen that generally speaking, the higher the floor, the higher the price. The price range of high floors is not as wide as that of low floors.
<img width="739" alt="截屏2024-04-25 下午8 26 36" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/0b5c80a2-b6f3-42ed-baa6-da75852bef05">


2. Median Resale Prices by Storey Range (1990 - 2024)
It can be seen that generally speaking, the higher the floor, the higher the price.
<img width="765" alt="截屏2024-04-25 下午8 26 58" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/234c870c-97e7-4e01-9caa-2130301680c4">


3. Scatter Plot of Resale Prices by Storey Range (1990 - 2024)
In the lower floors, there are mostly EXECUTIVE and MULTI-GENERATION models; in the middle floors, EXECUTIVE and 5 ROOM models are more common; in the higher floors, 4 ROOM and 5 ROOM models are more prevalent.
<img width="759" alt="截屏2024-04-25 下午8 27 16" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/27d1777f-b292-4ada-ad41-1ea1d9ce6345">


### By Floor Area
1. Relationship between Floor Area and Resale Price
As the floor area increases, the resale price also generally increases, which is indicated by the upward trend of the points.

The trendline suggests a positive correlation, meaning that as the floor area increases, the resale price tends to increase as well. However, there is significant variation around the trendline, indicating that while floor area is a factor in determining the resale price, it is not the only one, and there are properties that deviate from the trend.

In addition, The scatter plot suggests that there is a wide range of resale prices for properties with similar floor areas, indicating other factors beyond floor area may significantly influence the resale price, such as location, property age, market conditions, or specific property features.
<img width="711" alt="截屏2024-04-25 下午8 27 59" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/8119c893-b7c6-4618-8909-67c1fd6f60a0">


2. Resale Price Trend by Floor Area Sqm in 2012 - 2022
The resale price generally increases with the size of the flat, as denoted by the floor area. The 2-room, 3-room, 4-room, 5-room, and Executive flats, which are more common, form distinctive clusters with increasing floor areas and resale prices. 1 ROOM and multi-generation flats, while less numerous, show even higher resale prices, reflecting their larger sizes. The plot demonstrates that resale prices are stratified according to flat type, and there is a wide spread of resale prices within each flat type, especially as the floor area increases. This variation within each flat type category suggests other factors, in addition to floor area and flat type, affect the resale price.
<img width="751" alt="截屏2024-04-25 下午8 28 22" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/f625ef8e-0329-42c9-919e-d1d2a235e4c8">


### By Lease Commence Date
1. Resale Price By Lease Commence Year
The data reveals that properties with more recent Lease Commence Years tend to command higher Median Resale Prices. Additionally, each year is marked by a significant number of outliers, which could correspond to units in exceptionally favorable locations, like central urban areas or in proximity to key amenities such as schools and transportation hubs.
<img width="767" alt="截屏2024-04-25 下午8 28 57" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/ad182811-377a-47e3-a0b2-557179ecfc17">

### By Remaining Lease Year
1. Resale Price by Remaining Lease Period
The median resale price appears relatively stable across properties with different remaining lease periods, with a slight increase for those with the longest leases (90-100 years).

There is a significant number of outliers, particularly in the 40-50 year lease category, indicating that some properties are sold at prices much higher than the median for their lease period.

The interquartile range (the height of the boxes), which indicates the middle 50% of the data, does not show drastic differences across categories. This suggests that for each lease period category, the central half of the resale prices are somewhat similar in range.

The resale price range of these properties is larger in the remaining lease period of 40-50 years.

The data implies that while the remaining lease period does impact resale prices, there are properties with shorter leases that command high prices, possibly due to other factors such as location, renovation, and property condition.
<img width="740" alt="截屏2024-04-25 下午8 29 30" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/aed52b6d-16a8-435c-ad7c-e974fad52aed">


### By Distance to Nearest Amenities
1. Resale Price (Median of Each Town) vs Distance from Nearest MRT / Shopping Mall
The plot on the left shows the median resale price versus the distance from the nearest MRT station (in kilometers). Properties located closer to MRT stations appear to have a higher median resale price, with this price decreasing as the distance from the MRT station increases.

The plot on the right depicts the median resale price versus the distance from the nearest shopping mall (in kilometers). This plot also indicates that properties nearer to malls tend to have higher median resale prices, which seem to decrease with increasing distance from the malls.

<img width="750" alt="截屏2024-04-25 下午8 30 03" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/dd1f3ce3-badb-4438-9d77-23f08eb09741">


2. Resale Price vs Distance from Nearest MRT / Shopping Mall
The left plot shows resale prices versus distance from the nearest MRT station. It indicates a high density of properties with varying resale prices close to MRT stations, and a general decrease in resale prices as the distance from MRT stations increases.

The right plot shows resale prices versus distance from the nearest shopping mall. Similarly, a dense cluster of properties is observed close to shopping malls, with resale prices appearing to decline with increased distance from these amenities.

Both plots collectively suggest that proximity to MRT stations and shopping malls has a considerable impact on property resale prices in Singapore, with properties closer to these amenities generally commanding higher resale prices.
<img width="750" alt="截屏2024-04-25 下午8 30 31" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/5a2e96a9-19cd-4f7b-b267-ad453059851f">


### By travel time
1. Resale Price (Median of Each Town) vs Travel Time to Nearest MRT / Mall
The left plot shows median resale prices versus travel time to the nearest MRT station (in minutes). It indicates that properties with shorter travel times to MRT stations generally have higher median resale prices. There is a noticeable decline in median prices as the travel time increases, but the trend is not uniformly steep.

The right plot depicts median resale prices against travel time to the nearest mall (in minutes). Similar to the first plot, higher median resale prices are observed for properties with less travel time to malls, with a visible decrease in median resale prices as travel time increases.

Both plots suggest a trend where convenience, indicated by shorter travel times to key amenities like MRT stations and malls, has a positive correlation with the median resale price of properties. However, there are some data points with high resale prices even at longer travel times, which could be due to other location-specific factors.
<img width="743" alt="截屏2024-04-25 下午8 30 58" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/9eb8ed88-c179-4e19-ae6d-a6d9e1bec364">


2. Resale Price vs Travel Time from Nearest MRT / Shopping Mall
The plot on the left shows resale prices versus travel time to the nearest MRT station. There is a large concentration of properties with a wide range of prices close to MRT stations (travel time close to 0). Prices generally decrease with longer travel times, but not uniformly, and there are properties with high resale prices even with longer travel times to MRT.
The plot on the right illustrates resale prices relative to travel time to the nearest shopping mall. Again, a dense cluster of properties near malls (low travel time) is evident with high resale prices that tend to decrease as the travel time increases.
Both plots suggest that shorter travel times to important amenities like MRT stations and malls are generally associated with higher resale prices, although there are properties that defy this trend, which may be influenced by other factors like property attributes, market conditions, or specific locations.
<img width="754" alt="截屏2024-04-25 下午8 31 22" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/b4ff1c81-fda2-4a28-a870-1a1f21d46ac2">



#### By Location
1. Distribution of HDB Resale Prices by Location
Central Region: There is a high concentration of HDB units with higher resale prices, indicated by warm colors, suggesting this region's properties are among the most expensive.

Eastern and Northeasern Regions: These areas display a moderate to high range of resale prices with several hotspots where prices are particularly high, suggesting desirable locations within these regions.

Northern Region: The resale prices here vary, with some clusters of higher-priced units but generally cooler colors indicating more moderate prices.

Western Region: There's a diverse range of resale prices, with pockets of both high and moderate value HDB units, as shown by the mix of colors.

Overall, the central region appears to command the highest resale prices, with pockets of high-value properties scattered throughout the eastern, northeastern, and western regions. The northern region seems to have more uniformly moderate prices. This pattern suggests that central and accessible areas tend to have higher property values, reflecting the desirability of living closer to the city center or other amenities.
<img width="670" alt="截屏2024-04-25 下午8 32 53" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/cad558c2-c1cf-4c4b-8de8-d4229e255e80">


2. Distribution of HDB Price Per Square Meter by Location
- Central Region: This area displays some of the highest values, as indicated by the presence of red tones, suggesting that HDB flats in the central region have a higher price per square meter, reflecting high demand and desirability.
- Eastern and Northeastern Regions: These regions show a gradient of prices per square meter, with areas closer to the central region having warmer colors, indicating higher prices, which gradually decrease as one moves outward.
- Northern Region: The price per square meter here seems more moderate compared to the central region, with mostly yellow tones indicating lower prices than the central region but variable across different locales.
- Western Region: Similar to the northern region, the western region shows moderate prices per square meter, with a mix of yellow and orange tones, suggesting a range of values, with some areas having relatively higher prices.
The overall pattern indicates that the central region tends to be the most expensive on a per square meter basis, with prices generally decreasing as one moves towards the outer regions of Singapore. However, pockets of higher value can be seen in non-central areas, possibly due to local factors such as proximity to amenities or transport links.
<img width="704" alt="截屏2024-04-25 下午8 33 30" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/b0af2c10-ac0a-4bc7-9de5-816de3282a62">

#### HeatMap
1. Performed one-hot encoding on categorical columns ('town', 'flat_type', 'flat_model', 'region') to convert categorical data into numerical format, which is necessary for most machine learning algorithms.

2. Removed columns that were deemed not useful for the prediction model or were replaced by derived features: 'Unnamed: 0', 'month', 'block', 'street_name', 'storey_range', 'lease_commence_date', 'address', 'latitude', 'longitude', 'nearest_mrt_name', 'nearest_mall_name'.

3. Calculated the Variance Inflation Factor (VIF) to identify features with multicollinearity. Found that nearly 65 features had a VIF greater than 5, likely due to the increased number of features from one-hot encoding and feature engineering which can often introduce multicollinearity.

4. Instead of removing all features with a VIF greater than 5, which would significantly reduce the feature set, we chose to eliminate features based on a correlation threshold, removing one of a pair of features if their correlation exceeded 0.8. This approach targets specifically high multicollinearity without excessively reducing the number of features.

5. The merged DataFrame was then reviewed to confirm successful encoding, and the updated DataFrame was visualized using a heatmap to observe correlations between features, guiding the final feature selection to mitigate multicollinearity.
<img width="744" alt="截屏2024-04-25 下午8 34 13" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/7ce4b96b-6100-4e26-91b8-dbbf4f02e650">

   
### Statistical Analysis

<img width="517" alt="截屏2024-04-25 下午8 31 44" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/7462780a-ee5a-40f3-b4b5-08f160c4360a">
<img width="509" alt="截屏2024-04-25 下午8 32 04" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/306187aa-baee-4d20-9024-b94053263f7e">
<img width="343" alt="截屏2024-04-25 下午8 32 13" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/662ed0ce-9169-4114-8fd2-e6fce8dd0fc7">

---

## Task 5: Modeling and Visualization
In this part, we trained 3 models: Multi Linear Regression, LightGBM, Random Forest
### Multi Linear Regression
<img width="588" alt="截屏2024-04-25 下午8 35 35" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/1a3db3d1-8058-4903-a655-484d78db5424">
<img width="740" alt="截屏2024-04-25 下午8 35 50" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/b828d1f5-953a-49aa-a0bd-6b97d797c11c">


### LightGBM
The MSE for LightGBM is 729,604,786, which is significantly lower than that for Linear Regression, indicating that the predictions from the LightGBM model are much closer to the actual values. Lower MSE values generally signify better model performance, as they reflect smaller average errors in the predictions.

The R² score, which measures the proportion of variance in the dependent variable that is predictable from the independent variables, is substantially higher for LightGBM (0.9722) compared to Linear Regression (0.8315). An R² score closer to 1 indicates that the model explains a higher proportion of the variance in the data.
<img width="585" alt="截屏2024-04-25 下午8 36 27" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/90678cbd-7cef-4ba1-8917-197a3656edf3">

#### Tuning the lightGBM Model
We'll use Grid Search with cross-validation to find the optimal set of parameters. This method involves testing the model with various combinations of parameters and selecting the combination that results in the best score based on a specified metric, typically R² or Mean Squared Error for regression tasks.

Optimal Parameters:

Learning Rate: 0.1

Max Depth: 20

Number of Estimators: 500

Number of Leaves: 91

These parameters suggest a model configuration that is robust, allowing for a deeper tree structure (max depth of 20 and 91 leaves), increased complexity (which is handled well by the large number of estimators), and a relatively faster convergence (learning rate of 0.1).

Performance:

Cross-Validation R² Score: 0.975976889769278 Test Set R² Score: 0.9761390928787514

Both scores are exceptionally high, indicating that the model explains about 97.6% of the variability in the target variable across different subsets of the dataset. Showing that the model is highly effective at making predictions that are very close to the actual data points.
<img width="735" alt="截屏2024-04-25 下午8 37 48" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/866623f8-1cfc-4000-8691-4f1ee37a5714">

### Random Forest

The Random Forest Regressor model achieved an out-of-bag R-squared score of approximately 0.9695, indicating that around 96.95% of the variance in the target variable is captured by the model during training.

When applied to the test set, the model achieved a Mean Squared Error (MSE) of approximately 713,820,681 and an R-squared score of approximately 0.9728, indicating this model captures a high proportion of the variance in the target variable and exhibits low prediction error on unseen data. The MSE is lower than both the Linear Regression and LightGBM above, but the R square score is a bit smaller than LightGBM.

Next, we would further tuning the model parameters to see whether we could achieve better model performance.
<img width="591" alt="截屏2024-04-25 下午8 39 30" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/da92fc8e-b155-41c9-8e7e-d1f51b20e6a3">

#### Tuning the Random Forest Model
Optimal Parameters:

Max features: sqrt

Max depth: None

Min samples split: 5

Min samples leaf: 1

Number of estimators: 30

The optimal parameter settings indicate that 'sqrt' is chosen as the maximum number of features, indicating the square root of the number of features considered when splitting each node. The absence of a maximum depth restriction allows the decision trees to grow to any depth, potentially aiding in better fitting the data. Furthermore, a minimum samples split of 5 and minimum samples leaf of 1 help control model complexity to prevent overfitting.

Performance:

By tuning the hyperparameters, we enhanced the performance of the random forest model.

A Cross Validation R² Score of 0.9725 suggests that the model effectively captures a substantial portion of the variance in the data during training. This indicates a good fit to the training data.

With a Test set R² score of 0.9736, the model demonstrates strong generalization to unseen data, highlighting its robust performance beyond the training set.
<img width="556" alt="截屏2024-04-25 下午8 40 41" src="https://github.com/Maddiezheng/NUS_IT5100F-2404/assets/44678784/90c8263c-3f5a-49a5-b75e-d671f45a79d4">

We filtered out the features with importance greater than 0.01. Based on the feature importance plot, we can obviously observe that the top five features driving the resale price of HDB are year, floor_area_sqm, flat_model_premium and town_premium and flat_type_3_ROOM.

---

## Task 6: Report Insights and Conclusions
In this project, linear regression, random forest and lightGBM are used to investigate the drivers of HDB resale price.

Mutiple Linear Regression Model:

- Training: The Linear Regression model was trained using the standard training dataset split from the entire dataset.
- Performance Metrics: The model's performance was evaluated using the Mean Squared Error (MSE) and R² (coefficient of determination) metrics.

LightGBM Model:

- Training: The LightGBM model utilized the same training set but leveraged a gradient boosting framework, which is generally more robust to a variety of data irregularities, including outliers and non-linear relationships.
- Configuration: The model was configured with parameters aimed at optimizing performance and preventing overfitting, such as num_leaves, learning_rate, and bagging_fraction.
- Performance Metrics: Like the Linear Regression, its performance was evaluated using MSE and R².

Random Forest:

- Training: The Random Forest model also employed the same training dataset as the Linear Regression and LightGBM models. Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees during training.
- Configuration: The model's hyperparameters were carefully selected to balance bias and variance, promoting generalization while avoiding overfitting. Parameters such as n_estimators, max_depth,max_features were tuned.
- Performance Metrics: Similarly, the Random Forest's performance was assessed using the MSE and R² metrics on the evaluation dataset.


We compare these three models from several aspects.

|Model |MSE |R^2|Cross-Validation R^2|OOB R^2|
|------------|------|------|------|------|
| Multiple Linear Regression Model |4420438749.1606|0.8315| / | / |
| LightGBM  |729604786.8829|0.9722|/ | / |                 
| LightGBM(Optimized) |/| 0.9761 | 0.9760 | / |
| Random Forest |713820681.3898| 0.9728 | / | 0.9695 |
| Random Forest(Optimized) |/|  0.9736 | 0.9725 | / |

- Accuracy: The LightGBM model generally exhibited superior accuracy and a higher R² value compared to both the Random Forest and Linear Regression models, indicating a better fit to the data variability.

- Efficiency: LightGBM's handling of large datasets and its ability to deal with various types of data make it a more robust choice for complex regression tasks, surpassing both Random Forest and Linear Regression models in this aspect.

- Use Case Fit: For datasets where relationships between variables might be non-linear or when the dataset features a large number of observations and features, LightGBM is likely a more effective choice compared to both Random Forest and Linear Regression

- Implementation Considerations: While LightGBM provides better performance metrics, it requires careful tuning of parameters to avoid overfitting and to optimize performance. In contrast, Random Forest and Linear Regression, with fewer parameters to tune, might be more straightforward to implement and interpret. However, the Random Forest model offers better performance than Linear Regression while still maintaining some interpretability.

Based on the observed performance metrics, LightGBM is recommended for tasks requiring the handling of complex, large-scale datasets due to its efficiency in processing and potential for higher accuracy. However, for simpler or smaller datasets, or when model interpretability is a priority, Random Forest might still be a suitable choice, surpassing Linear Regression in predictive accuracy. In this project, we would choose LightGBM as the most relevant model.

Look at the output of the models, linear regression showed that flat_model is the most influential features for HDB resale price. But ramdom forest and lightGBM presented different results. From random forest, year, floor area, flat_model_premium and town_premium and flat_type_3_ROOM are top 5 features. From LightGBM, travel time to mrt, travel time to mall, floor area, remaining lease and year are top five features. It's possible for tree-based models to give lower importance to categorical variables compared to numerical variables. Combining the results, we think year, floor area, flat type, travel time and remaining lease are important drivers for HDB resale price.


Here are the recommendations to help buyers and sellers better understand and utilize the transaction data for HDB flats:

For buyers:

1. Consider Key Factors: When selecting an HDB flat, pay close attention to several critical factors, including the year of construction, floor area, flat type, commute time to the nearest MRT station and shopping centers, and the remaining lease term. These factors have been proven to significantly influence the resale value of the properties.

2. Evaluate Future Value: Consider the remaining lease term of the flat and future development plans in the area. These factors may impact future property prices, thereby helping buyers make more forward-looking investment decisions.

3. Utilize Data Models: Using data models to predict potential property prices can help buyers more accurately assess suitable properties within their budget.

For sellers:

1. Highlight Property Features: When selling an HDB flat, emphasize features that have the most significant impact on value, such as the year of construction, floor area, flat type, and locational advantages (like proximity to transport and commercial facilities), as well as the remaining lease term.

2. Sell at the Right Time: Considering the impact of the year of construction and the remaining lease term on prices, choosing to sell when the property value is relatively high could yield greater returns.

3. Use Analytical Tools: Employ various data models (such as Random Forest and LightGBM) to predict the market price of the property. Combining the insights from these tools to formulate a selling strategy can lead to more precise pricing and attract suitable buyers.

---



