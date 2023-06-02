import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("white")
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA as ARIMA
from sklearn.decomposition import PCA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import tqdm as tqdm
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession

def read_data(file1, file2):
    # Create a SparkSession
    spark = SparkSession.builder.appName("MyApp").getOrCreate()

    # Read the CSV file into a PySpark DataFrame
    train = spark.read.format("csv").option("header", "true").load(file1)


    # Show the first few rows of the DataFrame

    census_df = spark.read.format("csv").option("header", "true").load(file2)


    return train, census_df
    
def counties_stats(train_data):
    
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    import matplotlib.pyplot as plt
    # Group the data by state and count the number of counties in each state
    count_by_state = train_data.groupBy("state").count().orderBy(col("count").desc())

    # Convert the PySpark DataFrame to a Pandas DataFrame for plotting
    count_by_state_pandas = count_by_state.toPandas()

    # Create a count plot using the Pandas plot method
    count_by_state_pandas.plot(kind="barh", x="state", y="count", color="blue", legend=None)
    plt.xlabel("Counties count")
    plt.ylabel("")
    plt.title("Number of counties per State")
    plt.grid(True, color='black',linewidth = "0.6", alpha=0.6, axis="x")
    plt.show()


import pyspark.sql.functions as F
import matplotlib.pyplot as plt

def corr_plot_state(train_data):
    '''
    Barplot to plot correlation between time and microbusiness_density for every state to observe the correlation 
    type for the states 
    '''
    # Convert dates to days since today
    converted_dates = (F.datediff(F.current_date(), F.to_date(train_data["first_day_of_month"])) * -1).alias("days_passed")

    # Select necessary columns
    df1 = train_data.select(converted_dates, "state", "microbusiness_density")

    # Calculate correlation by state
    corr_by_state = df1.groupBy("state").agg(F.corr("days_passed", "microbusiness_density").alias("correlation"))

    # Plot results
    color_list = [F.when(corr_by_state["correlation"] > 0, 'green').otherwise('red').alias("color")]
    corr_by_state = corr_by_state.select("correlation", "state", *color_list)
    corr_by_state = corr_by_state.orderBy("correlation").toPandas()
    fig = plt.figure(figsize=(10,15))
    plt.barh(width=corr_by_state["correlation"],
             y=corr_by_state["state"],
            color=corr_by_state["color"])
    plt.grid(True, color = "black", linewidth = "0.5", alpha=0.5)
    plt.title("Correlations between dates and microbusiness_density by state")
    plt.xlabel("Correlation")
    plt.show()



import pandas as pd
def data_prepare(train_data):
    '''
    Initialize subset of data for plotting
    '''
    converted_dates = list()
    for date in pd.to_datetime(train_data["first_day_of_month"]):
        converted_dates.append((date - pd.to_datetime(date.today())).days)
    df2 = pd.concat([pd.Series(converted_dates),
                  train_data["county"] + ", " + train_data["state"],
                  train_data["state"],
                  train_data["microbusiness_density"]],
                 axis=1)
    df2.columns = ["days_passed", "county_state", "state", "microbusiness_density"]

    corr_by_county = list()
    for county in df2["county_state"].unique():
        corr_by_county.append(df2.loc[df2["county_state"]==county, ["days_passed", "microbusiness_density"]].corr().values[0])
    corr_by_county = pd.Series(corr_by_county, index=df2["county_state"].unique())
    return df2, corr_by_county

def corr_plot_county(train_data,counties_number,title,least_corr=False):
    '''
    Lineplot to plot correlation between time and microbusiness_density for the highest/least correlated counties
    '''
    df2,corr_by_county = data_prepare(train_data)
    fig = plt.figure(figsize=(14,counties_number*3))
    x = 1
    for idx, county in enumerate(corr_by_county.sort_values(ascending=least_corr).index[:counties_number]):
        ax = fig.add_subplot(counties_number+1,1,x)
        sns.lineplot(x=train_data.loc[df2["county_state"]==county, 'first_day_of_month'], y=train_data.loc[df2["county_state"]==county, "microbusiness_density"], color="#FFA500")
        plt.title(county + f" (Correlation: {np.round(corr_by_county.sort_values(ascending=least_corr).values[:counties_number][idx],2)})")
        plt.xlabel("")
        plt.tick_params(
            axis='x',          
            which='both',      
            bottom=False,      
            top=False,         
            labelbottom=False)
        x+=1
    ax = fig.add_subplot(counties_number+1,1,x)
    sns.lineplot(x=train_data.loc[df2["county_state"]==corr_by_county.sort_values(ascending=least_corr).index[counties_number], 'first_day_of_month'], y=train_data.loc[df2["county_state"]==corr_by_county.sort_values(ascending=least_corr).index[counties_number], "microbusiness_density"], color="#FFA500")
    plt.title(corr_by_county.sort_values(ascending=least_corr).index[:counties_number+1][-1] + f" (Correlation: {np.round(corr_by_county.sort_values(ascending=least_corr).values[:counties_number+1][-1],2)})")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("")
    ttl = plt.suptitle(f"{counties_number} {title}", fontsize=14, fontweight="bold")
    ttl.set_position([.51, 1.00])
    plt.tight_layout()
    plt.show()

def change_dtypes(train):

    from pyspark.sql.functions import year, month, dayofmonth 
    train = train.withColumn("date", F.to_date("first_day_of_month", "yyyy-MM-dd")) \
                 .withColumn("year", year("date")) \
                 .withColumn("month", month("date")) \
                 .withColumn("day", dayofmonth("date"))
    return train

def basic_statistics(train,census):
    from pyspark.sql.functions import countDistinct
    import pyspark.sql.functions as F
    
    num_counties = train.select(countDistinct('cfips')).collect()[0][0]
    print("Number of counties in train data:",num_counties)
    census_count = census.select('cfips').distinct().count()
    print("Number of counties in the census data: ",census_count)

    #remove - census_count = census.groupBy('cfips').count().count()

    # Inner join the two dataframes on 'cfips' column
    common_counties = census.join(train, 'cfips', 'inner')
    
    # Count the number of rows in the resulting dataframe to get the number of counties present in both datasets
    common_counties_count = common_counties.select(F.col('cfips')).distinct().count()
    print("Number of Common counties in the joined dataframe:", common_counties_count)


def drop_columns(train, colnames):
    for i in colnames:
        train = train.drop(i)
    return train


def cast_dataframes(df, col_names, dtypes):
    from pyspark.sql.functions import col
    for i,j in zip(col_names, dtypes):
        df = df.withColumn(i, col(i).cast(j))
    return df



def check_for_nulls(df,column_name):
    '''
      This function checks for nulls in the merged dataset for a given column
    '''
    k = df[df[column_name].isna()]['county']
    k1= k.unique()
      
    print("Counties that have null values are:")
    for i in k1:
        print(i +" has "+str(list(k).count(i))+" null values") 
        

def feature_importances(X, merged_df):
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import RandomForestRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.sql.functions import col

    from pyspark.ml.feature import StandardScaler

    assembler = VectorAssembler(inputCols= X.columns, outputCol="features")
    merged_df = merged_df.drop('first_day_of_month','date')
    data = assembler.transform(merged_df)

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    scaler_model = scaler.fit(data)
    scaled_df = scaler_model.transform(data)
    scaled_features = scaled_df.select("scaledFeatures").rdd.flatMap(lambda x: x).collect()

    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    rf = RandomForestRegressor(labelCol="microbusiness_density", featuresCol="features", numTrees=100, maxDepth=4, seed=42)
    model = rf.fit(trainingData)
    importances = model.featureImportances
    for feature, importance in zip(X.columns, importances):
        print("{}: {}".format(feature, importance))
    importances_dict = {}
    for feature, importance in zip(X.columns, importances):
        importances_dict[feature] = importance

    import pandas as pd
    importances_df = pd.DataFrame(list(importances_dict.items()),columns = ['Feature','Importance'])

    importances_df.sort_values(by=['Importance'], inplace=True, ascending=False)

    importances_df.plot.barh(x='Feature', y='Importance', figsize=(10,15))


def PCA(X):
    from pyspark.ml.feature import PCA
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql.functions import monotonically_increasing_id
    from pyspark.sql.functions import sum as sql_sum
    from pyspark.sql.window import Window
    assembler = VectorAssembler(inputCols=X.columns, outputCol="features")
    data = assembler.transform(X)
    variance_list = []
    for i in range(1, 10):
        pca = PCA(k=i, inputCol="features", outputCol="pca_features")
        pca_model = pca.fit(data)
        variance = pca_model.explainedVariance.toArray()
        variance_sum = sum(variance)
        variance_list.append(variance_sum)

    cumulative_variance = []
    cumulative_sum = 0
    for i in range(len(variance_list)):
        cumulative_sum += variance_list[i]
        cumulative_variance.append(cumulative_sum)
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (12,6)

    fig, ax = plt.subplots()
    xi = range(1, 10)

    plt.ylim(0.0,1.1)
    plt.plot(xi, variance_list, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(range(1, 10))
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

    ax.grid(axis='x')
    plt.show()

if __name__=="__main__":  
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    import matplotlib.pyplot as plt
    import seaborn as sns

    spark = SparkSession.builder.appName("MyApp").getOrCreate()
    train, census = read_data('train.csv','census_starter.csv')

    counties_stats(train)
    #line_plot(train.toPandas(),"first_day_of_month","microbusiness_density","Microbusiness density overtime","#FFA500")
    corr_plot_state(train)
    train = change_dtypes(train)
    basic_statistics(train, census)
    #train = drop_columns(train, ['date'])
    census = cast_dataframes(census, census.columns, len(census.columns) * ["float"])
    corr_plot_county(train.toPandas(), 10, "largest positive correlations between dates and microbusiness densities")


    merged_df = census.join(train, on='cfips', how='inner')
    mer = merged_df.toPandas()
    merged = merged_df.toPandas()
    check_for_nulls(merged,'median_hh_inc_2020')
    check_for_nulls(merged,'median_hh_inc_2021')
    check_for_nulls(merged,'median_hh_inc_2018')
    merged_df = merged_df.dropna()
    mer = merged_df.toPandas()
    merged_df = cast_dataframes(merged_df, ['active','microbusiness_density'],['integer','float'])
    y1 = mer['microbusiness_density']

    from pyspark.ml.regression import GBTRegressor
    from pyspark.ml.feature import VectorAssembler
    import pyspark.sql.functions as F
    from pyspark.sql.types import DoubleType
    from matplotlib import pyplot

    # Split data into X and y
    X = merged_df.drop('row_id', 'county', 'state', 'microbusiness_density','first_day_of_month','date')
    y = merged_df['microbusiness_density']
    feature_importances(X, merged_df)
    PCA(X)