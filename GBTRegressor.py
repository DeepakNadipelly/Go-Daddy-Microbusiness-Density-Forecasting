from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# Convert the input data to a feature vector
assembler = VectorAssembler(inputCols=merged_df_new.columns, outputCol="features")
df = assembler.transform(merged_df_new)

# Convert the target variable to a float
df = df.withColumn("microbusiness_density", df["microbusiness_density"].cast("float"))

# Split the data into training and testing sets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=123)

# Convert the LightGBM model to a PySpark Estimator
estimator = GBTRegressor(featuresCol="features", labelCol="microbusiness_density", predictionCol="prediction",
                         maxDepth=5, maxBins=32, maxIter=100)

# Define the parameter grid for tuning
param_grid = ParamGridBuilder() \
    .addGrid(estimator.maxDepth, [2,105]) \
    .addGrid(estimator.maxBins, [16,32]) \
    .addGrid(estimator.maxIter, [10,50,100]) \
    .build()

# Define the evaluation metric
evaluator = RegressionEvaluator(labelCol="microbusiness_density", predictionCol="prediction", metricName="rmse")

# Define the cross-validation object
cv = CrossValidator(estimator=estimator, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=2)

# Fit the model to the training data
model = cv.fit(train_data)

# Evaluate the model on the test data
predictions = model.transform(test_data)
rmse = evaluator.evaluate(predictions)
print("RMSE: ", rmse)
