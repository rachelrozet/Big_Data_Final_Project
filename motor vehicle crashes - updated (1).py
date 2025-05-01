# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Import Data Set

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/Motor_Vehicle_Collisions___Crashes.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "motor_vehicle_collisions"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# Crash time and contributing factors seem to vary greatly in dataset. Create query to determine how many unique values are in each column.

from pyspark.sql.functions import countDistinct

time_count = df.select(countDistinct("CRASH TIME")).collect()[0][0]
factor_count = df.select(countDistinct("CONTRIBUTING FACTOR VEHICLE 1")).collect()[0][0]

print(f"Number of unique values: {time_count}")
print(f"Number of unique values: {factor_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering & Data Cleaning

# COMMAND ----------

# Segment CRASH DATE into year, month, and day of week columns
from pyspark.sql.functions import unix_timestamp, from_unixtime

df = df.withColumn("date", from_unixtime(unix_timestamp("CRASH DATE", "MM/dd/yyyy"), "yyyy-MM-dd").cast("date"))

from pyspark.sql.functions import year, month, dayofweek, to_date
df = df.withColumn("year", year("date"))
df = df.withColumn("month", month("date"))
df = df.withColumn("day_of_week", dayofweek("date"))

display(df)

# COMMAND ----------

# Create 4 dedicated time segments to replace individual CRASH TIME (morning, afternoon, evening, and night)
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
df = df.withColumn("hour", split(df["CRASH TIME"], ":").getItem(0))
df = df.withColumn("hour", col("hour").cast(IntegerType()))

df = df.withColumn("time_segment",when(df["hour"] < 6, 'night').when(df["hour"] < 12, 'morning').when(df["hour"] < 18, 'afternoon').otherwise('evening'))

display(df)

# COMMAND ----------

# Cast integer columns as integer types
df = df.withColumn("NUMBER OF PERSONS INJURED", col("NUMBER OF PERSONS INJURED").cast(IntegerType()))
df = df.withColumn("NUMBER OF CYCLIST KILLED", col("NUMBER OF CYCLIST KILLED").cast(IntegerType()))
df = df.withColumn("NUMBER OF MOTORIST INJURED", col("NUMBER OF MOTORIST INJURED").cast(IntegerType()))

# Create aggregated column for total persons injured in an accident
df = df.withColumn('total_injured', df["NUMBER OF PERSONS INJURED"] + df["NUMBER OF PEDESTRIANS INJURED"] + df["NUMBER OF CYCLIST INJURED"] + df["NUMBER OF MOTORIST INJURED"])

# Create aggregated column for total persons killed in an accident
df = df.withColumn('total_killed', df["NUMBER OF PERSONS KILLED"] + df["NUMBER OF PEDESTRIANS KILLED"] + df["NUMBER OF CYCLIST KILLED"] + df["NUMBER OF MOTORIST KILLED"])

# Calculate fatality rate for accidenets
df = df.withColumn("fatality_rate",when((df['total_injured']+df['total_killed']) > 0, round(df['total_killed']/(df['total_injured']+df['total_killed']),2)).otherwise(0))

display(df)

# COMMAND ----------

# Aggregate values of contributing vehicle factors into lists
df_list = df.agg(collect_list("CONTRIBUTING FACTOR VEHICLE 1").alias("list1"), collect_list("CONTRIBUTING FACTOR VEHICLE 2").alias("list2"),collect_list("CONTRIBUTING FACTOR VEHICLE 3").alias("list3"),collect_list("CONTRIBUTING FACTOR VEHICLE 4").alias("list4"),collect_list("CONTRIBUTING FACTOR VEHICLE 5").alias("list5"))

# Combine lists and remove duplicates
df_distinct = df_list.select(flatten(array("list1", "list2","list3","list4","list5")).alias("combined_list")) \
    .select(array_distinct("combined_list").alias("distinct_list"))

# Extract ditisnct values for contributing vehicle factors (to later categorize into 5 grouped buckets)
result_list = df_distinct.first()["distinct_list"]

print(result_list)

# COMMAND ----------

# Create column to define total cars involved in an accident

df = df.withColumn("total_cars",when(df["VEHICLE TYPE CODE 5"].isNotNull(), 5).when(df["VEHICLE TYPE CODE 4"].isNotNull(), 4).when(df["VEHICLE TYPE CODE 3"].isNotNull(), 3).when(df["VEHICLE TYPE CODE 2"].isNotNull(), 2).otherwise(1))

display(df)

# COMMAND ----------

# Drop irrelevant columns

columns_to_drop = ["CRASH DATE", "CRASH TIME", "LATITUDE", "LONGITUDE", "LOCATION", "ON STREET NAME", "CROSS STREET NAME", "OFF STREET NAME", "CONTRIBUTING FACTOR VEHICLE 4", "CONTRIBUTING FACTOR VEHICLE 5", "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5","date","hour"]
df = df.drop(*columns_to_drop)

# Filter dataset to only contain accidents with 3 cars or more
df = df.filter(col("total_cars") < 4)

# Ensure borough and zip code are not null
df = df.where(col("BOROUGH").isNotNull())
df = df.where(col("ZIP CODE").isNotNull())

# Fill null values of contributing factors
df = df.fillna({"CONTRIBUTING FACTOR VEHICLE 1": 'Unspecified', "CONTRIBUTING FACTOR VEHICLE 2": 'Unspecified', "CONTRIBUTING FACTOR VEHICLE 3": 'Unspecified'})

display(df)

# COMMAND ----------

# Replace contributing factors for each vehicle with categorical buckets defined by team (Personal Motorist Issue, Vehicle Failure, External Force / Reaction, Driver Distraction, and Poor Driving)

from pyspark.sql.functions import when, col

def replace_if_in_list(df, column_name, value_list, replacement_value):

    return df.withColumn(
        column_name,
        when(col(column_name).isin(value_list), replacement_value).otherwise(col(column_name))
    )

personal_motorists = ["Alcohol Involvement", "Illnes", "Lost Consciousness", "Fell Asleep", "Drugs (illegal)", "Fatigued/Drowsy", "Physical Disability", "Prescription Medication", "Drugs (Illegal)", "Illness"]
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 1", personal_motorists, "Personal Motorist Issue")
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 2", personal_motorists, "Personal Motorist Issue")
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 3", personal_motorists, "Personal Motorist Issue")

vehicle_failure = ["Steering Failure","Other Vehicular","Accelerator Defective","Brakes Defective","Backing Unsafely","Tinted Windows","Other Lighting Defects","Tire Failure/Inadequate","Headlights Defective","Windshield Inadequate","Vehicle Vandalism", "Shoulders Defective/Improper"]
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 1", vehicle_failure, "Vehicle Failure")
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 2", vehicle_failure, "Vehicle Failure")
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 3", vehicle_failure, "Vehicle Failure")

external_force = ["Pavement Slippery","Unspecified","Reaction to Uninvolved Vehicle","Traffic Control Disregarded","Oversized Vehicle","Pedestrian/Bicyclist/Other Pedestrian Error/Confusion","View Obstructed/Limited","Glare","Obstruction/Debris","Animals Action","Pavement Defective","Driverless/Runaway Vehicle","Lane Marking Improper/Inadequate","Traffic Control Device Improper/Non-Working","Tow Hitch Defective","Reaction to Other Uninvolved Vehicle","1"]
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 1", external_force, "External Force / Reaction")
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 2", external_force, "External Force / Reaction")
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 3", external_force, "External Force / Reaction")

driver_distraction = ["Driver Inattention/Distraction","Passenger Distraction","Outside Car Distraction","Eating or Drinking","Cell Phone (hands-free)","Cell Phone (hand-Held)","Using On Board Navigation Device","Other Electronic Device","Listening/Using Headphones","Texting","Cell Phone (hand-held)"]
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 1", driver_distraction, "Driver Distraction")
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 2", driver_distraction, "Driver Distraction")
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 3", driver_distraction, "Driver Distraction")

poor_driving = ["Aggressive Driving/Road Rage","Following Too Closely","Passing Too Closely","Failure to Yield Right-of-Way","Driver Inexperience","Passing or Lane Usage Improper","Turning Improperly","Unsafe Lane Changing","Unsafe Speed","Failure to Keep Right","80"]
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 1", poor_driving, "Poor Driving")
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 2", poor_driving, "Poor Driving")
df = replace_if_in_list(df, "CONTRIBUTING FACTOR VEHICLE 3", poor_driving, "Poor Driving")

display(df)

# COMMAND ----------

# Rename columns as lowercase for easy use of typing
df = df.withColumnsRenamed({"BOROUGH": "borough", "ZIP CODE": "zip_code", "NUMBER OF PERSONS INJURED":"injured_persons", 
                            "NUMBER OF PERSONS KILLED":"killed_persons", "NUMBER OF PEDESTRIANS INJURED":"injured_peds",
                            "NUMBER OF PEDESTRIANS KILLED":"killed_peds", "NUMBER OF CYCLIST INJURED":"injured_cyclists",
                            "NUMBER OF CYCLIST KILLED":"killed_cyclists", "NUMBER OF MOTORIST INJURED":"injured_motorists",
                            "NUMBER OF MOTORIST KILLED":"killed_motorists", "CONTRIBUTING FACTOR VEHICLE 1":"factor_v1",
                            "CONTRIBUTING FACTOR VEHICLE 2":"factor_v2","CONTRIBUTING FACTOR VEHICLE 3":"factor_v3",
                            "COLLISION_ID":"crash_id"})

# Cast zip code as integer type
df = df.withColumn("zip_code", col("zip_code").cast(IntegerType()))

df=df.dropna()

display(df)

# COMMAND ----------

# Get row count of final dataset (for comparison of original dataset)
row_count_len = len(df.collect())
print("Row count using len():", row_count_len)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Testing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 1 - Predict Time Segment of Accident (Morning, Afternoon, Evening, Night)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Select feature columns
categorical_cols = ['borough', 'total_cars', 'factor_v1']
numerical_cols = ['month', 'zip_code', 'day_of_week', 'total_injured', 'total_killed']
label_col = 'time_segment'

# Step 1: Index + encode categorical features
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid='keep') for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_ohe") for col in categorical_cols]

# Step 2: Index label
label_indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid='keep')

# Step 3: Assemble features
feature_cols = [col + "_ohe" for col in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembler.setHandleInvalid("skip")

# COMMAND ----------

# Initialize model and pipeline
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler, rf])

# Split data into train and test datasets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Fit model on train data
model = pipeline.fit(train_data)

# Test model on test data
predictions = model.transform(test_data)

# COMMAND ----------

# Evaluate model accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", 
                    predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.3f}")

predictions.select("time_segment", "prediction").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 2 - Predict Borough in Which Accident Occurred

# COMMAND ----------

# Import necessary pyspark libraries
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Select feature columns
categorical_cols = ['time_segment', 'factor_v1']
numerical_cols = ['total_cars', 'year', 'month', 'day_of_week', 'total_injured', 'total_killed',]
label_col = 'borough'

# Step 1: Index + encode categorical features
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid='keep') for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_ohe") for col in categorical_cols]

# Step 2: Index label
label_indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid='keep')

# Step 3: Assemble features
feature_cols = [col + "_ohe" for col in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembler.setHandleInvalid("skip")

# COMMAND ----------

# Initialize model and piepline
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler, rf])

# Split data into train and test datasets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Fit model to train data
model = pipeline.fit(train_data)

# Test model on test data
predictions = model.transform(test_data)

# COMMAND ----------

# Evaluate predictions and print accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", 
                    predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.3f}")

predictions.select("borough", "prediction").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 3 - Predict Main Factor Contributing to Car Crash

# COMMAND ----------

# Import pyspark functions 
from pyspark.sql.functions import coalesce, col

df = df.withColumn("main_reason", coalesce(col("factor_v1"), col("factor_v2"), col("factor_v3"))) # takes first factor and second or third if unavailable
display(df)

# COMMAND ----------

# Import pyspark functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Select feature columns
categorical_cols = ['borough', 'zip_code', 'year', 'month', 'day_of_week', 'time_segment']
numerical_cols = ['total_injured', 'total_killed', 'fatality_rate', 'total_cars']
label_col = 'main_reason'

# Step 1: Index + encode categorical features
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid='keep') for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_ohe") for col in categorical_cols]

# Step 2: Index label
label_indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid='keep')

# Step 3: Assemble features
feature_cols = [col + "_ohe" for col in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembler.setHandleInvalid("skip")

# COMMAND ----------

# Initialize model and pipeline
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler, rf])

# Split data into train and test data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Fit model to train data
model = pipeline.fit(train_data)

# Test model on test data
predictions = model.transform(test_data)

# COMMAND ----------

# Evaluate model and print accuracy metric
evaluator = MulticlassClassificationEvaluator(labelCol="label", 
                    predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.3f}")

predictions.select("main_reason", "prediction").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 4 - Predict Total Injured Persons in Crash

# COMMAND ----------

# Import pyspark packages
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# List of categorical columns to index and encode
categorical_columns = ["borough", "time_segment","day_of_week"]

# List of numerical columns to use directly
numerical_columns = ["total_cars", "total_killed", "fatality_rate", "year"]

# Step 1: Apply StringIndexer to convert categorical columns to indices
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_columns]

# Optionally, apply OneHotEncoder to handle one-hot encoding for categorical features
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_onehot") for col in categorical_columns]

# Step 2: Assemble all features (indexed and numerical) into a single vector
assembler = VectorAssembler(inputCols=numerical_columns + [col + "_onehot" for col in categorical_columns], 
                            outputCol="features")

# Step 3: Train RandomForestModel
rf = RandomForestRegressor(featuresCol="features", labelCol="total_injured", maxDepth=5, maxBins=10)

# Step 4: Create a Pipeline to apply transformations and train the model
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

# Fit the model
model = pipeline.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Model performance (RMSE and R²)
from pyspark.ml.evaluation import RegressionEvaluator
evaluator_rmse = RegressionEvaluator(labelCol="total_injured", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="total_injured", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"RMSE for the Random Forest Model: {rmse}")
print(f"R² for the Random Forest Model: {r2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 5 - Predict Total Cars Involved in Crash

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Categorical and numerical columns
categorical_columns = ["borough", "time_segment", "day_of_week"]
numerical_columns = ["total_killed", "total_injured", "fatality_rate", "year"]  # total_killed now a feature

# Index and encode categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_columns]
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_onehot") for col in categorical_columns]

# Assemble features
assembler = VectorAssembler(
    inputCols=numerical_columns + [col + "_onehot" for col in categorical_columns],
    outputCol="features"
)

# Decision Tree Regressor with total_cars as the label
dt = DecisionTreeRegressor(featuresCol="features", labelCol="total_cars", maxDepth=5, maxBins=10)

# Build pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, dt])

# Fit model
model = pipeline.fit(train_data)

# Predict
predictions = model.transform(test_data)

# Evaluate
from pyspark.ml.evaluation import RegressionEvaluator

evaluator_rmse = RegressionEvaluator(labelCol="total_cars", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="total_cars", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"RMSE for the Decision Tree Model: {rmse}")
print(f"R² for the Decision Tree Model: {r2}")