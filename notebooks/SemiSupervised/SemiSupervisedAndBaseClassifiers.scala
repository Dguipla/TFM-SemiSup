// Databricks notebook source
// DBTITLE 1,Importing all the libraries for semisupervised 
import org.apache.spark.ml.semisupervised.SelfTraining
import org.apache.spark.ml.semisupervised.CoTraining
import org.apache.spark.ml.semisupervised.Supervised
import org.apache.spark.ml.semisupervised.UnlabeledTransformer
import org.apache.spark.ml.semisupervised.FunctionsSemiSupervised._
import org.apache.spark.ml.semisupervised.SemiSupervisedDataResults

// COMMAND ----------

// MAGIC %md
// MAGIC ## Comparing Spark ML with the Keel results using Semisupervised algorithms (numerical features)
// MAGIC 
// MAGIC ##### - sonar
// MAGIC ##### - banana
// MAGIC ##### - heart
// MAGIC ##### - coil2000
// MAGIC ##### - magic
// MAGIC ##### - spectfhear
// MAGIC ##### - wisconsin
// MAGIC ##### - titanic

// COMMAND ----------

// DBTITLE 1,Comparing - Data Processing & Data Preparation (Featurization) 
import org.apache.spark.sql.types.{StructType,StructField,StringType,DoubleType}
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}
import scala.util.control.Breaks._
import org.apache.spark.ml.PipelineStage
// clasificadores Base
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.LinearSVC


// data sets
val data = Array("coil2000.csv","sonar.csv","spectfheart-1.csv","heart.csv","wisconsinKeel-1.csv","magic.csv","titanic.csv","banana.csv")//Array("coil2000.csv")//Array("coil2000.csv","sonar.csv","spectfheart-1.csv","heart.csv","wisconsinKeel-1.csv")
val featurizationPipeline:Array[Pipeline] = new Array[Pipeline](data.size) 
val dataDF:Array[DataFrame] = new Array[DataFrame](data.size) 

for (posPipeline <- 0 to (data.size-1)){
  //reading
  var PATH="dbfs:/FileStore/tables/"
  dataDF(posPipeline) = spark.read.format("csv")
    .option("sep", ",")
    .option("inferSchema", "true")
    .option("header", "true")
    .load(PATH + data(posPipeline))
  dataDF(posPipeline) = dataDF(posPipeline).na.drop()

  //featurization
  var dataFeatures=dataDF(posPipeline).columns.diff(Array(dataDF(posPipeline).columns.last))
  var dataFeaturesLabelPipeline= new VectorAssembler().setOutputCol("features").setInputCols(dataFeatures)

  // StringIndexer para pasar el valor categorico a double de la clase , para la features no utilizamos pq ya son doubles. 
  var indexClassPipeline = new StringIndexer().setInputCol(dataDF(posPipeline).columns.last).setOutputCol("label").setHandleInvalid("skip")

  //pipelineFeaturization
  featurizationPipeline(posPipeline) = new Pipeline().setStages(Array(
                                                dataFeaturesLabelPipeline,
                                                indexClassPipeline))
} 

// COMMAND ----------

// DBTITLE 1,Class/Label Balance for - sonar - banana - heart - coil2000 - magic - spectfhear - wisconsin - titanic
//label balance
val dataCode =  data //Array("coil2000","sonar","spectfheart","heart","wisconsin")
var balanceLableDataSet:Array[DataFrame] = new Array[DataFrame](dataCode.size)
var balanceLableDataSetPercentage:Array[(String,Double)] = new Array[(String,Double)](dataCode.size)
println ("Label/Class balance")
for (posDataSet <- 0 to (dataCode.size-1)){
   balanceLableDataSet(posDataSet) = featurizationPipeline(posDataSet).fit(dataDF(posDataSet)).transform(dataDF(posDataSet))
   balanceLableDataSetPercentage(posDataSet) = (dataCode(posDataSet), 
                                               (balanceLableDataSet(posDataSet).filter(balanceLableDataSet(posDataSet)("label")===0).count).toDouble / (balanceLableDataSet(posDataSet).select("label").count).toDouble)
  
  println ("dataset, label 0 (%), label 1 (%)")
  print(balanceLableDataSetPercentage(posDataSet)._1)
  print(" , ")
  print(balanceLableDataSetPercentage(posDataSet)._2)
  print(" , ")
  println(1-balanceLableDataSetPercentage(posDataSet)._2)

}


// COMMAND ----------

// DBTITLE 1,Comparing Supervised
//base clasifiers 
val instanciaTrainingPipelineDT = new Supervised(new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineRF = new Supervised(new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineNB = new Supervised(new NaiveBayes().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineLR = new Supervised(new LogisticRegression().setFeaturesCol("features").setLabelCol("label"))

//Array of instances
val arrayClassifiers:Array[(String,PipelineStage)] = Array(("DT-Sark",instanciaTrainingPipelineDT),
                                       ("LR-Sark",instanciaTrainingPipelineLR),
                                       ("RF-Sark",instanciaTrainingPipelineRF),
                                       ("NB-Sark",instanciaTrainingPipelineNB)
                                      )

//parameters
val classifierBase = arrayClassifiers.map(cls =>cls._1)
val percentageLabeled = Array(0.05,0.1,0.15,0.2,0.3,0.6)
val threshold= Array(0.0) // Supervised is n.a 0
val criterion = Array("n.a") // Supervised is n.a
val dataCode =  Array("coil2000","sonar","spectfheart","heart","wisconsin")
var resultsSupervised:Array[DataFrame] = new Array[DataFrame](dataCode.size) 

//calculation
for (posDataSet <- 0 to (dataCode.size-1)){
  
  //template dataFrame results according the parameters
  var resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode(posDataSet),classifierBase, percentageLabeled)
  
  // final pipeline models with all the configurations (parameters)
  var modelsPipeline = criterion.map(crit=>(crit,percentageLabeled.map(per=>(per,threshold.map(th=>(th,arrayClassifiers.map(clasi=>(clasi._1,new Pipeline().setStages(Array(clasi._2))))))))))
  
  // dataframe of final results
  resultsSupervised(posDataSet) = SupervisedAndSemiSupervisedResuts (featurizationPipeline(posDataSet), 4,dataDF(posDataSet),modelsPipeline,resultsInfo)
}

//display(results)


// COMMAND ----------

// DBTITLE 1,COMPARING - SelfTraining
// base classifiers
val instTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")

//Array of instances
val arrayClassifiers:Array[(String,PipelineStage)] = Array(("ST-DT-Spark",instTrainingPipelineDT),
                                      ("ST-LR-Spark",instTrainingPipelineLR),
                                      ("ST-RF-Spark",instTrainingPipelineRF),
                                      ("ST-NB-Spark",instTrainingPipelineNB)
                                      )


// results for Semisupervised
var SemiSupervisedData = new SemiSupervisedDataResults ()

//parameters
val classifierBase = arrayClassifiers.map(cls =>cls._1)
val percentageLabeled = Array(0.05,0.15,0.3,0.6)//Array(0.05,0.1,0.15,0.2,0.3,0.6)//Array(0.1,0.2,0.3,0.4) 
val threshold=Array(0.4,0.5,0.6)//Array(0.7,0.8,0.9,0.95)//Array(0.7,0.8,0.9,0.95)//Array(0.9) //Array(0.7,0.8,0.9)
val kBest= Array(0.0)
val maxIter = 5
val criterion = Array("threshold")
val dataCode = Array("coil2000","sonar","spectfheart","heart","wisconsin")//Array("sonar","spectfheart","heart","wisconsin")
//Array("coil2000")//Array("coil2000","sonar","spectfheart","heart","wisconsin")//Array("titanic","coil2000","sonar","spectfheart","heart","banana","wisconsin","magic")
var resultsSelTraining:Array[DataFrame] = new Array[DataFrame](dataCode.size) 

//calculation
for (posDataSet <- 0 to (dataCode.size-1)){

  //template dataFrame results according the parameters
  var resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode(posDataSet),classifierBase, percentageLabeled,threshold,kBest,criterion)

  // final pipeline models with all the configurations (parameters)
  var modelsPipeline = pipelineModelsSelfTraining(threshold,kBest,percentageLabeled,SemiSupervisedData,arrayClassifiers,criterion,maxIter)

  // dataframe of final results
  resultsSelTraining(posDataSet) = SupervisedAndSemiSupervisedResuts (featurizationPipeline(posDataSet), 4,dataDF(posDataSet),modelsPipeline,resultsInfo,SemiSupervisedData)
}

//display(resultsSelTraining(0))

// COMMAND ----------

// DBTITLE 1,COMPARING - CoTraining
// base classifiers
val instTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")

//Array of instances
val arrayClassifiers:Array[(String,PipelineStage)] = Array(("CT-DT-Spark",instTrainingPipelineDT),
                                      ("CT-LR-Spark",instTrainingPipelineLR),
                                      ("CT-RF-Spark",instTrainingPipelineRF),
                                      ("CT-NB-Spark",instTrainingPipelineNB)
                                      )
// results for Semisupervised
var SemiSupervisedData = new SemiSupervisedDataResults ()

//parameters
val classifierBase = arrayClassifiers.map(cls =>cls._1)
val percentageLabeled = Array(0.05,0.15,0.3,0.6)//Array(0.05,0.1,0.15,0.2,0.3,0.6)//Array(0.1,0.2,0.3,0.4) 
val threshold=Array(0.4,0.5,0.6)//Array(0.7,0.8,0.9,0.95)//Array(0.7,0.8,0.9,0.95)//Array(0.9) //Array(0.7,0.8,0.9)
val kBest= Array(0.0)
val maxIter = 5
val criterion = Array("threshold")
val dataCode = Array("coil2000","sonar","spectfheart","heart","wisconsin")
//Array("coil2000")//Array("coil2000","sonar","spectfheart","heart","wisconsin")//Array("titanic","coil2000","sonar","spectfheart","heart","banana","wisconsin","magic")
var resultsCoTraining:Array[DataFrame] = new Array[DataFrame](dataCode.size) 

//calculation
for (posDataSet <- 0 to (dataCode.size-1)){

  //template dataFrame results according the parameters
  var resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode(posDataSet),classifierBase, percentageLabeled,threshold,kBest,criterion)

  // final pipeline models with all the configurations (parameters)
  var modelsPipeline = pipelineModelsCoTraining(threshold,kBest,percentageLabeled,SemiSupervisedData,arrayClassifiers,criterion,maxIter)

  // dataframe of final results
  resultsCoTraining(posDataSet) = SupervisedAndSemiSupervisedResuts (featurizationPipeline(posDataSet), 4,dataDF(posDataSet),modelsPipeline,resultsInfo,SemiSupervisedData)
}

//display(results)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Analizing different datasets for BIG DATA
// MAGIC ##### **Supervised** with data label reductions and **Semisupervisado** for Big data 
// MAGIC ##### - ADULT
// MAGIC ##### - POKER
// MAGIC ##### - TAXI NY

// COMMAND ----------

// DBTITLE 1, Data Processing & Data Preparation (Featurization) - ADULT
// LIBRERIAS necesarias (IMPORTACIONES)

import org.apache.spark.sql.types.{StructType,StructField,StringType,DoubleType}
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}
import scala.util.control.Breaks._

// clasificadores Base
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.LinearSVC


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// LECTURA
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//leemos el documento
val PATH="dbfs:/FileStore/tables/"
val datos="adult.data"


val datosDF = spark.read.format("csv")
  .option("sep", ",")
  .option("inferSchema", "true")
  .option("header", "false")
  .load(PATH + datos)
datosDF.printSchema() 


val DATA_training="adult.data"
val lines_training= sc.textFile(PATH + DATA_training)




//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//Cleaning
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// training
val nonEmpty_training= lines_training.filter(_.nonEmpty).filter(y => ! y.contains("?")) // Dado que representa en el peor de los casos un 3-4% approx, lo eliminamos
val parsed_training= nonEmpty_training.map(line => line.split(","))


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//Schema
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// generamos el esquema de los datos
val adultSchema= StructType(Array(
  StructField("age",StringType,true),  // continuos
  StructField("workclass",StringType,true),
  StructField("fnlwgt",StringType,true), // continuos
  StructField("education",StringType,true),
  StructField("education_num",StringType,true),// continuos
  StructField("marital_status",StringType,true),
  StructField("occupation",StringType,true),
  StructField("race",StringType,true),
   StructField("relationship",StringType,true),
  StructField("sex",StringType,true),
  StructField("capital_gain",StringType,true), // continuos
  StructField("capital_loss",StringType,true),// continuos
  StructField("hours_per_week",StringType,true), // continuos
  StructField("native_country",StringType,true),
  StructField("clase",StringType,true)
))


//creamos los data sets con la informacion del esquema antes generado

val income_trainingDF= spark.createDataFrame(parsed_training.map(Row.fromSeq(_)), adultSchema)

// convertimos los valores continuos a double los datos de trainning
val income_trainningDF_converted = income_trainingDF.withColumn("fnlwgt2", 'fnlwgt.cast("Double")).select('age, 'workclass, 'fnlwgt2 as 'fnlwgt,'education, 'education_num,'marital_status,'occupation,'relationship,'race,'sex,'capital_gain,'capital_loss,'hours_per_week,'native_country,'clase).withColumn("age2", 'age.cast("Double")).select('age2 as 'age, 'workclass,'fnlwgt,'education,  'education_num,'marital_status,'occupation,'relationship,'race,'sex,'capital_gain,'capital_loss,'hours_per_week,'native_country,'clase).
withColumn("education_num2", 'education_num.cast("Double")).select('age, 'workclass,'fnlwgt,'education,'education_num2 as 'education_num,'marital_status,'occupation,'relationship,'race,'sex,'capital_gain,'capital_loss,'hours_per_week,'native_country,'clase).
withColumn("capital_gain2", 'capital_gain.cast("Double")).select('age, 'workclass,'fnlwgt,'education, 'education_num,'marital_status,'occupation,'relationship,'race,'sex,'capital_gain2 as 'capital_gain,'capital_loss,'hours_per_week,'native_country,'clase).
withColumn("capital_loss2", 'capital_loss.cast("Double")).select('age, 'workclass,'fnlwgt,'education, 'education_num,'marital_status,'occupation,'relationship,'race,'sex,'capital_gain,'capital_loss2 as 'capital_loss,'hours_per_week,'native_country,'clase).
withColumn("hours_per_week2", 'hours_per_week.cast("Double")).select('age, 'workclass,'fnlwgt,'education, 'education_num,'marital_status,'occupation,'relationship,'race,'sex,'capital_gain,'capital_loss,'hours_per_week2 as 'hours_per_week,'native_country,'clase)



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//class distribution
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

println("DISTRIBUCION DE CLASE")


//verificamos la cantidad de lineas / elementos que hay para ver si se adapta a lo que dice practica 
val count_lines_training = lines_training.count() // 32k approx

println("NUMERO DE REGISTROS")
println("Numero de registros de entrenamiento: " + count_lines_training)

// distribuvion de clase:

val distribucionClase_Menor_igual_50 = (income_trainingDF.filter(income_trainingDF("clase")===" <=50K").count).toDouble / (income_trainingDF.select("clase").count).toDouble
val distribucionClase_Mayor_50k =(income_trainingDF.filter(income_trainingDF("clase")===" >50K").count).toDouble / (income_trainingDF.select("clase").count).toDouble


println("Distribucion de clase mayor a de 50k: " + distribucionClase_Mayor_50k)
println ("Distribucion de clase menor o igual a 50k:" +distribucionClase_Menor_igual_50)
// la distribucion nos indica una complejidad aceptable tiene una relacion de 25%  - 75%

val distribucionAdult = Array(distribucionClase_Menor_igual_50,distribucionClase_Mayor_50k)
val diferentesClases = income_trainingDF.select("clase").distinct.count()


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//featurization
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// StringIndexer para pasar el valor categorico a double de la clase 
val indiceClasePipelineAdult = new StringIndexer().setInputCol("clase").setOutputCol("label").setHandleInvalid("skip")

// valores a convertir a continuos
val valoresCategoricosFeatures= Array("workclass","education","marital_status", "occupation", "race", "relationship","sex","native_country")
//StringIndexer para la features
val indexStringFeaturesLlamada = indexStringColumnsStagePipeline(datosDFLabeled_trainningAdult,valoresCategoricosFeatures)

val indexStringFeaturesTodasNumAdult = indexStringFeaturesLlamada._1
val columnnasYaNumericasAdult = indexStringFeaturesLlamada._2

// juntamos las columnas convertidas a continuo con StringIndexer y las ya numericas/continuas
val datosDFSinClaseSoloFeaturesAdult = valoresCategoricosFeatures.par.map(x=>x+"-num").toArray.union(columnnasYaNumericasAdult)

//VectorAssembler para generar una array de valores para las features
val assemblerFeaturesLabelPipelineAdult= new VectorAssembler().setOutputCol("features").setInputCols(datosDFSinClaseSoloFeaturesAdult.diff(Array("clase"))) // sin la clase o label



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//pipeline featurization
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//generamos el pipeline poniendo correctamente el orden

val featurizationPipelineAdult = new Pipeline().setStages(Array(indexStringFeaturesTodasNumAdult,
                                                                assemblerFeaturesLabelPipelineAdult,
                                                                indiceClasePipelineAdult))






// COMMAND ----------

// DBTITLE 1,Supervised - ADULT  (DT, LR, RF,NB)
// base classifiers

val instanciaTrainingPipelineDT = new Supervised(new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label").setMaxBins(42))
val instanciaTrainingPipelineRF = new Supervised(new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label").setMaxBins(42))
val instanciaTrainingPipelineNB = new Supervised(new NaiveBayes().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineLR = new Supervised(new LogisticRegression().setFeaturesCol("features").setLabelCol("label"))


// intances of baseClasifiers
val arrayClassifiers_Supervised:Array[(String,PipelineStage)] = Array(("DT",instanciaTrainingPipelineDT)/*,
                                       ("LR",instanciaTrainingPipelineLR),
                                       ("RF",instanciaTrainingPipelineRF),
                                       ("NB",instanciaTrainingPipelineNB)*/
                                       //("LSVM",instanciaTrainingPipelineLSVM) 
                                      )



val threshold= Array(0.0) // Supervised is n.a 0
val criterion = Array("n.a") // Supervised is n.a

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//parameters:  
// 1.- classifiers, 
// 2.- percentage of data reduction
// 3.-threshold --> 0 is not applicable for Supervised
// 3.- type of data ie: "BCW"
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


val classifierBase = arrayClassifiers_Supervised.map(cls =>cls._1)
val percentageLabeled = Array(0.01)//Array(0.01,0.05,0.08,0.10,0.20,0.30)
val dataCode = "ADULT"


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Dataframe template for results
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,
                                          classifierBase,
                                          percentageLabeled 
                                         )


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// pipelines model creations for all the percentatge, baseClasifiers...
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


val modelsPipeline = criterion.map(crit=>(crit,percentageLabeled.map(per=>(per,threshold.map(th=>(th,arrayClassifiers_Supervised.map(clasi=>(clasi._1,new Pipeline().setStages(Array(clasi._2))))))))))
//percentageLabeled.map(per=>(per,threshold.map(th=>(th,arrayClassifiers_Supervised.map(clasi=>(clasi._1,new Pipeline().setStages(Array(clasi._2))))))))

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// final results dataframe
//+++++++++++++++++++++++++++++++++++++++++++++++

val resultsAdultSupervised = SupervisedAndSemiSupervisedResuts(featurizationPipelineAdult, 4,income_trainningDF_converted,modelsPipeline,resultsInfo)

display(resultsAdultSupervised)


// COMMAND ----------

// DBTITLE 1,SelfTrainning - ADULT (DT, LR, RF ....)

val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label").setMaxBins(42)
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label").setMaxBins(42)
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


val arrayClassifiers:Array[(String,PipelineStage)] = Array(("ST-DT",instanciaTrainingPipelineDT)
                                      //("ST-LR",instanciaTrainingPipelineRF ),
                                       //("ST-RF",instanciaTrainingPipelineNB ),
                                       //("ST-NB",instanciaTrainingPipelineLR)
                                      )

// results for Semisupervised
var SemiSupervisedData = new SemiSupervisedDataResults ()

//parameters
val classifierBase = arrayClassifiers.map(cls =>cls._1)
val percentageLabeled = Array(0.01)//Array(0.01,0.05,0.08,0.10,0.20)
val threshold= Array(0.8)//Array(0.8,0.9,0.95)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val kBest= Array(0.1)//Array(0.05,0.1,0.2,0.3,0.8)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95) //percentage per 1 of unlabeled total data that should be labeled at the end
val dataCode = "ADULT"
val criterion = Array("threshold")//Array("threshold","kBest")

//template dataFrame results according the parameters
val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,classifierBase, percentageLabeled,threshold,kBest,criterion)

// final pipeline models with all the configurations (parameters)
val modelsPipeline = pipelineModelsSelfTraining(threshold,kBest,percentageLabeled,SemiSupervisedData,arrayClassifiers,criterion)

// dataframe of final results
val resultsAdultST = SupervisedAndSemiSupervisedResuts (featurizationPipelineAdult, 4,income_trainningDF_converted,modelsPipeline,resultsInfo,SemiSupervisedData)

display(resultsAdultST)

// COMMAND ----------

// DBTITLE 1,Data Processing & Data Preparation (Featurization) - POKER
// LIBRERIAS necesarias (IMPORTACIONES)

import org.apache.spark.sql.types.{StructType,StructField,StringType,DoubleType}
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}
import scala.util.control.Breaks._

// clasificadores Base
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.LinearSVC


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// reading
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val PATH="dbfs:/FileStore/tables/"
val datos="pokerTraining.data"
val lines_datos= sc.textFile(PATH + datos)//.map(x=>x.split(","))

// creamos el DF
val datosDF = spark.read.format("csv")
  .option("sep", ",")
  .option("inferSchema", "true")
  .option("header", "false")
  .load(PATH + datos)

datosDF.printSchema() 




//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//cleaning
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//vamos a ver que valores nulos tenemos
val instanciasConNulos=datosDF.count() - datosDF.na.drop().count()
println("INSTANCIAS CON NULOS")
println(instanciasConNulos) //no hay nulos


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//class distribution
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

println("DISTRIBUCION DE CLASE")

val clasesTipo = datosDF.select("_c10").distinct.count.toInt
val distribuciones:Array[(Double)] = new Array[(Double)](clasesTipo) 

for(cls <- 0 to (clasesTipo-1)){
  distribuciones(cls)= ((datosDF.filter(datosDF("_c10")=== cls).count).toDouble / (datosDF.select("_c10").count).toDouble)*100
  print ("label distribucion (%) ") 
  print (cls)
  print (": ")
  println (distribuciones(cls))
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//transformation
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val datosDFNew=datosDF.withColumn("clase",
                                  when(((datosDF("_c10"))>0) ,"AtleastOnePair").
                                  when(((datosDF("_c10"))<=0) ,"Nothing")).drop("_c10")


println("Nueva distribucion Binaria")
println("Muy pocas opciones para ganar (Ni parejas)")
println (distribuciones(0))
println("Alguna opcion para ganar(Almenos pareja)")
println (100-(distribuciones(0)))



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//featurization
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// StringIndexer para pasar el valor categorico a double de la clase , para la features no utilizamos pq ya son doubles. 
val indiceClasePipelinePoker = new StringIndexer().setInputCol("clase").setOutputCol("label").setHandleInvalid("skip")

// valores a convertir a continuos
val valoresCategoricosFeatures= Array("_c0","_c1","_c2","_c3","_c4","_c5","_c6","_c7","_c8","_c9")
//StringIndexer para la features

val indexStringFeaturesLlamada = indexStringColumnsStagePipeline(datosDFNew,valoresCategoricosFeatures)

val indexStringFeaturesTodasNumPoker = indexStringFeaturesLlamada._1
val columnnasYaNumericasPoker = indexStringFeaturesLlamada._2

// juntamos las columnas convertidas a continuo con StringIndexer y las ya numericas/continuas
val datosDFSinClaseSoloFeaturesPoker = valoresCategoricosFeatures.par.map(x=>x+"-num").toArray.union(columnnasYaNumericasPoker)

//VectorAssembler para generar una array de valores para las features
val assemblerFeaturesLabelPipelinePoker= new VectorAssembler().setOutputCol("features").setInputCols(datosDFSinClaseSoloFeaturesPoker.diff(Array("clase"))) // sin la clase o label


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//pipeline
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//generamos el pipeline
val featurizationPipelinePoker = new Pipeline().setStages(Array(
                                              indexStringFeaturesTodasNumPoker,
                                              assemblerFeaturesLabelPipelinePoker,
                                              indiceClasePipelinePoker))






// COMMAND ----------

// DBTITLE 1,Supervised - POKER (DT, LR, RF,NB) 
// base classifiers

val instanciaTrainingPipelineDT = new Supervised(new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineRF = new Supervised(new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineNB = new Supervised(new NaiveBayes().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineLR = new Supervised(new LogisticRegression().setFeaturesCol("features").setLabelCol("label"))


// intances of baseClasifiers
val arrayClassifiers_Supervised:Array[(String,PipelineStage)] = Array(("DT-Spark",instanciaTrainingPipelineDT),
                                       ("LR-Spark",instanciaTrainingPipelineLR),
                                       ("RF-Spark",instanciaTrainingPipelineRF),
                                       ("NB-Spark",instanciaTrainingPipelineNB)
                                       //("LSVM",instanciaTrainingPipelineLSVM) 
                                      )



val threshold= Array(0.0) // Supervised is n.a 0
val criterion = Array("n.a") // Supervised is n.a

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//parameters:  
// 1.- classifiers, 
// 2.- percentage of data reduction
// 3.-threshold --> 0 is not applicable for Supervised
// 3.- type of data ie: "BCW"
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


val classifierBase = arrayClassifiers_Supervised.map(cls =>cls._1)
val percentageLabeled =Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val dataCode = "POKER"


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Dataframe template for results
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,
                                          classifierBase,
                                          percentageLabeled 
                                         )


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// pipelines model creations for all the percentatge, baseClasifiers...
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


val modelsPipeline = criterion.map(crit=>(crit,percentageLabeled.map(per=>(per,threshold.map(th=>(th,arrayClassifiers_Supervised.map(clasi=>(clasi._1,new Pipeline().setStages(Array(clasi._2))))))))))
//percentageLabeled.map(per=>(per,threshold.map(th=>(th,arrayClassifiers_Supervised.map(clasi=>(clasi._1,new Pipeline().setStages(Array(clasi._2))))))))

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// final results dataframe
//+++++++++++++++++++++++++++++++++++++++++++++++

val resultsPokerSupervised = SupervisedAndSemiSupervisedResuts(featurizationPipelinePoker, 4,datosDFNew,modelsPipeline,resultsInfo)

//display(resultsPokerSupervised)


// COMMAND ----------

// DBTITLE 1,SelfTrainning - POKER (DT, LR, RF ....)

val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


val arrayClassifiers:Array[(String,PipelineStage)] = Array(("ST-DT-Spark",instanciaTrainingPipelineDT),
                                      ("ST-RF-Spark",instanciaTrainingPipelineRF ),
                                      ("ST-NB-Spark",instanciaTrainingPipelineNB ),
                                      ("ST-LR-Spark",instanciaTrainingPipelineLR)
                                      )

// results for Semisupervised
var SemiSupervisedData = new SemiSupervisedDataResults ()

//parameters
val classifierBase = arrayClassifiers.map(cls =>cls._1)
val percentageLabeled = Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val threshold= Array(0.7,0.8,0.95)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val kBest= Array(0.1)//Array(0.05,0.1,0.2,0.3,0.8)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95) //percentage per 1 of unlabeled total data that should be labeled at the end
val dataCode = "POKER"
val criterion = Array("threshold")//Array("threshold","kBest")

//template dataFrame results according the parameters
val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,classifierBase, percentageLabeled,threshold,kBest,criterion)

// final pipeline models with all the configurations (parameters)
val modelsPipeline = pipelineModelsSelfTraining(threshold,kBest,percentageLabeled,SemiSupervisedData,arrayClassifiers,criterion)

// dataframe of final results
val resultsPokerST = SupervisedAndSemiSupervisedResuts (featurizationPipelinePoker, 4,datosDFNew,modelsPipeline,resultsInfo,SemiSupervisedData)

//display(resultsPokerST)

// COMMAND ----------

// DBTITLE 1,CoTraining - Poker (DT,LR...)

val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


val arrayClassifiers:Array[(String,PipelineStage)] = Array(("CT-DT-Spark",instanciaTrainingPipelineDT),
                                      ("CT-RF-Spark",instanciaTrainingPipelineRF ),
                                      ("CT-NB-Spark",instanciaTrainingPipelineNB ),
                                      ("CT-LR-Spark",instanciaTrainingPipelineLR)
                                      )

// results for Semisupervised
var SemiSupervisedData = new SemiSupervisedDataResults ()

//parameters
val classifierBase = arrayClassifiers.map(cls =>cls._1)
val percentageLabeled = Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val threshold= Array(0.7,0.8,0.95)//Array(0.9)//Array(0.8,0.9,0.95)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val kBest= Array(0.1)//Array(0.05,0.1,0.2,0.3,0.8)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95) //percentage per 1 of unlabeled total data that should be labeled at the end
val dataCode = "POKER"
val criterion = Array("threshold")//Array("threshold","kBest")

//template dataFrame results according the parameters
val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,classifierBase, percentageLabeled,threshold,kBest,criterion)

// final pipeline models with all the configurations (parameters)
val modelsPipeline = pipelineModelsCoTraining(threshold,kBest,percentageLabeled,SemiSupervisedData,arrayClassifiers,criterion)

// dataframe of final results
val resultsPoker_CT = SupervisedAndSemiSupervisedResuts (featurizationPipelinePoker, 4,datosDFNew,modelsPipeline,resultsInfo,SemiSupervisedData)


// COMMAND ----------

// DBTITLE 1,Data Processing & Data Preparation (Featurization) - TAXI NY
import scala.math.sqrt
import scala.math.pow
import scala.math.toRadians
import scala.math.sin
import scala.math.cos
import scala.math.atan2

// LIBRERIAS necesarias (IMPORTACIONES)

import org.apache.spark.sql.types.{StructType,StructField,StringType,DoubleType}
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}
import scala.util.control.Breaks._
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.functions._

// clasificadores Base
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.LinearSVC

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// reading
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import org.apache.spark.sql.functions.{concat, lit}
val PATH="/FileStore/tables/"  
val entreno = "train.csv"
val training = sc.textFile(PATH + entreno)

// schema
case class fields   (id: String, // 0
                     vendor_id: Int, // categorico 1
                     pickup_datetime: String, //2
                     dropoff_datetime:String, //3
                     passenger_count:Int, // categorico 4
                     pickup_longitude:Double, // 5
                     pickup_latitude:Double, // 6
                     dropoff_longitude:Double, // 7
                     dropoff_latitude:Double, //8
                     store_and_fwd_flag:String, // categorico 9
                     trip_duration:Int) //10




//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//cleaning
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val nonEmpty_training= training.filter(_.nonEmpty)
// Separamos por , y eliminamos la primera linea head
val parsed_training= nonEmpty_training.map(line => line.split(",")).zipWithIndex().filter(_._2 >= 1).keys
// Asociamos los campos a la clase
val training_reg = parsed_training.map(x=>fields(x(0),
                                                 x(1).toInt,x(2),x(3),x(4).toInt,x(5).toDouble,x(6).toDouble,x(7).toDouble,x(8).toDouble,x(9),x(10).toInt)) 



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//class distribution
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val longTravel = training_reg.filter(x => x.trip_duration > 900).count
val shortTravel = training_reg.filter(x => x.trip_duration <= 900).count


val longTravelPorcentage = (longTravel.toDouble/training_reg.count())*100
val shortTravelPorcentage = (shortTravel.toDouble/training_reg.count())*100

println("Porcentaje de viajes cortos: " + shortTravelPorcentage)
println("Porcentaje de viajes largos: " + longTravelPorcentage)

// tenemos una distribucion 67% - 33% aprox con lo que la complejidad del clasificador no sera muy elevada.



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//Transformation
// (id, vendor_id, prickup_month, pickup_day, pickup_time, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude,
//  dropoff_latitude, store_and_fwd_flag, diff_distance, trip_duration)
//*****************************************************************************


// para ello hemos eliminado atributos que no nos generan valor:
// - año 
// - tiempo de llegada esto esta informacion esta en el tiempo del trayecto
// - de momento hemos selecionado  passenger count  
// - Por otro lado podriamos eliminar id ya que hay uno por registro no aporta informacion


case class fieldsAfterExpo   (id: String, //0
                     vendor_id: Int, //0
                     pickup_month: String,
                     pickup_day: Int,
                     pickup_time: Double,
                     passenger_count: Int, 
                     pickup_longitude: Double, 
                     pickup_latitude: Double, 
                     dropoff_longitude: Double, 
                     dropoff_latitude: Double, 
                     store_and_fwd_flag: String, 
                     diff_distance: Double,
                     trip_duration: Int) 

val training_reg_final = parsed_training.map(x => fieldsAfterExpo(
                                            x(0), 
                                            x(1).toInt,
                                            {val splittedmonth = x(2).split(" ")  // mes
                                            val splittedmonth2 = splittedmonth(0).split("-")
                                            (splittedmonth2(1))
                                            },
                                            {val splittedday = x(2).split(" ") // day
                                            val splittedday2 = splittedday(0).split("-")
                                            (splittedday2(2))
                                            }.toInt,
                                            {val splittedtime = x(2).split(" ") // hora
                                            val splittedtime2 = splittedtime(1).split(":")
                                            splittedtime2(0).toDouble + splittedtime2(1).toDouble/60 + splittedtime2(2).toDouble/3600
                                            },
                                           x(4).toInt,
                                           x(5).toDouble,
                                           x(6).toDouble,
                                           x(7).toDouble,
                                           x(8).toDouble,
                                           x(9),
                                           6371*2*atan2(sqrt(
                                                       pow(sin(toRadians(x(8).toDouble - x(6).toDouble)/2),2) +
                                                       cos(toRadians(x(6).toDouble)) * 
                                                       cos(toRadians(x(8).toDouble)) *
                                                       pow(sin(toRadians(x(7).toDouble - x(5).toDouble)/2),2)
                                                       //pow(((x.pickup_longitude).abs - (x.dropoff_longitude).abs),2)
                                                       ),
                                                       sqrt(1-(
                                                       pow(sin(toRadians(x(8).toDouble - x(6).toDouble)/2),2) +
                                                       cos(toRadians(x(6).toDouble)) * cos(toRadians(x(8).toDouble))*
                                                       pow(sin(toRadians(x(7).toDouble - x(5).toDouble)/2),2)
                                                       //pow(((x.pickup_longitude).abs - (x.dropoff_longitude).abs),2)
                                                       )
                                                      )
                                                     ),
                                           x(10).toInt)

                                        )
//outliers o valores incorrectos (ruido) o  poco representativos  POSIBLE LIMPIEZA: 
val casosDistanciaMayor30k=training_reg_final.filter(x=>x.diff_distance>30).count //573 casos --> 0.03%
val casosDistanciaMenor300m=training_reg_final.filter(x=>x.diff_distance<0.3).count //22469 casos --> 1.5%
val casosTiempoMayor2h = training_reg_final.filter(x=>x.trip_duration>7200).count//2253 casos --> 0.154%
//Filtramos recoridos mayores de 30k son
//Filtramos recoridos menores a 300m
//Filtramos recoridos tiempos mayores a 2h


//convertimos a DF
val training_reg_final_DF=training_reg_final.toDF
//Filtramos recoridos mayores de 30k son
//Filtramos recoridos menores a 300m
//Filtramos recoridos tiempos mayores a 2h
val training_reg_final_filtrado_DF_1=training_reg_final_DF.filter((training_reg_final_DF("diff_distance")<=30))
val training_reg_final_filtrado_DF_2=training_reg_final_filtrado_DF_1.filter((training_reg_final_DF("diff_distance")>=0.3))
val training_reg_final_filtrado_DF_3 = training_reg_final_filtrado_DF_2.filter((training_reg_final_DF("trip_duration")<=7200))
val training_reg_final_filtrado_DF= training_reg_final_filtrado_DF_3.filter((training_reg_final_DF("passenger_count")<=6)) //eliminamos los casos menos representativos como 0 pasajersos (es un error) 7,8,9(que suman 4 casos en total)

// CREAMOS LA CLASE MAYOR DE 15 MIN O MENOR IGUAL 
val training_reg_final_DF_new=training_reg_final_DF.withColumn("clase",when(((training_reg_final_DF("trip_duration"))>900) ,"Long").
                                           when(((training_reg_final_DF("trip_duration"))<= 900) ,"Short"))


// seleccionamos los atributos con los que vamos a trabajar:
val datosDF_NY=training_reg_final_DF_new.select("vendor_id",
                                                "pickup_month",
                                                "pickup_day",
                                                "pickup_time",
                                                "passenger_count",
                                                "store_and_fwd_flag",
                                                "diff_distance",
                                                "clase")




//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//featurization
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// StringIndexer para pasar el valor categorico a double de la clase , para la features no utilizamos pq ya son doubles. 
val indiceClasePipelineNY = new StringIndexer().setInputCol("clase").setOutputCol("label").setHandleInvalid("skip")

// valores a convertir a continuos
val valoresCategoricosFeatures= Array("vendor_id","pickup_month","pickup_day", "passenger_count", "store_and_fwd_flag")


//StringIndexer para la features
val indexStringFeaturesLlamada = indexStringColumnsStagePipeline(datosDF_NY,valoresCategoricosFeatures)

val indexStringFeaturesTodasNumNY = indexStringFeaturesLlamada._1
val columnnasYaNumericasNY = indexStringFeaturesLlamada._2

// juntamos las columnas convertidas a continuo con StringIndexer y las ya numericas/continuas
val datosDFSinClaseSoloFeaturesNY = valoresCategoricosFeatures.par.map(x=>x+"-num").toArray.union(columnnasYaNumericasNY)

//VectorAssembler para generar una array de valores para las features
val assemblerFeaturesLabelPipelineNY= new VectorAssembler().setOutputCol("features").setInputCols(datosDFSinClaseSoloFeaturesNY.diff(Array("clase"))) // sin la clase o label


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//pipeline
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//generamos el pipeline
val featurizationPipelineNY = new Pipeline().setStages(Array(
                                              indexStringFeaturesTodasNumNY,
                                              assemblerFeaturesLabelPipelineNY,
                                              indiceClasePipelineNY))


// COMMAND ----------

// DBTITLE 1,Supervised - TAXI NY(DT, LR, RF,NB)
// base classifiers

val instanciaTrainingPipelineDT = new Supervised(new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineRF = new Supervised(new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineNB = new Supervised(new NaiveBayes().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineLR = new Supervised(new LogisticRegression().setFeaturesCol("features").setLabelCol("label"))


// intances of baseClasifiers
val arrayClassifiers_Supervised:Array[(String,PipelineStage)] = Array(("DT-Spark",instanciaTrainingPipelineDT),
                                       ("LR-Spark",instanciaTrainingPipelineLR),
                                       ("RF-Spark",instanciaTrainingPipelineRF),
                                       ("NB-Spark",instanciaTrainingPipelineNB)
                                       //("LSVM",instanciaTrainingPipelineLSVM) 
                                      )



val threshold= Array(0.0) // Supervised is n.a 0
val criterion = Array("n.a") // Supervised is n.a

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//parameters:  
// 1.- classifiers, 
// 2.- percentage of data reduction
// 3.-threshold --> 0 is not applicable for Supervised
// 3.- type of data ie: "BCW"
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


val classifierBase = arrayClassifiers_Supervised.map(cls =>cls._1)
val percentageLabeled =Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val dataCode = "TXNY"


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Dataframe template for results
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,
                                          classifierBase,
                                          percentageLabeled 
                                         )


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// pipelines model creations for all the percentatge, baseClasifiers...
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


val modelsPipeline = criterion.map(crit=>(crit,percentageLabeled.map(per=>(per,threshold.map(th=>(th,arrayClassifiers_Supervised.map(clasi=>(clasi._1,new Pipeline().setStages(Array(clasi._2))))))))))
//percentageLabeled.map(per=>(per,threshold.map(th=>(th,arrayClassifiers_Supervised.map(clasi=>(clasi._1,new Pipeline().setStages(Array(clasi._2))))))))

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// final results dataframe
//+++++++++++++++++++++++++++++++++++++++++++++++

val resultsPokerSupervised = SupervisedAndSemiSupervisedResuts(featurizationPipelineNY, 4,datosDF_NY,modelsPipeline,resultsInfo)

//display(resultsPokerSupervised)



// COMMAND ----------

// DBTITLE 1,SelfTrainning - TAXI NY (DT, LR, RF ....)

// clasificadores Base
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.LinearSVC


val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


val arrayClassifiers:Array[(String,PipelineStage)] = Array(("ST-DT-Spark",instanciaTrainingPipelineDT),
                                      ("ST-RF-Spark",instanciaTrainingPipelineRF ),
                                      ("ST-NB-Spark",instanciaTrainingPipelineNB ),
                                      ("ST-LR-Spark",instanciaTrainingPipelineLR)
                                      )

// results for Semisupervised
var SemiSupervisedData = new SemiSupervisedDataResults ()

//parameters
val classifierBase = arrayClassifiers.map(cls =>cls._1)
val percentageLabeled = Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val threshold= Array(0.7,0.8,0.95)//Array(0.9)//Array(0.8,0.9,0.95)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val kBest= Array(0.1)//Array(0.05,0.1,0.2,0.3,0.8)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95) //percentage per 1 of unlabeled total data that should be labeled at the end
val dataCode = "TXNY"
val criterion = Array("threshold")//Array("threshold","kBest")

//template dataFrame results according the parameters
val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,classifierBase, percentageLabeled,threshold,kBest,criterion)

// final pipeline models with all the configurations (parameters)
val modelsPipeline = pipelineModelsSelfTraining(threshold,kBest,percentageLabeled,SemiSupervisedData,arrayClassifiers,criterion)

// dataframe of final results
val resultsTaxiNYST = SupervisedAndSemiSupervisedResuts (featurizationPipelineNY, 4,datosDF_NY,modelsPipeline,resultsInfo,SemiSupervisedData)

//display(resultsTaxiNYST)

// COMMAND ----------

// DBTITLE 1,Co-Training - TAXI

val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


val arrayClassifiers:Array[(String,PipelineStage)] = Array(("CT-DT-Spark",instanciaTrainingPipelineDT),
                                      ("CT-RF-Spark",instanciaTrainingPipelineRF ),
                                      ("CT-NB-Spark",instanciaTrainingPipelineNB ),
                                      ("CT-LR-Spark",instanciaTrainingPipelineLR)
                                      )

// results for Semisupervised
var SemiSupervisedData = new SemiSupervisedDataResults ()

//parameters
val classifierBase = arrayClassifiers.map(cls =>cls._1)
val percentageLabeled = Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val threshold= Array(0.7,0.8,0.95)//Array(0.9)//Array(0.8,0.9,0.95)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val kBest= Array(0.1)//Array(0.05,0.1,0.2,0.3,0.8)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95) //percentage per 1 of unlabeled total data that should be labeled at the end
val dataCode = "TXNY"
val criterion = Array("threshold")//Array("threshold","kBest")

//template dataFrame results according the parameters
val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,classifierBase, percentageLabeled,threshold,kBest,criterion)

// final pipeline models with all the configurations (parameters)
val modelsPipeline = pipelineModelsCoTraining(threshold,kBest,percentageLabeled,SemiSupervisedData,arrayClassifiers,criterion)

// dataframe of final results
val resultsTaxi_CT = SupervisedAndSemiSupervisedResuts (featurizationPipelineNY, 4,datosDF_NY,modelsPipeline,resultsInfo,SemiSupervisedData)
