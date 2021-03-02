// Databricks notebook source
// miramos los documentos adjuntados
display(dbutils.fs.ls("/FileStore/tables"))

// COMMAND ----------

// DBTITLE 1,Class Data SemiSupervised (data result--> data labeled initial,  data unlabeled initial, dataLabeled Final, iterations...)
//++++++++++++++++++++++++++++++++++++++++++++++
// data results for SemiSupervised
//++++++++++++++++++++++++++++++++++++++++++++++++
class  SemiSupervisedDataResults  {

  var dataLabeledFinal :Long =0
  var dataUnDataLabeledFinal:Long =0
  var dataLabeledIni:Long =0
  var dataUnLabeledIni:Long =0
  var iteracionSemiSuper:Int =0
  
}

// COMMAND ----------

// DBTITLE 1,Class UnlabeledTransformer
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{ Param, ParamMap }
import org.apache.spark.ml.util.{ DefaultParamsReadable, DefaultParamsWritable, Identifiable }
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._


//+++++++++++++++++++++++++++++++++++++
//class to untag data 
//+++++++++++++++++++++++++++++++++++++++++

class UnlabeledTransformer(override val uid: String) extends Transformer with DefaultParamsWritable {
  
  var percentageLabeled:Double = 0.1
  var seedValue:Long =11L
  var columnNameNewLabels :String ="labelSelection"

  def this() = this(Identifiable.randomUID("UnlabeledTransformer"))
  
  def setPercentage(percentage:Double)={
    percentageLabeled= percentage
    this
   }
  
  def setColumnLabelName(nameColumn:String)={
    columnNameNewLabels = nameColumn
    this
  }
  
  def setSeed(seedV:Long)={
    seedValue= seedV
    this
  }
  
  override def transform(data: Dataset[_]): DataFrame = {
    val dataSp=  data.randomSplit(Array(percentageLabeled, 1-percentageLabeled),seed = seedValue)
    val dataLabeled = dataSp(0).toDF.withColumn(columnNameNewLabels,col("label"))
    val dataUnlabeled=dataSp(1).toDF.withColumn(columnNameNewLabels,col("label")*Double.NaN) 
    dataLabeled.unionAll(dataUnlabeled)

  }

  override def copy(extra: ParamMap): UnlabeledTransformer = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = schema
}

object UnlabeledTransformer extends DefaultParamsReadable[UnlabeledTransformer] {
  override def load(path: String): UnlabeledTransformer= super.load(path)
}


// COMMAND ----------

// DBTITLE 1,Class Supervised (for labels reduction)
import org.apache.spark.ml.util.Identifiable

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.linalg.DenseVector
import sqlContext.implicits._
import org.apache.spark.sql.functions.udf

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Clase supervisada donde solo entrenamos los datos etiquetado
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Supervised [
  FeatureType,
  E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
] (
    val uid: String,
    val baseClassifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  
    
  ) extends 
org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]
//org.apache.spark.ml.classification.ClassifierParams[FeatureType, E, M]
with Serializable {
  var columnNameNewLabels :String ="labelSelection"
  //uid
  def this(classifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]) =
    this(Identifiable.randomUID("Supervised"), classifier)
  
  // set column labels 
  def setColumnLabelName(nameColumn:String)={
    columnNameNewLabels = nameColumn
    this
  }
  
  
  def train(dataset: org.apache.spark.sql.Dataset[_]): M= {

    val dataUnLabeled=dataset.filter(dataset(columnNameNewLabels).isNaN).toDF
    val dataLabeled = dataset.toDF.exceptAll(dataUnLabeled)
    baseClassifier.fit(dataLabeled)

  }
  
  override def copy(extra: org.apache.spark.ml.param.ParamMap):E = defaultCopy(extra)
}





// COMMAND ----------

// DBTITLE 1,Class SemiSupervised- SelfTrainning 
import org.apache.spark.ml.util.Identifiable

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.linalg.DenseVector
import sqlContext.implicits._
import org.apache.spark.sql.functions.udf

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Class --> SELF TRAINING 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SelfTraining [
  FeatureType,
  E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
] (
    val uid: String,
    val baseClassifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],

    
  ) extends org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M] with Serializable {
  
  
  var porcentajeLabeled:Double = 1.0
  var threshold:Double=0.7
  var maxIter:Int=7
  var criterion:String= "threshold"
  var kBest:Double=1.0 // percentage
  var countDataLabeled:Long = _
  var countDataUnLabeled:Long = _
  var dataLabeledIni:Long =_
  var dataUnLabeledIni:Long = _
  var iter:Int = 0
  var columnNameNewLabels :String ="labelSelection"
  var resultsSelfTrainingData: SemiSupervisedDataResults =_
  var numberOfkBest:Int=0

  //uid
  def this(classifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]) =
    this(Identifiable.randomUID("selfTrainning"),classifier)
  

  //SETTERS
  
 
    //Set columnLabels --> Labeled and Unlabeled
  def setSemiSupervisedDataResults(semiSupervisedResults:SemiSupervisedDataResults)={
    resultsSelfTrainingData = semiSupervisedResults
    this
  }
  
  
  
  //Set columnLabels --> Labeled and Unlabeled
  def setColumnLabelName(nameColumn:String)={
    columnNameNewLabels = nameColumn
    this
  }
  
  
  // set porcentaje
  def setPorcentaje(porcentaje:Double)={
    porcentajeLabeled = porcentaje
    this
    
  }
  

  def setThreshold(thres:Double)={
    threshold = thres
    this
    
  }
  
  // maxIter
  def setMaxITer(maIter:Int)={
    maxIter= maIter
    this
    
  }
  
  // criterion
  def setCriterion(cri:String)={
    criterion= cri
    this
    
  }
  
  //kBest
  def setKbest(kb:Double)={
    kBest = kb
    this
  }
  
    
  // getters
  
  def getDataLabeledFinal():Long={
    countDataLabeled
  }
  
  def getUnDataLabeledFinal():Long={
    countDataUnLabeled
  }  
  
  def getDataLabeledIni():Long={
    dataLabeledIni
  }
  
   def getUnDataLabeledIni():Long={
     dataUnLabeledIni
  }  
  
  def getIter():Int={
     iter
  }  
  

  
  def train(dataset: org.apache.spark.sql.Dataset[_]): M= {
    iter = 1
    //udf to get he max value from probabilisti array
    val max = udf((v: org.apache.spark.ml.linalg.Vector) => v.toArray.max)
    
    var dataUnLabeled=dataset.filter(dataset(columnNameNewLabels).isNaN).toDF.cache()
    var dataLabeled = dataset.toDF.exceptAll(dataUnLabeled).cache()

    //get the data labeled and unlabeled initial
    dataLabeledIni = dataLabeled.count()
    dataUnLabeledIni = dataUnLabeled.count()
    //selection features and labels
    dataLabeled = dataLabeled.select("features","label")
    dataUnLabeled = dataUnLabeled .select("features","label")
    countDataLabeled = dataLabeled.count()
    countDataUnLabeled = dataUnLabeled.count()
    

    
    var modeloIterST = baseClassifier.fit(dataLabeled)
    var prediIterST = modeloIterST.transform(dataUnLabeled)
    
    dataLabeled.unpersist()
    dataUnLabeled.unpersist()
  
    if (criterion == "threshold"){
      
      while ((iter<maxIter) && (countDataUnLabeled>0)){

        var modificacionPrediccion=prediIterST.withColumn("probMax", max($"probability"))
        var labelsHigherOfThreshold=modificacionPrediccion.filter(modificacionPrediccion("probMax")>threshold)
        var labelsLowerOfThreshold =modificacionPrediccion.filter(modificacionPrediccion("probMax")<=threshold)

        // get features and predictions and change the name from prediction to label in order to add as new data labeled
        var newLabeledFeaturesLabels = labelsHigherOfThreshold.select ("features","prediction").withColumnRenamed("prediction","label")
        var newUnLabeledFeaturesLabels = labelsLowerOfThreshold.select ("features","prediction").withColumnRenamed("prediction","label")

        dataLabeled = dataLabeled.union(newLabeledFeaturesLabels).cache()
        dataUnLabeled = newUnLabeledFeaturesLabels.cache()
        countDataUnLabeled = dataUnLabeled.count()
        countDataLabeled = dataLabeled.count()


        if (countDataUnLabeled>0 && iter<maxIter ){
          
          modeloIterST = baseClassifier.fit(dataLabeled)
          prediIterST = modeloIterST.transform(dataUnLabeled)
          iter = iter+1
          
        }
        else{ //final
          modeloIterST = baseClassifier.fit(dataLabeled)
        }

        dataLabeled.unpersist()
        dataUnLabeled.unpersist()

      }
    
    }
    else if (criterion == "kBest"){
      
      numberOfkBest = ((kBest* countDataUnLabeled)/(maxIter-1)).round.toInt
      
      while ((iter<maxIter) && (countDataUnLabeled>0)){
        
        var modificacionPrediccion=prediIterST.withColumn("probMax", max($"probability"))
        var newLabeledFeaturesLabelsHigherProb  = modificacionPrediccion.sort(col("probMax").desc).limit(numberOfkBest)
        var newUnLabeledFeaturesLabels =  modificacionPrediccion.exceptAll(newLabeledFeaturesLabelsHigherProb).select ("features","prediction").withColumnRenamed("prediction","label")
        var newLabeledFeaturesLabels  = newLabeledFeaturesLabelsHigherProb.select ("features","prediction").withColumnRenamed("prediction","label")        
        
        dataLabeled = dataLabeled.union(newLabeledFeaturesLabels).cache()
        dataUnLabeled = newUnLabeledFeaturesLabels.cache()
        countDataUnLabeled = dataUnLabeled.count()
        countDataLabeled = dataLabeled.count()


        if (countDataUnLabeled>0 && iter<maxIter ){
          modeloIterST = baseClassifier.fit(dataLabeled)
          prediIterST = modeloIterST.transform(dataUnLabeled)
          iter = iter+1
        }
        else{ //final
          modeloIterST = baseClassifier.fit(dataLabeled)
        }

        dataLabeled.unpersist()
        dataUnLabeled.unpersist()
      }
      
      
      
    }

    // load the semisupervised results regarding the labeled and unlabeled data using the SemiSupervisedDataResults class
    resultsSelfTrainingData.dataLabeledFinal =countDataLabeled
    resultsSelfTrainingData.dataUnDataLabeledFinal =countDataUnLabeled
    resultsSelfTrainingData.dataLabeledIni =  dataLabeledIni
    resultsSelfTrainingData.dataUnLabeledIni = dataUnLabeledIni
    resultsSelfTrainingData.iteracionSemiSuper = iter
    
    // Final model
    modeloIterST



  }
  
  
  override def transformSchema(schema: StructType): StructType = schema
  
  override def copy(extra: org.apache.spark.ml.param.ParamMap):E = defaultCopy(extra)
}





// COMMAND ----------

// DBTITLE 1,Class CoTraining SemiSupervised
import org.apache.spark.ml.util.Identifiable

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.linalg.DenseVector
import sqlContext.implicits._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Class --> CoTraining
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CoTraining [
  FeatureType,
  E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
] (
    val uid: String,
    val baseClassifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],

    
  ) extends org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M] with Serializable {
  
  
  var porcentajeLabeled:Double = 1.0
  var threshold:Double=0.7
  var maxIter:Int=3
  var criterion:String= "threshold"
  var kBest:Double=1.0 // percentage
  var countDataLabeled:Long = _
  var countDataLabeled_1:Long = _
  var countDataLabeled_2:Long = _
  var countDataUnLabeled_1:Long = 1
  var countDataUnLabeled_2:Long = 1
  var countDataUnLabeled:Long = _
  var dataLabeledIni:Long =_

  var dataUnLabeledIni:Long = _

  var dataLabeled:Long = _
  
  var iter:Int = 0
  var columnNameNewLabels :String ="labelSelection"
  var resultsSelfTrainingData: SemiSupervisedDataResults =_
  var numberOfkBest:Int=0
  var modeloIterST: M=_

  //uid
  def this(classifier1: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]/*,
           classifier2: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
           classifier3: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]*/) =
    this(Identifiable.randomUID("CoTrainning"),classifier1/*,classifier2,classifier3*/)
  

  //SETTERS
  
 
    //Set columnLabels --> Labeled and Unlabeled
  def setSemiSupervisedDataResults(semiSupervisedResults:SemiSupervisedDataResults)={
    resultsSelfTrainingData = semiSupervisedResults
    this
  }
  
  
  
  //Set columnLabels --> Labeled and Unlabeled
  def setColumnLabelName(nameColumn:String)={
    columnNameNewLabels = nameColumn
    this
  }
  
  
  // set porcentaje
  def setPorcentaje(porcentaje:Double)={
    porcentajeLabeled = porcentaje
    this
    
  }
  

  def setThreshold(thres:Double)={
    threshold = thres
    this
    
  }
  
  // maxIter
  def setMaxITer(maIter:Int)={
    maxIter= maIter
    this
    
  }
  
  // criterion
  def setCriterion(cri:String)={
    criterion= cri
    this
    
  }
  
  //kBest
  def setKbest(kb:Double)={
    kBest = kb
    this
  }
  
    
  // getters
  
  def getDataLabeledFinal():Long={
    countDataLabeled
  }
  
  def getUnDataLabeledFinal():Long={
    countDataUnLabeled
  }  
  
  def getDataLabeledIni():Long={
    dataLabeledIni
  }
  
   def getUnDataLabeledIni():Long={
     dataUnLabeledIni
  }  
  
  def getIter():Int={
     iter
  }  
  

  
  def train(dataset: org.apache.spark.sql.Dataset[_]): M= {
    iter = 1
    //udf to get he max value from probabilisti array
    val max = udf((v: org.apache.spark.ml.linalg.Vector) => v.toArray.max)
    
    //var dataUnLabeled=dataset.filter(dataset(columnNameNewLabels).isNaN).toDF.cache()
    //var dataLabeled = dataset.toDF.except(dataUnLabeled).cache()
    //println ("dataset: " + dataset.count)
    var dataUnLabeled=dataset.filter(dataset(columnNameNewLabels).isNaN).toDF.cache()
    var dataLabeled = dataset.toDF.exceptAll(dataUnLabeled).cache()

    //get the data labeled and unlabeled initial
    dataLabeledIni = dataLabeled.count()
    dataUnLabeledIni = dataUnLabeled.count()
    
    //split data in two datasets.
    
    val dataSplitsLabel=  dataLabeled.randomSplit(Array(0.5, 0.5),seed = 8L)
    var dataLabeled_1 = dataSplitsLabel(0)
    var dataLabeled_2 = dataSplitsLabel(1)
    
    val dataSplitsUnLabel=  dataUnLabeled.randomSplit(Array(0.5, 0.5),seed = 8L)
    var dataUnLabeled_1 = dataSplitsUnLabel(0)
    var dataUnLabeled_2 = dataSplitsUnLabel(1)

    
    //selection features and labels
    dataUnLabeled = dataUnLabeled .select("features","label")
    dataUnLabeled_1 = dataUnLabeled_1 .select("features","label").cache()
    dataUnLabeled_2 = dataUnLabeled_2 .select("features","label").cache()
    dataLabeled_1 = dataLabeled_1.select("features","label").cache()
    dataLabeled_2 = dataLabeled_2.select("features","label").cache()

    countDataLabeled_1 = dataLabeled_1.count()
    countDataLabeled_2 = dataLabeled_2.count()
    countDataUnLabeled = dataUnLabeled.count()
    
    /*println("+++++++++++++++++++++++++++++++++++++++++++")
    println("Initial data")
    println("+++++++++++++++++++++++++++++++++++++++++++")
    println ("countDataLabeled_1: "+countDataLabeled_1)
    println ("countDataLabeled_2: "+countDataLabeled_2)
    println ("countDataUnLabeled: "+countDataUnLabeled)*/
    
    
    var modeloIterST_1 = baseClassifier.fit(dataLabeled_1)
    var prediIterST_1 = modeloIterST_1.transform(dataUnLabeled_1)
    
    var modeloIterST_2 = baseClassifier.fit(dataLabeled_2)
    var prediIterST_2 = modeloIterST_2.transform(dataUnLabeled_2)
    
 /*   dataLabeled.unpersist()
    dataUnLabeled.unpersist()
    dataLabeled_1.unpersist()
    dataLabeled_2.unpersist()*/
  
    if (criterion == "threshold"){
      
      while ((iter<maxIter) && (countDataUnLabeled_1>0) &&(countDataUnLabeled_2>0)){
        
        // model 1
        var modificacionPrediccion_1=prediIterST_1.withColumn("probMax", max($"probability"))
        var labelsHigherOfThreshold_1=modificacionPrediccion_1.filter(modificacionPrediccion_1("probMax")>threshold)
        //var labelsLowerOfThreshold_1 =modificacionPrediccion_1.filter(modificacionPrediccion_1("probMax")<=threshold)
        dataUnLabeled_1 =modificacionPrediccion_1.filter(modificacionPrediccion_1("probMax")<=threshold).select ("features","prediction").withColumnRenamed("prediction","label")
        var newLabeledFeaturesLabels_1 = labelsHigherOfThreshold_1.select ("features","prediction").withColumnRenamed("prediction","label")
        //var newUnLabeledFeaturesLabels_1 = labelsLowerOfThreshold_1.select ("features","prediction").withColumnRenamed("prediction","label")
        
        //model 2
        var modificacionPrediccion_2=prediIterST_2.withColumn("probMax", max($"probability"))
        var labelsHigherOfThreshold_2=modificacionPrediccion_2.filter(modificacionPrediccion_2("probMax")>threshold)
       // var labelsLowerOfThreshold_2 =modificacionPrediccion_2.filter(modificacionPrediccion_2("probMax")<=threshold)
        dataUnLabeled_2 =modificacionPrediccion_2.filter(modificacionPrediccion_2("probMax")<=threshold).select ("features","prediction").withColumnRenamed("prediction","label")
        var newLabeledFeaturesLabels_2 = labelsHigherOfThreshold_2.select ("features","prediction").withColumnRenamed("prediction","label")
        //var newUnLabeledFeaturesLabels_2 = labelsLowerOfThreshold_2.select ("features","prediction").withColumnRenamed("prediction","label")
        
        // new Unlabeled data coming from newUnlabeleded_1 and newUnlabeled_2
        //var newUnLabeledFeaturesLabels = dataUnLabeled.exceptAll(ewUnLabeledFeaturesLabels_1).exceptAll(newUnLabeledFeaturesLabels_2)//newUnLabeledFeaturesLabels_2.union(newUnLabeledFeaturesLabels_1)
        
       
        //println("newLabeledFeaturesLabels_1: "+newLabeledFeaturesLabels_1.count)
        //println("newLabeledFeaturesLabels_2: "+newLabeledFeaturesLabels_2.count)
        
        dataLabeled_1 = dataLabeled_1.unionAll(newLabeledFeaturesLabels_2)//.cache()
        dataLabeled_2 = dataLabeled_2.unionAll(newLabeledFeaturesLabels_1)//.cache()
        //dataUnLabeled = dataUnLabeled.except(newLabeledFeaturesLabels_1).except(newLabeledFeaturesLabels_2)//newUnLabeledFeaturesLabels.cache()
        

        countDataUnLabeled_1 = dataUnLabeled_1.count()
        countDataUnLabeled_2 = dataUnLabeled_2.count()
        countDataLabeled_1 = dataLabeled_1.count()
        countDataLabeled_2 = dataLabeled_2.count()

        if ((countDataUnLabeled_1>0)&& (countDataUnLabeled_2>0) && (iter<maxIter) ){
          
         /* println("+++++++++++++++++++++++++++++++++++++++++++")
          println("iter: "+iter)
          println("+++++++++++++++++++++++++++++++++++++++++++")
          println ("countDataLabeled_1: "+countDataLabeled_1)
          println ("countDataLabeled_2: "+countDataLabeled_2)
          println ("countDataUnLabeled_1: "+countDataUnLabeled_1)
          println ("countDataUnLabeled_2: "+countDataUnLabeled_2)*/
          
          modeloIterST_1 = baseClassifier.fit(dataLabeled_1)
          prediIterST_1 = modeloIterST_1.transform(dataUnLabeled_1)

          modeloIterST_2 = baseClassifier.fit(dataLabeled_2)
          prediIterST_2 = modeloIterST_2.transform(dataUnLabeled_2)

          iter = iter+1
        }
        /*else {
        
          println("+++++++++++++++++++++++++++++++++++++++++++")
          println("No new model - iter: "+iter)
          println("+++++++++++++++++++++++++++++++++++++++++++")
          println ("countDataLabeled_1: "+countDataLabeled_1)
          println ("countDataLabeled_2: "+countDataLabeled_2)
          println ("countDataUnLabeled_1: "+countDataUnLabeled_1)
          println ("countDataUnLabeled_2: "+countDataUnLabeled_2)
        }*/

        /*dataLabeled_1.unpersist()
        dataLabeled_2.unpersist()
        dataUnLabeled_1.unpersist()
        dataUnLabeled_2.unpersist()
        dataUnLabeled.unpersist()
        //dataLabeled_1.persist() */
        
      }
    // final model
    //ini 

    //countDataUnLabeled_1 = 1
    //countDataUnLabeled_2 = 1

    //unpersist
    dataLabeled_1.unpersist()
    dataLabeled_2.unpersist()
    dataUnLabeled_1.unpersist()
    dataUnLabeled_2.unpersist()
    dataUnLabeled.unpersist()
      //dataLabeled_1.persist()

    }
    else if (criterion == "kBest"){
      
      numberOfkBest = ((kBest* countDataUnLabeled)/(maxIter-1)).round.toInt
      
       while ((iter<maxIter) && (countDataUnLabeled_1>0) &&(countDataUnLabeled_2>0)){
        
        // model 1
        
        var modificacionPrediccion_1=prediIterST_1.withColumn("probMax", max($"probability"))
        var newLabeledFeaturesLabelsHigherProb_1  = modificacionPrediccion_1.sort(col("probMax").desc).limit(numberOfkBest)
        var newUnLabeledFeaturesLabels_1 =  modificacionPrediccion_1.exceptAll(newLabeledFeaturesLabelsHigherProb_1).select ("features","prediction").withColumnRenamed("prediction","label")
        var newLabeledFeaturesLabels_1  = newLabeledFeaturesLabelsHigherProb_1.select ("features","prediction").withColumnRenamed("prediction","label")  
        var dataUnLabeled_1 = newUnLabeledFeaturesLabels_1//.cache()
        
        // model 2
        
        var modificacionPrediccion_2=prediIterST_2.withColumn("probMax", max($"probability"))
        var newLabeledFeaturesLabelsHigherProb_2  = modificacionPrediccion_2.sort(col("probMax").desc).limit(numberOfkBest)
        var newUnLabeledFeaturesLabels_2 =  modificacionPrediccion_2.exceptAll(newLabeledFeaturesLabelsHigherProb_2).select ("features","prediction").withColumnRenamed("prediction","label")
        var newLabeledFeaturesLabels_2  = newLabeledFeaturesLabelsHigherProb_2.select ("features","prediction").withColumnRenamed("prediction","label")        
        var dataUnLabeled_2 = newUnLabeledFeaturesLabels_2//.cache()
        
        // new Unlabeled data coming from newUnlabeleded_1 and newUnlabeled_2
        //var newUnLabeledFeaturesLabels = newUnLabeledFeaturesLabels_2.union(newUnLabeledFeaturesLabels_1)

        
         
        dataLabeled_1 = dataLabeled_1.unionAll(newLabeledFeaturesLabels_2)//.cache()
        dataLabeled_2 = dataLabeled_2.unionAll(newLabeledFeaturesLabels_1)//.cache()
        
        countDataUnLabeled_1 = dataUnLabeled_1.count()
        countDataUnLabeled_2 = dataUnLabeled_2.count()
        countDataLabeled_1 = dataLabeled_1.count()
        countDataLabeled_2 = dataLabeled_2.count()

        if ((countDataUnLabeled_1>0)&& (countDataUnLabeled_2>0) && (iter<maxIter) ){
          
          println("+++++++++++++++++++++++++++++++++++++++++++")
          println("iter: "+iter)
          println("+++++++++++++++++++++++++++++++++++++++++++")
          println ("countDataLabeled_1: "+countDataLabeled_1)
          println ("countDataLabeled_2: "+countDataLabeled_2)
          println ("countDataUnLabeled_1: "+countDataUnLabeled_1)
          println ("countDataUnLabeled_2: "+countDataUnLabeled_2)
          
          modeloIterST_1 = baseClassifier.fit(dataLabeled_1)
          prediIterST_1 = modeloIterST_1.transform(dataUnLabeled_1)

          modeloIterST_2 = baseClassifier.fit(dataLabeled_2)
          prediIterST_2 = modeloIterST_2.transform(dataUnLabeled_2)

          iter = iter+1
        }
        else {
        
          println("+++++++++++++++++++++++++++++++++++++++++++")
          println("No new model - iter: "+iter)
          println("+++++++++++++++++++++++++++++++++++++++++++")
          println ("countDataLabeled_1: "+countDataLabeled_1)
          println ("countDataLabeled_2: "+countDataLabeled_2)
          println ("countDataUnLabeled_1: "+countDataUnLabeled_1)
          println ("countDataUnLabeled_2: "+countDataUnLabeled_2)
        }

        /*dataUnLabeled.unpersist()
        dataLabeled_1.unpersist()
        dataLabeled_2.unpersist()
        dataUnLabeled_1.unpersist()
        dataUnLabeled_2.unpersist() */

        
      }
      // final model


      //unpersist
      dataLabeled_1.unpersist()
      dataLabeled_2.unpersist()
      dataUnLabeled_1.unpersist()
      dataUnLabeled_2.unpersist()
      dataUnLabeled.unpersist()
       
    }

    // load the semisupervised results regarding the labeled and unlabeled data using the SemiSupervisedDataResults class
    resultsSelfTrainingData.dataLabeledFinal =countDataLabeled_1 + countDataLabeled_2
    resultsSelfTrainingData.dataUnDataLabeledFinal =countDataUnLabeled_1 + countDataUnLabeled_2
    resultsSelfTrainingData.dataLabeledIni =  dataLabeledIni
    resultsSelfTrainingData.dataUnLabeledIni = dataUnLabeledIni
    resultsSelfTrainingData.iteracionSemiSuper = iter
    
    //ini
    countDataUnLabeled_1 = 1
    countDataUnLabeled_2 = 1
    
    // Final model
    baseClassifier.fit(dataLabeled_1.unionAll(dataLabeled_2))



  }
  
  

  override def transformSchema(schema: StructType): StructType = schema
  
  override def copy(extra: org.apache.spark.ml.param.ParamMap):E = defaultCopy(extra)
}





// COMMAND ----------

// DBTITLE 1,Functions Super/SemiSupervised
import org.apache.spark.sql.types.{StructType,StructField,StringType,DoubleType}
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.col
import spark.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.PipelineStage
import scala.util.control.Breaks._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils.kFold
import org.apache.spark.sql.functions._
import org.apache.spark.ml.PipelineStage
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object functionsSemiSupervised {
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  //creation PipelineStatge to convert features from categorical to continuos
  //output array[PipelineStage]
  //*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  def indexStringColumnsStagePipeline(df:DataFrame,cols:Array[String]):(Pipeline,Array[String])= {
    var intermedioStages:Array[(PipelineStage)] = new Array[(PipelineStage)](cols.size)
    var posicion = 0
    for(col <-cols) {
      val si= new StringIndexer().setInputCol(col).setOutputCol(col+"-num")
      intermedioStages(posicion) = si.setHandleInvalid("keep")
      posicion = posicion +1
    }
    val output = new Pipeline().setStages(intermedioStages)
    (output,df.columns.diff(cols))
  }

  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // pipelines SelfTraining Creation
  // percentatge, threshold, Classifier, pipeline, i.e: (0.01,0.4,ST-DT,pipeline_87c711c3e400)
  //                                                    (0.01,0.4,ST-LB,pipeline_87c711c3e400)
  //                                                    ...
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  def pipelineModelsSelfTraining [
      FeatureType,
      E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
      M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
    ] (threshold:Array[Double],
       kBest:Array[Double],
       percentage:Array[Double],
       resultsSemiSupData:SemiSupervisedDataResults,
       arrayClasiInstanceModel: Array[(String, org.apache.spark.ml.PipelineStage)],
       criterion:Array[String],
       iterations:Int=7):Array[(String, Array[(Double, Array[(Double, Array[(String, org.apache.spark.ml.Pipeline)])])])]= 
  {
      criterion.map(crit=>(crit,percentage.map(per=>(per, 
                                                     if (crit =="threshold")
                                                     {
                                                       threshold.map(th=>(th,arrayClasiInstanceModel.map(clasi=>(clasi._1,new Pipeline().setStages(Array(new SelfTraining(clasi._2.asInstanceOf[E])
                                                                                                                                      .setThreshold(th)
                                                                                                                                      .setCriterion(crit)
                                                                                                                                      .setMaxITer(iterations)
                                                                                                                                      .setSemiSupervisedDataResults(resultsSemiSupData)))))))
                                                     }
                                                     else // kBest
                                                     {
                                                       kBest.map(kb=>(kb,arrayClasiInstanceModel.map(clasi=>(clasi._1,new Pipeline().setStages(Array(new SelfTraining(clasi._2.asInstanceOf[E])
                                                                                                                                      .setKbest(kb)
                                                                                                                                      .setCriterion(crit)
                                                                                                                                      .setMaxITer(iterations)
                                                                                                                                      .setSemiSupervisedDataResults(resultsSemiSupData)))))))
                                                     }
                                                    ))))

  }


  def pipelineModelsCoTraining [
      FeatureType,
      E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
      M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
    ] (threshold:Array[Double],
       kBest:Array[Double],
       percentage:Array[Double],
       resultsSemiSupData:SemiSupervisedDataResults,
       arrayClasiInstanceModel: Array[(String, org.apache.spark.ml.PipelineStage)],
       criterion:Array[String],
       iterations:Int=7):Array[(String, Array[(Double, Array[(Double, Array[(String, org.apache.spark.ml.Pipeline)])])])]= 
  {
      criterion.map(crit=>(crit,percentage.map(per=>(per, 
                                                     if (crit =="threshold")
                                                     {
                                                       threshold.map(th=>(th,arrayClasiInstanceModel.map(clasi=>(clasi._1,new Pipeline()
                                                                                                                 .setStages(Array(new CoTraining(clasi._2.asInstanceOf[E]
                                                                                                                                                 /*,clasi._2.asInstanceOf[E],clasi._2.asInstanceOf[E]*/)
                                                                                                                                      .setThreshold(th)
                                                                                                                                      .setCriterion(crit)
                                                                                                                                      .setMaxITer(iterations)
                                                                                                                                      .setSemiSupervisedDataResults(resultsSemiSupData)))))))
                                                     }
                                                     else // kBest
                                                     {
                                                       kBest.map(kb=>(kb,arrayClasiInstanceModel.map(clasi=>(clasi._1,new Pipeline()
                                                                                                             .setStages(Array(new CoTraining(clasi._2.asInstanceOf[E]
                                                                                                                                             /*,clasi._2.asInstanceOf[E],clasi._2.asInstanceOf[E]*/)
                                                                                                                                      .setKbest(kb)
                                                                                                                                      .setCriterion(crit)
                                                                                                                                      .setMaxITer(iterations)
                                                                                                                                      .setSemiSupervisedDataResults(resultsSemiSupData)))))))
                                                     }
                                                    ))))

  }



  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // restults DataFrame template
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


  //|data| clasifi| porcentaje| threshold| iter| LabeledInicial| UnlabeledInicial| LabeledFinal| UnlabeledFinal| porEtiFinal| Acc| AUC| PR| F1Score|
  //+----+--------+----------+-----------+-----+---------------+-----------------+-------------+---------------+------------+----+----+----+-------+
  //| BCW|   ST-DT|      0.001|       0.6|    2|            130|           100600|        98000|           2600|       97.4 | x.x| x.x| x.x|    x.x|
  //| BCW|   ST-DT|      0.005|       0.6|    3|            270|           100460|        96000|           2800|       97.21| x.x| x.x| x.x|    x.x|
  //....


  def generadorDataFrameResultadosSemiSuper(data:String,
                                            classifiers:Array[String],
                                            percentage:Array[Double],
                                            threshold:Array[Double]=Array(0.0),
                                            kBest:Array[Double]=Array(0.0),
                                            criterion:Array[String]=Array("n.a"),
                                           ):DataFrame= 
  {
    // creamos la array de salida con el type y el tamaño
    var seqValores=Seq[(String,String,String,Double, Double,Int,Int,Long,Long,Long,Double,Double, Double,Double, Double)]() 
    //var seqValores=Seq[(String, String,Double, Double,Int,Int,Int,Int,Int,Double,Double, Double,Double, Double)]() 
    var posicion = 0
    var thresholdOrKbest:Array[Double]=Array(0.0)
    for (crit <-criterion){
      if (crit == "kBest"){
        thresholdOrKbest = kBest
      }
      else if (crit == "threshold"){
        thresholdOrKbest =threshold
      }
      for(posClasi <-classifiers) {
        for(posPorce <-percentage){
          for (posThreshold <- thresholdOrKbest){
            seqValores = seqValores :+ (data,posClasi,crit,posPorce,posThreshold,0,0,0.toLong,0.toLong,0.toLong,0.00,0.00,0.00,0.00,0.00)
          }
        }
      }
    }
    // generamos el DataFrame que sera la salida
    spark.createDataFrame(seqValores).toDF("data",
                                           "classifier",//"clasificador",
                                           "criterion",
                                           "percentageLabeled",//"porcentajeEtiquetado",
                                           "thresholdOrKBest",
                                           "iteration",//"iteracion",
                                           "LabeledInitial",//"LabeledInicial",
                                           "UnLabeledInitial",//"UnLabeledInicial",
                                           "LabeledFinal",
                                           "UnLabeledFinal",
                                           "percentageLabeledFinal",//"porEtiFinal", 
                                           "accuracy",
                                           "AUC",
                                           "PR",
                                           "F1score")
  }

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  //results calculation
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


  def SupervisedAndSemiSupervisedResuts  (featurization:Pipeline,
       kfold:Int,
       data:DataFrame,
       modelsPipeline: Array[(String, Array[(Double, Array[(Double, Array[(String, org.apache.spark.ml.Pipeline)])])])],
       info:DataFrame,
       resultSemiSupervisedData:SemiSupervisedDataResults = new SemiSupervisedDataResults()):DataFrame= 
  {
    var newdf= info
    var unlabeledProcess =new UnlabeledTransformer()
    var pipeline:Pipeline= new Pipeline()
    var results:(Double, Double, Double, Double, Long, Long, Long, Long, Int) = (0.0,0.0,0.0,0.0,0.toLong,0.toLong,0.toLong,0.toLong,0)
    //modelsPipeline.map(criterion=>criterion._2.map
    modelsPipeline.map(criterion=>criterion._2.map(percentatge=>percentatge._2.map(threshold=>threshold._2.map(classi=>(pipeline = new Pipeline().setStages(Array(featurization,
                                                                                                                                      unlabeledProcess.setPercentage(percentatge._1),
                                                                                                                                      classi._2)),

                                                                                            results = crossValidation(data,kfold,pipeline,resultSemiSupervisedData),

                                                                                            newdf = newdf.withColumn("accuracy",when(newdf("percentageLabeled")=== percentatge._1 &&
                                                                                                                                    newdf("criterion")===criterion._1 &&
                                                                                                                                    newdf("classifier")===classi._1&&
                                                                                                                                    newdf("thresholdOrKBest")=== threshold._1
                                                                                                                                    ,results._1).otherwise (newdf("accuracy"))),
                                                                                            newdf = newdf.withColumn("AUC",when(newdf("percentageLabeled")===percentatge._1 &&
                                                                                                                                newdf("criterion")===criterion._1 &&
                                                                                                                                newdf("classifier")===classi._1 &&
                                                                                                                                newdf("thresholdOrKBest")=== threshold._1
                                                                                                                                ,results._2).otherwise (newdf("AUC"))),
                                                                                            newdf = newdf.withColumn("PR",when(newdf("percentageLabeled")===percentatge._1 &&
                                                                                                                               newdf("criterion")===criterion._1 &&
                                                                                                                                newdf("classifier")===classi._1 && 
                                                                                                                                newdf("thresholdOrKBest")=== threshold._1
                                                                                                                                , results._3).otherwise (newdf("PR"))),
                                                                                            newdf = newdf.withColumn("F1score",when(newdf("percentageLabeled")===percentatge._1 &&
                                                                                                                                    newdf("criterion")===criterion._1 &&
                                                                                                                                    newdf("classifier")===classi._1 &&
                                                                                                                                    newdf("thresholdOrKBest")=== threshold._1
                                                                                                                                    ,results._4).otherwise (newdf("F1score"))),
                                                                                            newdf = newdf.withColumn("iteration",when(newdf("percentageLabeled")===percentatge._1 && 
                                                                                                                                      newdf("criterion")===criterion._1 &&
                                                                                                                                      newdf("classifier")===classi._1  &&
                                                                                                                                      newdf("thresholdOrKBest")=== threshold._1
                                                                                                                                      ,results._9).otherwise (newdf("iteration"))),
                                                                                            newdf = newdf.withColumn("LabeledInitial",when(newdf("percentageLabeled")===percentatge._1  && 
                                                                                                                                           newdf("criterion")===criterion._1 &&
                                                                                                                                           newdf("classifier")===classi._1 &&  
                                                                                                                                           newdf("thresholdOrKBest")=== threshold._1 
                                                                                                                                           ,results._5 ).otherwise (newdf("LabeledInitial"))),
                                                                                            newdf = newdf.withColumn("UnLabeledInitial",when(newdf("percentageLabeled")===percentatge._1  &&
                                                                                                                                             newdf("criterion")===criterion._1 &&
                                                                                                                                             newdf("classifier")===classi._1 && 
                                                                                                                                             newdf("thresholdOrKBest")=== threshold._1 
                                                                                                                                             ,results._6).otherwise (newdf("UnLabeledInitial"))),
                                                                                            newdf = newdf.withColumn("LabeledFinal",when(newdf("percentageLabeled")===percentatge._1  && 
                                                                                                                                         newdf("criterion")===criterion._1 &&
                                                                                                                                         newdf("classifier")===classi._1 && 
                                                                                                                                         newdf("thresholdOrKBest")=== threshold._1 
                                                                                                                                         ,results._7).otherwise (newdf("LabeledFinal"))),
                                                                                            newdf = newdf.withColumn("UnLabeledFinal",when(newdf("percentageLabeled")===percentatge._1 &&
                                                                                                                                           newdf("criterion")===criterion._1 &&
                                                                                                                                           newdf("classifier")===classi._1 && 
                                                                                                                                           newdf("thresholdOrKBest")=== threshold._1 
                                                                                                                                           , results._8).otherwise(newdf("UnLabeledFinal"))),
                                                                                            newdf = newdf.withColumn("percentageLabeledFinal",when(newdf("percentageLabeled")===percentatge._1 && 
                                                                                                                                        newdf("criterion")===criterion._1 &&
                                                                                                                                        newdf("classifier")===classi._1 && 
                                                                                                                                        newdf("thresholdOrKBest")=== threshold._1
                                                                                                                                        ,(1 -(results._8.toDouble/results._6.toDouble)))
                                                                                                                     .otherwise (newdf("percentageLabeledFinal"))) 

                                                                                           )))))
     newdf
  }
  
  
   //++++++++++++++++++++++++++++++++++++++++++++++
  //Cross Validator
  //++++++++++++++++++++++++++++++++++++++++++++++
  def crossValidation[
      FeatureType,
      E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
      M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
    ](data:DataFrame,
      kFolds:Int,
      modelsPipeline:Pipeline,
      resultsSSData:SemiSupervisedDataResults=new SemiSupervisedDataResults()): (Double, Double, Double, Double, Long, Long, Long, Long, Int)= {
    // creamos la array de salida con el type y el tamaño

    var folds = kFold(data.rdd, kFolds, 8L)
    var acierto:Double = 0.0
    var auROC:Double = 0.0
    var auPR:Double = 0.0
    var f1Score:Double = 0.0
    var labeledIni:Int = 0

    var dataLabeledFinal :Long =0
    var dataUnDataLabeledFinal:Long =0
    var dataLabeledIni:Long =0
    var dataUnLabeledIni:Long =0
    var iteracionSemiSuper:Int =0



    for(iteration <-0 to kFolds-1) {
       var dataTraining=spark.createDataFrame(folds(iteration)._1, data.schema)
       var dataTest=spark.createDataFrame(folds(iteration)._2, data.schema)
       dataTraining.persist()
       dataTest.persist()



       var predictionsAndLabelsRDD=modelsPipeline.fit(dataTraining)
        .transform(dataTest)
        .select("prediction", "label").rdd.map(row=> (row.getDouble(0), row.getDouble(1)))


      dataTraining.unpersist() 
      dataTest.unpersist()

      var metrics= new MulticlassMetrics(predictionsAndLabelsRDD)
      var metrics2 = new BinaryClassificationMetrics(predictionsAndLabelsRDD)

      // total label and unlabeled data from the begining until the end.
      dataLabeledFinal = resultsSSData.dataLabeledFinal + dataLabeledFinal
      dataUnDataLabeledFinal = resultsSSData.dataUnDataLabeledFinal + dataUnDataLabeledFinal
      dataLabeledIni = resultsSSData.dataLabeledIni + dataLabeledIni
      dataUnLabeledIni =resultsSSData.dataUnLabeledIni + dataUnLabeledIni
      iteracionSemiSuper = resultsSSData.iteracionSemiSuper + iteracionSemiSuper 

      acierto = metrics.accuracy+acierto
      auROC = metrics2.areaUnderROC+auROC
      auPR = metrics2.areaUnderPR+auPR 
      f1Score = metrics.fMeasure(1)+f1Score


    }
    acierto = acierto/kFolds
    auROC = auROC/kFolds
    auPR = auPR/kFolds
    f1Score =f1Score/kFolds
    dataLabeledFinal = dataLabeledFinal/kFolds
    dataUnDataLabeledFinal = dataUnDataLabeledFinal/kFolds
    dataLabeledIni = dataLabeledIni/kFolds
    dataUnLabeledIni =dataUnLabeledIni/kFolds
    iteracionSemiSuper = iteracionSemiSuper/kFolds

    (acierto,auROC,auPR,f1Score,dataLabeledIni,dataUnLabeledIni,dataLabeledFinal,dataUnDataLabeledFinal,iteracionSemiSuper)

  }
}

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

import functionsSemiSupervised._

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

// clasificadores Base
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.LinearSVC



val data = Array("coil2000.csv","sonar.csv","spectfheart-1.csv","heart.csv","wisconsinKeel-1.csv")//Array("coil2000.csv")//Array("coil2000.csv","sonar.csv","spectfheart-1.csv","heart.csv","wisconsinKeel-1.csv")
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

  //Featurization
  var dataFeatures=dataDF(posPipeline).columns.diff(Array(dataDF(posPipeline).columns.last))

  var dataFeaturesLabelPipeline= new VectorAssembler().setOutputCol("features").setInputCols(dataFeatures)

  // StringIndexer para pasar el valor categorico a double de la clase , para la features no utilizamos pq ya son doubles. 
  var indexClassPipeline = new StringIndexer().setInputCol(dataDF(posPipeline).columns.last).setOutputCol("label").setHandleInvalid("skip")

  //generamos el pipeline
  featurizationPipeline(posPipeline) = new Pipeline().setStages(Array(
                                                dataFeaturesLabelPipeline,
                                                indexClassPipeline))
} 

// COMMAND ----------

// DBTITLE 1,Comparing Supervised
val instanciaTrainingPipelineDT = new Supervised(new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineRF = new Supervised(new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineNB = new Supervised(new NaiveBayes().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineLR = new Supervised(new LogisticRegression().setFeaturesCol("features").setLabelCol("label"))


val arrayClassifiers:Array[(String,PipelineStage)] = Array(("DT-Sark",instanciaTrainingPipelineDT),
                                       ("LR-Sark",instanciaTrainingPipelineLR),
                                       ("RF-Sark",instanciaTrainingPipelineRF),
                                       ("NB-Sark",instanciaTrainingPipelineNB)
                                       //("LSVM",instanciaTrainingPipelineLSVM) 
                                      )


//parameters
val classifierBase = arrayClassifiers.map(cls =>cls._1)
val percentageLabeled = Array(0.05,0.1,0.15,0.2,0.3,0.6)
val threshold= Array(0.0) // Supervised is n.a 0
val criterion = Array("n.a") // Supervised is n.a
val dataCode =  Array("coil2000","sonar","spectfheart","heart","wisconsin")
var resultsSupervised:Array[DataFrame] = new Array[DataFrame](dataCode.size) 


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

display(results(0).union(results(1)).union(results(2)).union(results(3)).union(results(4)))

// COMMAND ----------

// DBTITLE 1,COMPARING - SelfTraining

// base classifiers
val instTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


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

//display(results(1).union(results(0)))
val results:Array[DataFrame]=resultsSelTraining
//display(results(0).union(results(1)).union(results(2)).union(results(3)).union(results(4)))
display(results(0).union(results(1)).union(results(2)).union(results(3)).union(results(4)))

// COMMAND ----------

// DBTITLE 1,COMPARING - CoTraining
// base classifiers
val instTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


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

val results:Array[DataFrame]=resultsCoTraining
//display(results(0).union(results(1)).union(results(2)).union(results(3)))
//display(results(0).union(results(1)).union(results(2)).union(results(3)).union(results(4)).union(results(5)).union(results(6)).union(results(7)))
display(results(0))

// COMMAND ----------

//display(results(1).union(results(0)))
//val results:Array[DataFrame]=resultsSelTraining
display(results(0).union(results(1)).union(results(2)).union(results(3)).union(results(4)).union(results(5)).union(results(6)).union(results(7)))


// COMMAND ----------

//display(results(1).union(results(0)))
val results:Array[DataFrame]=resultsSelTraining
display(results(0).union(results(1)).union(results(2)).union(results(3)))

// COMMAND ----------

// MAGIC %md
// MAGIC ## Analizing different datasets for BIG DATA
// MAGIC ##### **Supervised** with data label reductions and **Semisupervisado** for Big data 
// MAGIC ##### - BCW
// MAGIC ##### - ADULT
// MAGIC ##### - POKER
// MAGIC ##### - TAXI NY

// COMMAND ----------

// DBTITLE 1,Data Processing & Data Preparation (Featurization) - BCW  
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
val datos="cancerWisconsin.csv"
val lines_datos= sc.textFile(PATH + datos)//.map(x=>x.split(","))

// creamos el DF
val datosDF = spark.read.format("csv")
  .option("sep", ",")
  .option("inferSchema", "true")
  .option("header", "true")
  .load(PATH + datos)
//datosDF.printSchema() 

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//CLEANING y PRE-PROCESADO
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//observamos la ultima linea sin header, vamos a ver que hay en ella:
val valoresDistintos= datosDF.select("_c32").distinct().count() //=1
// Vemos que es valoresDistintos es 1, con lo cual se puede elimnar esta columna, ya que ha sido un error ya que en el ultimo header se ha añadido una ","despues
val datosDF_2=datosDF.drop("_c32")
// donde: 

datosDF_2.printSchema() // dataFrame correcto


//vamos a ver que valores nulos tenemos
val instanciasConNulos=datosDF_2.count() - datosDF_2.na.drop().count()
println("INSTANCIAS CON NULOS")
println(instanciasConNulos) //no hay nulos

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//DISTRIBUCION DE CLASES
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

println("DISTRIBUCION DE CLASE")

val distribucion_Maligno= ((datosDF_2.filter(datosDF_2("diagnosis")==="M").count).toDouble / (datosDF_2.select("diagnosis").count).toDouble)*100
val distribucion_Benigno= ((datosDF_2.filter(datosDF_2("diagnosis")==="B").count).toDouble / (datosDF_2.select("diagnosis").count).toDouble)*100

// Vemos que hay un cierto equilibrio en la distribución de clases
println ("Maligno %")
println (distribucion_Maligno)
println ("Benigno %")
println (distribucion_Benigno)


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//DATA SPLIT  TRAINING & TEST  (el split de los datos de training en 2%, 5% ... se hace posteriormente)
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//dividimos en datos de trainning 75% y datos de test 25%
val dataSplits= datosDF_2.randomSplit(Array(0.75, 0.25),seed = 8L)
val datosDFLabeled_trainning = dataSplits(0)
val datosDFLabeled_test = dataSplits(1)

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//FEATURIZATION -> PREPARAMOS INSTANCIAS
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//seleccionamos solo los nombres de las columnas de features 
val datosDFSinClaseSoloFeatures=datosDF_2.columns.diff(Array("diagnosis"))

//creamos las instancias
//VectorAssembler para generar una array de valores para las features
val datosFeaturesLabelPipeline= new VectorAssembler().setOutputCol("features").setInputCols(datosDFSinClaseSoloFeatures)

// StringIndexer para pasar el valor categorico a double de la clase , para la features no utilizamos pq ya son doubles. 
val indiceClasePipeline = new StringIndexer().setInputCol("diagnosis").setOutputCol("label").setHandleInvalid("skip")

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//PIPELINE FEATURIZATION para Breast Cancer Wisconsin
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//generamos el pipeline
val featurizationPipelineBCW = new Pipeline().setStages(Array(
                                              datosFeaturesLabelPipeline,
                                              indiceClasePipeline)
                                                    )



// COMMAND ----------

// DBTITLE 1,Supervised - BCW  (DT, LR, RF,NB)


val instanciaTrainingPipelineDT = new Supervised(new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineRF = new Supervised(new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineNB = new Supervised(new NaiveBayes().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineLR = new Supervised(new LogisticRegression().setFeaturesCol("features").setLabelCol("label"))
//val instanciaTrainingPipelineLSVM = new Supervised(new LinearSVC().setFeaturesCol("features").setLabelCol("label")


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
val percentageLabeled = Array(0.01,0.2)//Array(0.01,0.05,0.08,0.10,0.20,0.30)
val dataCode = "BCW"


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

val resultsBCWSupervised = SupervisedAndSemiSupervisedResuts(featurizationPipelineBCW, 4,datosDF_2,modelsPipeline,resultsInfo)

display(resultsBCWSupervised)


// COMMAND ----------

// DBTITLE 1,SelfTrainning - BCW (DT, LR, RF ....)


val instTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


val arrayClassifiers:Array[(String,PipelineStage)] = Array(("ST-RF",instTrainingPipelineRF))/*Array(("ST-DT",instTrainingPipelineDT),
                                      ("ST-LR",instTrainingPipelineLR),
                                      ("ST-RF",instTrainingPipelineRF),
                                      ("ST-NB",instTrainingPipelineNB)
                                      )*/

// results for Semisupervised
var SemiSupervisedData = new SemiSupervisedDataResults ()

//parameters
val classifierBase = arrayClassifiers.map(cls =>cls._1)
val percentageLabeled = Array(0.01,0.05,0.08,0.10,0.20)
val threshold= Array(0.8,0.9,0.95)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val kBest= Array(0.05,0.1,0.2,0.3,0.8)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95) //percentage per 1 of unlabeled total data that should be labeled at the end
val dataCode = "BCW"
val criterion = Array("threshold","kBest")

//template dataFrame results according the parameters
val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,classifierBase, percentageLabeled,threshold,kBest,criterion)

// final pipeline models with all the configurations (parameters)
val modelsPipeline = pipelineModelsSelfTraining(threshold,kBest,percentageLabeled,SemiSupervisedData,arrayClassifiers,criterion)

// dataframe of final results
val resultsBCWST = SupervisedAndSemiSupervisedResuts (featurizationPipelineBCW, 4,datosDF_2,modelsPipeline,resultsInfo,SemiSupervisedData)

display(resultsBCWST)

// COMMAND ----------

// DBTITLE 1,CoTraining -BCW


val instTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


val arrayClassifiers:Array[(String,PipelineStage)] = Array(("CT-DT", instTrainingPipelineDT),("CT-RF",instTrainingPipelineRF))
//Array(("CT-DT",instTrainingPipelineDT),("CT-LR",instTrainingPipelineLR),("CT-RF",instTrainingPipelineRF),("CT-NB",instTrainingPipelineNB))

// results for Semisupervised
var SemiSupervisedData = new SemiSupervisedDataResults ()

//parameters
val classifierBase = arrayClassifiers.map(cls =>cls._1)
val percentageLabeled = Array(0.05,0.1,0.2)//Array(0.01,0.05,0.08,0.10,0.20,0.30)
val threshold= Array(0.9)//Array(0.7,0.8,0.9,0.95)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val kBest= Array(0.1)//Array(0.05,0.1,0.2,0.3,0.5)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95) //percentage per 1 of unlabeled total data that should be labeled at the end
val dataCode = "BCW"
val criterion = Array("threshold")//,"kBest")

//template dataFrame results according the parameters
val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,classifierBase, percentageLabeled,threshold,kBest,criterion)

// final pipeline models with all the configurations (parameters)
val modelsPipeline = pipelineModelsCoTraining(threshold,kBest,percentageLabeled,SemiSupervisedData,arrayClassifiers,criterion)

// dataframe of final results
val resultsBCWST = SupervisedAndSemiSupervisedResuts (featurizationPipelineBCW, 4,datosDF_2,modelsPipeline,resultsInfo,SemiSupervisedData)

display(resultsBCWST)

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
val percentageLabeled =Array(0.0001)//Array(0.0001,0.001,0.01,0.05,0.1,0.3)
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

display(resultsPokerSupervised)


// COMMAND ----------

// DBTITLE 1,SelfTrainning - POKER (DT, LR, RF ....)

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
val percentageLabeled = Array(0.0001)//Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val threshold= Array(0.8)//Array(0.8,0.9,0.95)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val kBest= Array(0.1)//Array(0.05,0.1,0.2,0.3,0.8)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95) //percentage per 1 of unlabeled total data that should be labeled at the end
val dataCode = "POKER"
val criterion = Array("threshold")//Array("threshold","kBest")

//template dataFrame results according the parameters
val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,classifierBase, percentageLabeled,threshold,kBest,criterion)

// final pipeline models with all the configurations (parameters)
val modelsPipeline = pipelineModelsSelfTraining(threshold,kBest,percentageLabeled,SemiSupervisedData,arrayClassifiers,criterion)

// dataframe of final results
val resultsPokerST = SupervisedAndSemiSupervisedResuts (featurizationPipelinePoker, 4,datosDFNew,modelsPipeline,resultsInfo,SemiSupervisedData)

display(resultsPokerST)

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
val percentageLabeled =Array(0.0001)//Array(0.0001,0.001,0.01,0.05,0.1,0.3)//(0.01,0.05,0.10,0.30) //Array(0.01)
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

val resultsTaxiNYSupervised = SupervisedAndSemiSupervisedResuts(featurizationPipelineNY, 4,datosDF_NY,modelsPipeline,resultsInfo)

display(resultsTaxiNYSupervised)


// COMMAND ----------

// DBTITLE 1,SelfTrainning - TAXI NY (DT, LR, RF ....)

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
val percentageLabeled = Array(0.0001)//Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val threshold= Array(0.8)//Array(0.8,0.9,0.95)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val kBest= Array(0.1)//Array(0.05,0.1,0.2,0.3,0.8)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95) //percentage per 1 of unlabeled total data that should be labeled at the end
val dataCode = "TXNY"
val criterion = Array("threshold")//Array("threshold","kBest")

//template dataFrame results according the parameters
val resultsInfo=generadorDataFrameResultadosSemiSuper(dataCode,classifierBase, percentageLabeled,threshold,kBest,criterion)

// final pipeline models with all the configurations (parameters)
val modelsPipeline = pipelineModelsSelfTraining(threshold,kBest,percentageLabeled,SemiSupervisedData,arrayClassifiers,criterion)

// dataframe of final results
val resultsTaxiNYST = SupervisedAndSemiSupervisedResuts (featurizationPipelineNY, 4,datosDF_NY,modelsPipeline,resultsInfo,SemiSupervisedData)

display(resultsTaxiNYST)

// COMMAND ----------

// DBTITLE 1,Pruebas -SSC - SelfTrainning - 1



import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.linalg.DenseVector
import sqlContext.implicits._
import org.apache.spark.sql.functions.udf

/*
 * Self-Training model realization
 * Works with dataframes
 * Constructor takes uid (uniq ID) and classifier (base learner)
 */

class SelfTrainingClassifier2 [
  FeatureType,
  E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
] (
    val uid: String,
    val baseClassifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
    val porcentajeLabeled:Double =0.1,
    val threshold:Double=0.8,
    val criterion:String= "threshold",
    val kBest:Int=10,
    val maxIter:Int=10
  ) extends org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M] with Serializable {
    
  //uid
  def this(classifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]) =
    this(Identifiable.randomUID("selfTrainning"), classifier)
  
  //hacemos el split entre los datos de entrenamiento L y U labeled y unlabeler respecticamente en funcion del porcentaje
  def resplit(data:org.apache.spark.sql.Dataset[_],porcentajeEtiquetado:Double):Array[org.apache.spark.sql.Dataset[_]] ={
    
    val dataSp=  data.randomSplit(Array(porcentajeEtiquetado, 1-porcentajeEtiquetado),seed = 11L)
    val dataLabeled = dataSp(0)
    val dataUnLabeled = dataSp(1)
    Array(dataLabeled,dataUnLabeled)
    
  }

  
  def train2(dataset: org.apache.spark.sql.Dataset[_]): M = {//DataFrame = {
    var iter = 1
    //udf para coger el valor mas elevado de la array de probabilidades
    val max = udf((v: org.apache.spark.ml.linalg.Vector) => v.toArray.max)
    //splitamos con el % seleccionado 
    val dataSplited = resplit(dataset,porcentajeLabeled)
    var dataLabeled = dataSplited(0).toDF // las reutilizaremos
    var dataUnLabeled = dataSplited(1).toDF
    var countDataLabeled = dataLabeled.count()
    var countDataUnLabeled = dataUnLabeled.count()

    // total data inicial   
    println("iterancion: 0" )
    println ("total labeled Inicial " + countDataLabeled)
    println ("total unlabeled Inicial " + countDataUnLabeled)

    // generamos modelo y predecimos las unlabeled data
    var modeloIterST = baseClassifier.fit(dataLabeled)
    var prediIterST = modeloIterST.transform(dataUnLabeled)

    while ((iter<=maxIter) && (countDataUnLabeled>0)){

      val modificacionPrediccion=prediIterST.withColumn("probMax", max($"probability"))
      val seleccionEtiquetasMayoresThreshold=modificacionPrediccion.filter(modificacionPrediccion("probMax")>threshold)
      val seleccionEtiquetasMenoresIgualThreshold=modificacionPrediccion.filter(modificacionPrediccion("probMax")<=threshold)
      // selecionamos las features y las predicciones y cambiamos el nombre de las predicciones por labels
      val newLabeledFeaturesLabels = seleccionEtiquetasMayoresThreshold.select ("features","prediction").withColumnRenamed("prediction","label")
      val newUnLabeledFeaturesLabels = seleccionEtiquetasMenoresIgualThreshold.select ("features","prediction").withColumnRenamed("prediction","label")

      println("tamaño filtrado por Threshold (new Labeled data): " + seleccionEtiquetasMayoresThreshold.count)
      dataLabeled = dataLabeled.union(newLabeledFeaturesLabels)
      dataUnLabeled = newUnLabeledFeaturesLabels
      countDataUnLabeled = dataUnLabeled.count()
      countDataLabeled = dataLabeled.count()
      println("iterancion: " + iter)
      println ("max iter: " + maxIter)
      println ("total labeled: "+iter +" tamaño: " + countDataLabeled)
      println ("total unlabeled: "+iter +" tamaño: " + countDataUnLabeled)


      if (countDataUnLabeled>0 && iter<=maxIter ){
        println ("Trainning... Next Iteration")
        modeloIterST = baseClassifier.fit(dataLabeled)
        prediIterST = modeloIterST.transform(dataUnLabeled)
        iter = iter+1
      }
      else{ //final del ciclo
        
        modeloIterST = baseClassifier.fit(dataLabeled)
        iter = maxIter
      }
    }
    //prediIterST
    modeloIterST
    // if trheshould
    //val rdd = predictions.rdd.filter(x=> x(x.fieldIndex("probability")).asInstanceOf[DenseVector].toArray.max > thresholdTrust)
      // mirar la probabilidad y seleccionar aquellos con la probabilidad igual o mayor
      // unir los datos con la probabilidad mayor (datos anteriores) a labeledData
      // eliminar los datos de el conjutno de unlabeled data
   // ......


  //}

  }
  
  
  

  def train(dataset: org.apache.spark.sql.Dataset[_]): M  = {
  //def train(dataset: org.apache.spark.sql.Dataset[_]): DataFrame  = {
    //val data = dataset.select(data.labelCol, data.featuresCol).cache()
    //this.syncPipeline()
    val dataSplited = resplit(dataset,porcentajeLabeled)
    var dataLabeled = dataSplited(0) // las reutilizaremos
    var dataUnLabeled = dataSplited(1)
    var countDataLabeled = dataLabeled.count()
    var countDataUnLabeled = 0//dataLabeled.count() //0
    var iter = 11//0 //11
    //while ((iter<maxIter) | (countDataUnLabeled>0)){
    print("iterancion")
    println(iter)
    val modeloIterST = baseClassifier.fit(dataLabeled)
    val prediIterST = modeloIterST.transform(dataUnLabeled)
    modeloIterST
  }
  
  def train3(dataset: org.apache.spark.sql.Dataset[_]): M = {//DataFrame = {
  //def train(dataset: org.apache.spark.sql.Dataset[_]): DataFrame  = {
    //val data = dataset.select(data.labelCol, data.featuresCol).cache()
    //this.syncPipeline()
    val dataSplited = resplit(dataset,porcentajeLabeled)
    var dataLabeled = dataSplited(0) // las reutilizaremos
    var dataUnLabeled = dataSplited(1)
    var countDataLabeled = dataLabeled.count()
    var countDataUnLabeled = 0//dataLabeled.count() //0
    var iter = 11//0 //11
    //while ((iter<maxIter) | (countDataUnLabeled>0)){
    print("iterancion")
    println(iter)
    val modeloIterST = baseClassifier.fit(dataLabeled)
    val prediIterST = modeloIterST.transform(dataUnLabeled)
    modeloIterST
  }
  

  override def copy(extra: org.apache.spark.ml.param.ParamMap):E = defaultCopy(extra)
}





// COMMAND ----------


import com.github.import com.github.mrpowers.spark.daria.sql._



// COMMAND ----------

// DBTITLE 1,Pruebas -SSC - SelfTrainning - 2
//val featBCW = featurizationPipelineBCW.fit(datosDFLabeled_trainning).transform(datosDFLabeled_trainning)
//val featBCW2=featBCW.select("features","label")

val featAdult =featurizationPipelineAdult.fit(datosDFLabeled_trainningAdult).transform(datosDFLabeled_trainningAdult)
val featAdult2=featAdult.select("features","label")


//val modelInstancia = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val modelInstancia = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
modelInstancia.setMaxBins(42)

//val modelInstaPipeline = new Pipeline().setStages(Array(modelInstancia))


val SemiSupeModelInsta = new SelfTrainingClassifier2(modelInstancia)

val semi = new SelfTraining(modelInstancia).setPorcentaje(0.1).setThreshold(0.8)


val pipelineTestSSC = new Pipeline().setStages(Array(featurizationPipelineAdult,semi))


//datosDFLabeled_testAdult
//datosDFLabeled_trainningAdult
val modelosalidaSemiSC = pipelineTestSSC.fit(datosDFLabeled_trainningAdult)


println ("datosLabeledFinal: " + semi.getDataLabeledFinal())
println ("datosLabeledInicial: " + semi.getDataLabeledIni())                                   
                                      

//val modelosalidaSemiSC = semi.fit(featAdult)

//val modelosalidaSemiSC = SemiSupeModelInsta.train2(featAdult2)
val modelosalidaSuperC = SemiSupeModelInsta.train3(featAdult2)


val dataSpT=  featAdult2.randomSplit(Array(0.75, 0.25),seed = 8L)
val salidaDF2 =  modelInstancia.fit(dataSpT(0)).transform(dataSpT(1))


// SemiSupervisado
val salidaSemiSC =modelosalidaSemiSC.transform(datosDFLabeled_testAdult)
val resultadosSSC=salidaSemiSC.select("prediction", "label")                                                                                        
val predictionsAndLabelsRDD_SSC=resultadosSSC.rdd.map(row=> (row.getDouble(0), row.getDouble(1)))
val metricsSSC= new MulticlassMetrics(predictionsAndLabelsRDD_SSC)
val aciertoSSC = metricsSSC.accuracy
println ("SemiSupervisado 0.75 y despues %: " +aciertoSSC)

//Supervisado
val salidaSuperC =modelosalidaSuperC.transform(dataSpT(1))
val resultadosSC=salidaSuperC.select("prediction", "label")                                                                                        
val predictionsAndLabelsRDD_SC=resultadosSC.rdd.map(row=> (row.getDouble(0), row.getDouble(1)))
val metricsSC= new MulticlassMetrics(predictionsAndLabelsRDD_SC)
val aciertoSC = metricsSC.accuracy
println ("Supervisado 0.75 y despues %: " +aciertoSC)



val resultadosN=salidaDF2.select("prediction", "label")                                                                                        
val predictionsAndLabelsRDD_N=resultadosN.rdd.map(row=> (row.getDouble(0), row.getDouble(1)))
val metricsN= new MulticlassMetrics(predictionsAndLabelsRDD_N)
val aciertoN = metricsN.accuracy
println ("Supervisado 0.75: " +aciertoN)


//modelInstancia.explainParams