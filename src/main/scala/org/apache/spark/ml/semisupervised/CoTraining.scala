package org.apache.spark.ml.semisupervised

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{SparkSession, SQLContext}


/* Class --> CoTraining*/

class CoTraining [
  FeatureType,
  E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
] (
    val uid: String,
    val baseClassifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  ) extends org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M] with Serializable {
  
  var porcentajeLabeled: Double = 1.0
  var threshold: Double = 0.7
  var maxIter: Int = 3
  var criterion: String = "threshold"
  var kBest: Double = 1.0 // percentage
  var countDataLabeled: Long = _
  var countDataLabeled_1: Long = _
  var countDataLabeled_2: Long = _
  var countDataUnLabeled_1: Long = 1
  var countDataUnLabeled_2: Long = 1
  var countDataUnLabeled: Long = _
  var dataLabeledIni: Long =_
  var dataUnLabeledIni: Long = _
  var dataLabeled: Long = _
  var iter: Int = 0
  var columnNameNewLabels: String = "label"
  var resultsSelfTrainingData: SemiSupervisedDataResults = new SemiSupervisedDataResults ()
  var numberOfkBest: Int = 0
  var modeloIterST: M= _

  //uid
  def this(classifier1: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]) =
    this(Identifiable.randomUID("CoTrainning"),classifier1)
           
  ///////////////////////////////////////////////////////////////////////////
  // Setters
 ///////////////////////////////////////////////////////////////////////////
 
    //Set columnLabels --> Labeled and Unlabeled
  def setSemiSupervisedDataResults(semiSupervisedResults: SemiSupervisedDataResults) = {
    resultsSelfTrainingData = semiSupervisedResults
    this
  }
   
  //Set columnLabels --> Labeled and Unlabeled
  def setColumnLabelName(nameColumn: String) = {
    columnNameNewLabels = nameColumn
    this
  }
  
  // set porcentaje
  def setPorcentaje(porcentaje: Double) = {
    porcentajeLabeled = porcentaje
    this  
  }
  
  def setThreshold(thres: Double) = {
    threshold = thres
    this
  }
  
  // maxIter
  def setMaxITer(maIter: Int) = {
    maxIter = maIter
    this   
  }
  
  // criterion
  def setCriterion(cri: String) = {
    criterion = cri
    this
  }
  
  //kBest
  def setKbest(kb: Double) = {
    kBest = kb
    this
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // Getters
  ///////////////////////////////////////////////////////////////////////////
           
  def getDataLabeledFinal(): Long = {
    countDataLabeled
  }
  
  def getUnDataLabeledFinal(): Long = {
    countDataUnLabeled
  }  
  
  def getDataLabeledIni(): Long = {
    dataLabeledIni
  }
  
   def getUnDataLabeledIni(): Long = {
     dataUnLabeledIni
  }  
  
  def getIter(): Int = {
     iter
  }  
  

  ///////////////////////////////////////////////////////////////////////////
  // Train
  ///////////////////////////////////////////////////////////////////////////
  
  def train(dataset: org.apache.spark.sql.Dataset[_]): M = {
    iter = 1
    val sql = SparkSession.builder().getOrCreate()
    import sql.implicits._
    
    //udf to get he max value from probabilisti array
    val max = udf((v: org.apache.spark.ml.linalg.Vector) => v.toArray.max)
    var dataUnLabeled = dataset.filter(dataset(columnNameNewLabels).isNaN).toDF.cache()
    var dataLabeled = dataset.toDF.exceptAll(dataUnLabeled).cache()
    
    //get the data labeled and unlabeled initial
    dataLabeledIni = dataLabeled.count()
    dataUnLabeledIni = dataUnLabeled.count()
    
    //split data in two datasets.
    val dataSplitsLabel =  dataLabeled.randomSplit(Array(0.5, 0.5),seed = 8L)
    var dataLabeled_1 = dataSplitsLabel(0)
    var dataLabeled_2 = dataSplitsLabel(1)
    val dataSplitsUnLabel =  dataUnLabeled.randomSplit(Array(0.5, 0.5),seed = 8L)
    var dataUnLabeled_1 = dataSplitsUnLabel(0)
    var dataUnLabeled_2 = dataSplitsUnLabel(1)
    
    //selection features and labels
    dataUnLabeled = dataUnLabeled .select("features", "label")
    dataUnLabeled_1 = dataUnLabeled_1 .select("features", "label").cache()
    dataUnLabeled_2 = dataUnLabeled_2 .select("features", "label").cache()
    dataLabeled_1 = dataLabeled_1.select("features", "label").cache()
    dataLabeled_2 = dataLabeled_2.select("features", "label").cache()
    countDataLabeled_1 = dataLabeled_1.count()
    countDataLabeled_2 = dataLabeled_2.count()
    countDataUnLabeled = dataUnLabeled.count()    
    var modeloIterST_1 = baseClassifier.fit(dataLabeled_1)
    var prediIterST_1 = modeloIterST_1.transform(dataUnLabeled_1)
    var modeloIterST_2 = baseClassifier.fit(dataLabeled_2)
    var prediIterST_2 = modeloIterST_2.transform(dataUnLabeled_2)
    
    if (criterion == "threshold"){  
      while ((iter<maxIter) && (countDataUnLabeled_1 > 0) &&(countDataUnLabeled_2 > 0)){
        
        // model 1
        var modificacionPrediccion_1 = prediIterST_1.withColumn("probMax", max($"probability"))
        var labelsHigherOfThreshold_1 = modificacionPrediccion_1.filter(modificacionPrediccion_1("probMax") > threshold)
        dataUnLabeled_1 = modificacionPrediccion_1.filter(modificacionPrediccion_1("probMax") <= threshold).select ("features", "prediction").withColumnRenamed("prediction", "label")
        var newLabeledFeaturesLabels_1 = labelsHigherOfThreshold_1.select ("features", "prediction").withColumnRenamed("prediction", "label")

        //model 2
        var modificacionPrediccion_2 = prediIterST_2.withColumn("probMax", max($"probability"))
        var labelsHigherOfThreshold_2 = modificacionPrediccion_2.filter(modificacionPrediccion_2("probMax")>threshold)
        dataUnLabeled_2 = modificacionPrediccion_2.filter(modificacionPrediccion_2("probMax") <= threshold).select ("features", "prediction").withColumnRenamed("prediction", "label")
        var newLabeledFeaturesLabels_2 = labelsHigherOfThreshold_2.select ("features", "prediction").withColumnRenamed("prediction", "label")
        dataLabeled_1 = dataLabeled_1.unionAll(newLabeledFeaturesLabels_2)
        dataLabeled_2 = dataLabeled_2.unionAll(newLabeledFeaturesLabels_1)
        
        //count dataLabeled and Unlabeled in each iteration
        countDataUnLabeled_1 = dataUnLabeled_1.count()
        countDataUnLabeled_2 = dataUnLabeled_2.count()
        countDataLabeled_1 = dataLabeled_1.count()
        countDataLabeled_2 = dataLabeled_2.count()

        if ((countDataUnLabeled_1 > 0) && (countDataUnLabeled_2 > 0) && (iter < maxIter) ){          
          modeloIterST_1 = baseClassifier.fit(dataLabeled_1)
          prediIterST_1 = modeloIterST_1.transform(dataUnLabeled_1)
          modeloIterST_2 = baseClassifier.fit(dataLabeled_2)
          prediIterST_2 = modeloIterST_2.transform(dataUnLabeled_2)
          iter = iter + 1
        }        
      }
      
    //unpersist
    dataLabeled_1.unpersist()
    dataLabeled_2.unpersist()
    dataUnLabeled_1.unpersist()
    dataUnLabeled_2.unpersist()
    dataUnLabeled.unpersist()
    }
    
    else if (criterion == "kBest"){
      numberOfkBest = ((kBest * countDataUnLabeled) / (maxIter - 1)).round.toInt
       while ((iter < maxIter) && (countDataUnLabeled_1 > 0) &&(countDataUnLabeled_2 > 0)){
        
        // model 1
        var modificacionPrediccion_1 = prediIterST_1.withColumn("probMax", max($"probability"))
        var newLabeledFeaturesLabelsHigherProb_1  = modificacionPrediccion_1.sort(col("probMax").desc).limit(numberOfkBest)
        var newUnLabeledFeaturesLabels_1 =  modificacionPrediccion_1.exceptAll(newLabeledFeaturesLabelsHigherProb_1).select ("features", "prediction").withColumnRenamed("prediction", "label")
        var newLabeledFeaturesLabels_1  = newLabeledFeaturesLabelsHigherProb_1.select ("features", "prediction").withColumnRenamed("prediction", "label")  
        var dataUnLabeled_1 = newUnLabeledFeaturesLabels_1
        
        // model 2
        var modificacionPrediccion_2 = prediIterST_2.withColumn("probMax", max($"probability"))
        var newLabeledFeaturesLabelsHigherProb_2  = modificacionPrediccion_2.sort(col("probMax").desc).limit(numberOfkBest)
        var newUnLabeledFeaturesLabels_2 =  modificacionPrediccion_2.exceptAll(newLabeledFeaturesLabelsHigherProb_2).select ("features", "prediction").withColumnRenamed("prediction", "label")
        var newLabeledFeaturesLabels_2  = newLabeledFeaturesLabelsHigherProb_2.select ("features", "prediction").withColumnRenamed("prediction", "label")        
        var dataUnLabeled_2 = newUnLabeledFeaturesLabels_2
        dataLabeled_1 = dataLabeled_1.unionAll(newLabeledFeaturesLabels_2)
        dataLabeled_2 = dataLabeled_2.unionAll(newLabeledFeaturesLabels_1)
        countDataUnLabeled_1 = dataUnLabeled_1.count()
        countDataUnLabeled_2 = dataUnLabeled_2.count()
        countDataLabeled_1 = dataLabeled_1.count()
        countDataLabeled_2 = dataLabeled_2.count()

        if ((countDataUnLabeled_1 > 0) && (countDataUnLabeled_2 > 0) && (iter < maxIter) ){
          modeloIterST_1 = baseClassifier.fit(dataLabeled_1)
          prediIterST_1 = modeloIterST_1.transform(dataUnLabeled_1)
          modeloIterST_2 = baseClassifier.fit(dataLabeled_2)
          prediIterST_2 = modeloIterST_2.transform(dataUnLabeled_2)
          iter = iter + 1
        }        
      }
      
      //unpersist
      dataLabeled_1.unpersist()
      dataLabeled_2.unpersist()
      dataUnLabeled_1.unpersist()
      dataUnLabeled_2.unpersist()
      dataUnLabeled.unpersist()
    }
    
    // load the semisupervised results regarding the labeled and unlabeled data using the SemiSupervisedDataResults class
    resultsSelfTrainingData.dataLabeledFinal = countDataLabeled_1 + countDataLabeled_2
    resultsSelfTrainingData.dataUnDataLabeledFinal = countDataUnLabeled_1 + countDataUnLabeled_2
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
  override def copy(extra: org.apache.spark.ml.param.ParamMap): E = defaultCopy(extra)
}
