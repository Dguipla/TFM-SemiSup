
package org.apache.spark.ml.semisupervised
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.linalg.DenseVector
import sqlContext.implicits._
import org.apache.spark.sql.functions.udf

/** Class --> SELF TRAINING */ 

class SelfTraining [
  FeatureType,
  E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
] (
    val uid: String,
    val baseClassifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],    
  ) extends org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M] with Serializable {
  
  var porcentajeLabeled: Double = 1.0
  var threshold: Double = 0.7
  var maxIter: Int = 7
  var criterion: String = "threshold"
  var kBest: Double = 1.0 /** percentage*/
  var countDataLabeled: Long = _
  var countDataUnLabeled: Long = _
  var dataLabeledIni: Long = _
  var dataUnLabeledIni: Long = _
  var iter: Int = 0
  var columnNameNewLabels: String = "labelSelection"
  var resultsSelfTrainingData: SemiSupervisedDataResults = _
  var numberOfkBest: Int = 0

  //uid 
  def this(classifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]) =
    this(Identifiable.randomUID("selfTrainning"),classifier)

  ///////////////////////////////////////////////////////////////////////////
  // Setters
  ///////////////////////////////////////////////////////////////////////////
 
  //Set columnLabels --> Labeled and Unlabeled
  def setSemiSupervisedDataResults(semiSupervisedResults: SemiSupervisedDataResults) = {
    resultsSelfTrainingData = semiSupervisedResults
    this
  }
  
  // Set columnLabels --> Labeled and Unlabeled
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
  
  //maxIter
  def setMaxITer(maIter: Int) = {
    maxIter = maIter
    this
  }
  
  //criterion
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
    
    //udf to get he max value from probabilisti array
    val max = udf((v: org.apache.spark.ml.linalg.Vector) => v.toArray.max)
    var dataUnLabeled = dataset.filter(dataset(columnNameNewLabels).isNaN).toDF.cache()
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
      while ((iter < maxIter) && (countDataUnLabeled > 0)){  
        var modificacionPrediccion = prediIterST.withColumn("probMax", max($"probability"))
        var labelsHigherOfThreshold = modificacionPrediccion.filter(modificacionPrediccion("probMax")>threshold)
        var labelsLowerOfThreshold = modificacionPrediccion.filter(modificacionPrediccion("probMax")<=threshold)
        // get features and predictions and change the name from prediction to label in order to add as new data labeled
        var newLabeledFeaturesLabels = labelsHigherOfThreshold.select ("features", "prediction").withColumnRenamed("prediction", "label")
        var newUnLabeledFeaturesLabels = labelsLowerOfThreshold.select ("features", "prediction").withColumnRenamed("prediction", "label")
        dataLabeled = dataLabeled.union(newLabeledFeaturesLabels).cache()
        dataUnLabeled = newUnLabeledFeaturesLabels.cache()
        countDataUnLabeled = dataUnLabeled.count()
        countDataLabeled = dataLabeled.count()
        if (countDataUnLabeled > 0 && iter < maxIter ){
          modeloIterST = baseClassifier.fit(dataLabeled)
          prediIterST = modeloIterST.transform(dataUnLabeled)
          iter = iter + 1  
        }
        //final
        else{
          modeloIterST = baseClassifier.fit(dataLabeled)
        }
        dataLabeled.unpersist()
        dataUnLabeled.unpersist()
      } 
    }
    
    else if (criterion == "kBest"){
      numberOfkBest = ((kBest * countDataUnLabeled)/(maxIter-1)).round.toInt
      
      while ((iter < maxIter) && (countDataUnLabeled > 0)){
        var modificacionPrediccion = prediIterST.withColumn("probMax", max($"probability"))
        var newLabeledFeaturesLabelsHigherProb  = modificacionPrediccion.sort(col("probMax").desc).limit(numberOfkBest)
        var newUnLabeledFeaturesLabels =  modificacionPrediccion.exceptAll(newLabeledFeaturesLabelsHigherProb).select ("features", "prediction").withColumnRenamed("prediction", "label")
        var newLabeledFeaturesLabels  = newLabeledFeaturesLabelsHigherProb.select ("features", "prediction").withColumnRenamed("prediction", "label")          
        dataLabeled = dataLabeled.union(newLabeledFeaturesLabels).cache()
        dataUnLabeled = newUnLabeledFeaturesLabels.cache()
        countDataUnLabeled = dataUnLabeled.count()
        countDataLabeled = dataLabeled.count()
        if (countDataUnLabeled > 0 && iter < maxIter ){
          modeloIterST = baseClassifier.fit(dataLabeled)
          prediIterST = modeloIterST.transform(dataUnLabeled)
          iter = iter+1
        }
        //final
        else{ 
          modeloIterST = baseClassifier.fit(dataLabeled)
        }
        dataLabeled.unpersist()
        dataUnLabeled.unpersist()
      }
    }

    // load the semisupervised results regarding the labeled and unlabeled data using the SemiSupervisedDataResults class
    resultsSelfTrainingData.dataLabeledFinal = countDataLabeled
    resultsSelfTrainingData.dataUnDataLabeledFinal = countDataUnLabeled
    resultsSelfTrainingData.dataLabeledIni =  dataLabeledIni
    resultsSelfTrainingData.dataUnLabeledIni = dataUnLabeledIni
    resultsSelfTrainingData.iteracionSemiSuper = iter
    
    // Final model
    modeloIterST
  }
  override def transformSchema(schema: StructType): StructType = schema
  override def copy(extra: org.apache.spark.ml.param.ParamMap): E = defaultCopy(extra)
}
