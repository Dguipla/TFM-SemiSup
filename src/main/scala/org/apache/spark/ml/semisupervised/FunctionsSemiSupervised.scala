package org.apache.spark.ml.semisupervised
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
//import spark.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.PipelineStage
//import scala.util.control.Breaks._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils.kFold
import org.apache.spark.sql.functions._
import org.apache.spark.ml.PipelineStage
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{SparkSession, SQLContext}

object FunctionsSemiSupervised extends Serializable {
  
  /**
  *creation PipelineStatge to convert features from categorical to continuos
  *output array[PipelineStage]
  */
  
  def indexStringColumnsStagePipeline(df: DataFrame, cols: Array[String]): (Pipeline, Array[String]) = {
    var intermedioStages: Array[(PipelineStage)] = new Array[(PipelineStage)](cols.size)
    var posicion = 0
    for(col <-cols) {
      val si = new StringIndexer().setInputCol(col).setOutputCol(col+"-num")
      intermedioStages(posicion) = si.setHandleInvalid("keep")
      posicion = posicion + 1
    }
    val output = new Pipeline().setStages(intermedioStages)
    (output,df.columns.diff(cols))
  }

  /**
  * pipelines SelfTraining Creation
  *percentatge, threshold, Classifier, pipeline, i.e: (0.01,0.4,ST-DT,pipeline_87c711c3e400)
  *                                                    (0.01,0.4,ST-LB,pipeline_87c711c3e400)
  *                                                    ...
  */
  
  def pipelineModelsSelfTraining [
      FeatureType,
      E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
      M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
    ] (threshold: Array[Double],
       kBest: Array[Double],
       percentage: Array[Double],
       resultsSemiSupData: SemiSupervisedDataResults,
       arrayClasiInstanceModel: Array[(String, org.apache.spark.ml.PipelineStage)],
       criterion: Array[String],
       iterations: Int = 7): Array[(String, Array[(Double, Array[(Double, Array[(String, org.apache.spark.ml.Pipeline)])])])]= 
  {
      criterion.map(crit =>(crit, percentage.map(per =>(per, 
                                                     if (crit == "threshold")
                                                     {
                                                       threshold.map(th => (th, arrayClasiInstanceModel.map(clasi => (clasi._1, new Pipeline().setStages(Array(new SelfTraining(clasi._2.asInstanceOf[E])
                                                                                                                                      .setThreshold(th)
                                                                                                                                      .setCriterion(crit)
                                                                                                                                      .setMaxITer(iterations)
                                                                                                                                      .setSemiSupervisedDataResults(resultsSemiSupData)))))))
                                                     }
                                                     
                                                     // kBest
                                                     else 
                                                     {
                                                       kBest.map(kb => (kb, arrayClasiInstanceModel.map(clasi => (clasi._1, new Pipeline().setStages(Array(new SelfTraining(clasi._2.asInstanceOf[E])
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
    ] (threshold: Array[Double],
       kBest: Array[Double],
       percentage: Array[Double],
       resultsSemiSupData: SemiSupervisedDataResults,
       arrayClasiInstanceModel: Array[(String, org.apache.spark.ml.PipelineStage)],
       criterion: Array[String],
       iterations: Int = 7): Array[(String, Array[(Double, Array[(Double, Array[(String, org.apache.spark.ml.Pipeline)])])])]= 
  {
      criterion.map(crit => (crit, percentage.map(per => (per, 
                                                     if (crit == "threshold")
                                                     {
                                                       threshold.map(th => (th, arrayClasiInstanceModel.map(clasi => (clasi._1, new Pipeline()
                                                                                                                 .setStages(Array(new CoTraining(clasi._2.asInstanceOf[E])
                                                                                                                                      .setThreshold(th)
                                                                                                                                      .setCriterion(crit)
                                                                                                                                      .setMaxITer(iterations)
                                                                                                                                      .setSemiSupervisedDataResults(resultsSemiSupData)))))))
                                                     }
                                                     
                                                     // kBest
                                                     else 
                                                     {
                                                       kBest.map( kb =>(kb, arrayClasiInstanceModel.map(clasi => (clasi._1, new Pipeline()
                                                                                                             .setStages(Array(new CoTraining(clasi._2.asInstanceOf[E])
                                                                                                                                      .setKbest(kb)
                                                                                                                                      .setCriterion(crit)
                                                                                                                                      .setMaxITer(iterations)
                                                                                                                                      .setSemiSupervisedDataResults(resultsSemiSupData)))))))
                                                     }
                                                    ))))

  }



  /**
  *restults DataFrame template
  * Structure:
  * |data| clasifi| porcentaje| threshold| iter| LabeledInicial| UnlabeledInicial| LabeledFinal| UnlabeledFinal| porEtiFinal| Acc| AUC| PR| F1Score|
  * +----+--------+----------+-----------+-----+---------------+-----------------+-------------+---------------+------------+----+----+----+-------+
  * | BCW|   ST-DT|      0.001|       0.6|    2|            130|           100600|        98000|           2600|       97.4 | x.x| x.x| x.x|    x.x|
  * | BCW|   ST-DT|      0.005|       0.6|    3|            270|           100460|        96000|           2800|       97.21| x.x| x.x| x.x|    x.x|
  * .... 
  */
  def generadorDataFrameResultadosSemiSuper(data:String,
                                            classifiers: Array[String],
                                            percentage: Array[Double],
                                            threshold: Array[Double]=Array(0.0),
                                            kBest: Array[Double] = Array(0.0),
                                            criterion: Array[String] = Array("n.a"),
                                           ):DataFrame = 
  {
    var seqValores = Seq[(String,String,String,Double, Double,Int,Int,Long,Long,Long,Double,Double, Double,Double, Double)]() 
    var posicion = 0
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
    var thresholdOrKbest:Array[Double] = Array(0.0)
    for (crit <- criterion){
      if (crit == "kBest"){
        thresholdOrKbest = kBest
      }
      else if (crit == "threshold"){
        thresholdOrKbest =threshold
      }
      for(posClasi <- classifiers) {
        for(posPorce <- percentage){
          for (posThreshold <- thresholdOrKbest){
            seqValores = seqValores :+ (data, posClasi, crit, posPorce, posThreshold, 0, 0, 0.toLong, 0.toLong, 0.toLong, 0.00, 0.00, 0.00, 0.00, 0.00)
          }
        }
      }
    }
    // DataFrame creation
    spark.createDataFrame(seqValores).toDF("data",
                                           "classifier",
                                           "criterion",
                                           "percentageLabeled",
                                           "thresholdOrKBest",
                                           "iteration",
                                           "LabeledInitial",
                                           "UnLabeledInitial",
                                           "LabeledFinal",
                                           "UnLabeledFinal",
                                           "percentageLabeledFinal",
                                           "accuracy",
                                           "AUC",
                                           "PR",
                                           "F1score")
  }

  /**
  * results calculation
  */
  def SupervisedAndSemiSupervisedResuts  (featurization:Pipeline,
       kfold: Int,
       data: DataFrame,
       modelsPipeline: Array[(String, Array[(Double, Array[(Double, Array[(String, org.apache.spark.ml.Pipeline)])])])],
       info: DataFrame,
       resultSemiSupervisedData: SemiSupervisedDataResults = new SemiSupervisedDataResults()): DataFrame = 
  {
    var newdf = info
    var unlabeledProcess = new UnlabeledTransformer()
    var pipeline:Pipeline = new Pipeline()
    var results:(Double, Double, Double, Double, Long, Long, Long, Long, Int) = (0.0, 0.0, 0.0, 0.0, 0.toLong, 0.toLong, 0.toLong, 0.toLong, 0)
    modelsPipeline.map(criterion => criterion._2.map(percentatge => percentatge._2.map(threshold => threshold._2.map(classi => (pipeline = new Pipeline().setStages(Array(featurization,
                                                                                                                                      unlabeledProcess.setPercentage(percentatge._1),
                                                                                                                                      classi._2)),
                                                                                            results = crossValidation(data,kfold,pipeline,resultSemiSupervisedData),
                                                                                            newdf = newdf.withColumn("accuracy", when(newdf("percentageLabeled") === percentatge._1 &&
                                                                                                                                    newdf("criterion") === criterion._1 &&
                                                                                                                                    newdf("classifier") === classi._1&&
                                                                                                                                    newdf("thresholdOrKBest") === threshold._1
                                                                                                                                    ,results._1).otherwise (newdf("accuracy"))),
                                                                                            newdf = newdf.withColumn("AUC", when(newdf("percentageLabeled") === percentatge._1 &&
                                                                                                                                newdf("criterion") === criterion._1 &&
                                                                                                                                newdf("classifier") === classi._1 &&
                                                                                                                                newdf("thresholdOrKBest") === threshold._1
                                                                                                                                ,results._2).otherwise (newdf("AUC"))),
                                                                                            newdf = newdf.withColumn("PR", when(newdf("percentageLabeled") === percentatge._1 &&
                                                                                                                               newdf("criterion") === criterion._1 &&
                                                                                                                                newdf("classifier") === classi._1 && 
                                                                                                                                newdf("thresholdOrKBest") === threshold._1
                                                                                                                                , results._3).otherwise (newdf("PR"))),
                                                                                            newdf = newdf.withColumn("F1score", when(newdf("percentageLabeled") === percentatge._1 &&
                                                                                                                                    newdf("criterion") === criterion._1 &&
                                                                                                                                    newdf("classifier") === classi._1 &&
                                                                                                                                    newdf("thresholdOrKBest") === threshold._1
                                                                                                                                    ,results._4).otherwise (newdf("F1score"))),
                                                                                            newdf = newdf.withColumn("iteration", when(newdf("percentageLabeled") === percentatge._1 && 
                                                                                                                                      newdf("criterion") === criterion._1 &&
                                                                                                                                      newdf("classifier") === classi._1  &&
                                                                                                                                      newdf("thresholdOrKBest") === threshold._1
                                                                                                                                      ,results._9).otherwise (newdf("iteration"))),
                                                                                            newdf = newdf.withColumn("LabeledInitial", when(newdf("percentageLabeled")=== percentatge._1  && 
                                                                                                                                           newdf("criterion") === criterion._1 &&
                                                                                                                                           newdf("classifier") === classi._1 &&  
                                                                                                                                           newdf("thresholdOrKBest") === threshold._1 
                                                                                                                                           ,results._5 ).otherwise (newdf("LabeledInitial"))),
                                                                                            newdf = newdf.withColumn("UnLabeledInitial", when(newdf("percentageLabeled") === percentatge._1  &&
                                                                                                                                             newdf("criterion") === criterion._1 &&
                                                                                                                                             newdf("classifier") === classi._1 && 
                                                                                                                                             newdf("thresholdOrKBest") === threshold._1 
                                                                                                                                             ,results._6).otherwise (newdf("UnLabeledInitial"))),
                                                                                            newdf = newdf.withColumn("LabeledFinal", when(newdf("percentageLabeled") === percentatge._1  && 
                                                                                                                                         newdf("criterion") === criterion._1 &&
                                                                                                                                         newdf("classifier") === classi._1 && 
                                                                                                                                         newdf("thresholdOrKBest") === threshold._1 
                                                                                                                                         ,results._7).otherwise (newdf("LabeledFinal"))),
                                                                                            newdf = newdf.withColumn("UnLabeledFinal", when(newdf("percentageLabeled") === percentatge._1 &&
                                                                                                                                           newdf("criterion") === criterion._1 &&
                                                                                                                                           newdf("classifier") === classi._1 && 
                                                                                                                                           newdf("thresholdOrKBest") === threshold._1 
                                                                                                                                           , results._8).otherwise(newdf("UnLabeledFinal"))),
                                                                                            newdf = newdf.withColumn("percentageLabeledFinal", when(newdf("percentageLabeled") === percentatge._1 && 
                                                                                                                                        newdf("criterion") === criterion._1 &&
                                                                                                                                        newdf("classifier")===classi._1 && 
                                                                                                                                        newdf("thresholdOrKBest")=== threshold._1
                                                                                                                                        ,(1 - (results._8.toDouble/results._6.toDouble)))
                                                                                                                     .otherwise (newdf("percentageLabeledFinal"))) 

                                                                                           )))))
     newdf
  }
  
  
   /**
   *Cross Validator
  */
  def crossValidation[
      FeatureType,
      E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
      M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
    ](data: DataFrame,
      kFolds: Int,
      modelsPipeline: Pipeline,
      resultsSSData: SemiSupervisedDataResults = new SemiSupervisedDataResults()): (Double, Double, Double, Double, Long, Long, Long, Long, Int)= {
    
    // creamos la array de salida con el type y el tamaño
    var folds = kFold(data.rdd, kFolds, 8L)
    var acierto: Double = 0.0
    var auROC: Double = 0.0
    var auPR: Double = 0.0
    var f1Score: Double = 0.0
    var labeledIni: Int = 0
    var dataLabeledFinal: Long =0
    var dataUnDataLabeledFinal: Long =0
    var dataLabeledIni: Long =0
    var dataUnLabeledIni: Long =0
    var iteracionSemiSuper: Int =0
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
    
    for(iteration <- 0 to kFolds-1) {
       var dataTraining=spark.createDataFrame(folds(iteration)._1, data.schema)
       var dataTest=spark.createDataFrame(folds(iteration)._2, data.schema)
       dataTraining.persist()
       dataTest.persist()
       var predictionsAndLabelsRDD = modelsPipeline.fit(dataTraining)
        .transform(dataTest)
        .select("prediction", "label").rdd.map(row => (row.getDouble(0), row.getDouble(1)))
      
      //persistents
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
      
      //metrics
      acierto = metrics.accuracy + acierto
      auROC = metrics2.areaUnderROC + auROC
      auPR = metrics2.areaUnderPR + auPR 
      f1Score = metrics.fMeasure(1) + f1Score
    }
    
    //avg metrics
    acierto = acierto / kFolds
    auROC = auROC / kFolds
    auPR = auPR / kFolds
    f1Score =f1Score / kFolds
    dataLabeledFinal = dataLabeledFinal / kFolds
    dataUnDataLabeledFinal = dataUnDataLabeledFinal / kFolds
    dataLabeledIni = dataLabeledIni / kFolds
    dataUnLabeledIni =dataUnLabeledIni / kFolds
    iteracionSemiSuper = iteracionSemiSuper/ kFolds
    
    //out
    (acierto, auROC, auPR, f1Score, dataLabeledIni, dataUnLabeledIni, dataLabeledFinal, dataUnDataLabeledFinal, iteracionSemiSuper)

  }
}
