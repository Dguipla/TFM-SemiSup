package org.apache.spark.ml.semisupervised

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
//import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.linalg.DenseVector
//import sqlContext.implicits._
import org.apache.spark.sql.functions.udf

/** Clase supervisada donde solo entrenamos los datos etiquetado */

class Supervised [
  FeatureType,
  E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
] (
    val uid: String,
    val baseClassifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  ) extends org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]
with Serializable {
  var columnNameNewLabels: String = "labelSelection"
  
  //uid
  def this(classifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]) =
    this(Identifiable.randomUID("Supervised"), classifier)
  
  ///////////////////////////////////////////////////////////////////////////
  // setter
  ///////////////////////////////////////////////////////////////////////////
  def setColumnLabelName(nameColumn: String) = {
    columnNameNewLabels = nameColumn
    this
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // train
  ///////////////////////////////////////////////////////////////////////////
  def train(dataset: org.apache.spark.sql.Dataset[_]): M = {
    val dataUnLabeled = dataset.filter(dataset(columnNameNewLabels).isNaN).toDF
    val dataLabeled = dataset.toDF.exceptAll(dataUnLabeled)
    baseClassifier.fit(dataLabeled)
  }
  override def copy(extra: org.apache.spark.ml.param.ParamMap): E = defaultCopy(extra)
}
