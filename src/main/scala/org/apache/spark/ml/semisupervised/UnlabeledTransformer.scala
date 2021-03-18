package org.apache.spark.ml.semisupervised

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{ Param, ParamMap }
import org.apache.spark.ml.util.{ DefaultParamsReadable, DefaultParamsWritable, Identifiable }
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._

/** class to untag data */

class UnlabeledTransformer(override val uid: String) extends Transformer with DefaultParamsWritable {
  var percentageLabeled: Double = 0.1
  var seedValue: Long = 11L
  var columnNameNewLabels: String = "labelSelection"
  def this() = this(Identifiable.randomUID("UnlabeledTransformer"))
  
  ///////////////////////////////////////////////////////////////////////////
  // setters
  ///////////////////////////////////////////////////////////////////////////

  def setPercentage(percentage: Double) = {
    percentageLabeled= percentage
    this
   }
  
  def setColumnLabelName(nameColumn: String) = {
    columnNameNewLabels = nameColumn
    this
  }
  
  def setSeed(seedV: Long) = {
    seedValue= seedV
    this
  }
  
  
  ///////////////////////////////////////////////////////////////////////////
  // transform
  ///////////////////////////////////////////////////////////////////////////
  override def transform(data: Dataset[_]): DataFrame = {
    val dataSp = data.randomSplit(Array(percentageLabeled, 1-percentageLabeled),seed = seedValue)
    val dataLabeled = dataSp(0).toDF.withColumn(columnNameNewLabels,col("label"))
    val dataUnlabeled = dataSp(1).toDF.withColumn(columnNameNewLabels,col("label")*Double.NaN) 
    dataLabeled.unionAll(dataUnlabeled)

  }

  override def copy(extra: ParamMap): UnlabeledTransformer = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = schema
}

object UnlabeledTransformer extends DefaultParamsReadable[UnlabeledTransformer] {
  override def load(path: String): UnlabeledTransformer = super.load(path)
}

