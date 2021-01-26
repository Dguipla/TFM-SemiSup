// Databricks notebook source
// miramos los documentos adjuntados
display(dbutils.fs.ls("/FileStore/tables"))

// COMMAND ----------

// DBTITLE 1,Class UnlabeledTransformer
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{ Param, ParamMap }
import org.apache.spark.ml.util.{ DefaultParamsReadable, DefaultParamsWritable, Identifiable }
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._


//+++++++++++++++++++++++++++++++++++++
//clase para desetiquetar conjunto de datos
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
    val dataUnlabeled=dataSp(1).toDF.withColumn(columnNameNewLabels,col("label")*Double.NaN) //unlabeled esta parte de conjunto de datos
   // println(dataLabeled.count)
    //println(dataUnlabeled.count)
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
// Clase SELF TRAINING dode generaremos el modelo
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SelfTraining [
  FeatureType,
  E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
  M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
] (
    val uid: String,
    var baseClassifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
    
  ) extends org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M] with Serializable {
  
  //variables
  var porcentajeLabeled:Double = 1.0//0.002  // reduccion de porcentaje
  var threshold:Double=0.7
  var maxIter:Int=7
  var criterion:String= "threshold"
  var kBest:Int=10 // porcentaje podria ser una opcion
  var countDataLabeled:Long = _
  var countDataUnLabeled:Long = _
  var dataLabeledIni:Long =_
  var dataUnLabeledIni:Long = _
  var iter:Int = _
  var columnNameNewLabels :String ="labelSelection"

  //uid
  def this(classifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]) =
  //def this() =
    this(Identifiable.randomUID("selfTrainning"),classifier)
  

  //SETTERS
  
 
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
  
  // set umbral de confianza
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
  def setKbest(kb:Int)={
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
  
  
  
  //hacemos el split entre los datos de entrenamiento L y U labeled y unlabeler respecticamente en funcion del porcentaje
  def resplit(data:org.apache.spark.sql.Dataset[_],porcentajeEtiquetado:Double):Array[org.apache.spark.sql.Dataset[_]] ={
    val dataSp=  data.randomSplit(Array(porcentajeEtiquetado, 1-porcentajeEtiquetado),seed = 11L)
    Array(dataSp(0),dataSp(1))
    
  }
  
 
  

  
  def train(dataset: org.apache.spark.sql.Dataset[_]): M= {
    iter = 1
    //udf para coger el valor mas elevado de la array de probabilidades
    val max = udf((v: org.apache.spark.ml.linalg.Vector) => v.toArray.max)
    
    var dataUnLabeled=dataset.filter(dataset(columnNameNewLabels).isNaN).toDF.cache()
    var dataLabeled = dataset.toDF.exceptAll(dataUnLabeled).cache()

    //guardamos el tamaño inicial de los datos etiquetados y no etiquetados
    dataLabeledIni = dataLabeled.count()
    dataUnLabeledIni = dataUnLabeled.count()
    //seleccionamos features and labels
    dataLabeled = dataLabeled.select("features","label")
    dataUnLabeled = dataUnLabeled .select("features","label")
    countDataLabeled = dataLabeled.count()
    countDataUnLabeled = dataUnLabeled.count()
    

    
    // total data inicial
    /*println ("EMPIEZA EL SELF-TRAINING")
    println ("threshold: "+threshold)
    println ("total labeled Inicial " + countDataLabeled)
    println ("total unlabeled Inicial " + countDataUnLabeled)
    */
    

    var modeloIterST = baseClassifier.fit(dataLabeled)
    var prediIterST = modeloIterST.transform(dataUnLabeled)
    
    dataLabeled.unpersist()
    dataUnLabeled.unpersist()
    

    while ((iter<maxIter) && (countDataUnLabeled>0)){

      val modificacionPrediccion=prediIterST.withColumn("probMax", max($"probability"))
      val seleccionEtiquetasMayoresThreshold=modificacionPrediccion.filter(modificacionPrediccion("probMax")>threshold)
      val seleccionEtiquetasMenoresIgualThreshold=modificacionPrediccion.filter(modificacionPrediccion("probMax")<=threshold)
      
      // selecionamos las features y las predicciones y cambiamos el nombre de las predicciones por labels
      val newLabeledFeaturesLabels = seleccionEtiquetasMayoresThreshold.select ("features","prediction").withColumnRenamed("prediction","label")
      val newUnLabeledFeaturesLabels = seleccionEtiquetasMenoresIgualThreshold.select ("features","prediction").withColumnRenamed("prediction","label")

      dataLabeled = dataLabeled.union(newLabeledFeaturesLabels).cache()
      dataUnLabeled = newUnLabeledFeaturesLabels.cache()
      countDataUnLabeled = dataUnLabeled.count()
      countDataLabeled = dataLabeled.count()
      /// persist dataLabeled and unlabeled
      
      
      if (countDataUnLabeled>0 && iter<maxIter ){
        //println ("Trainning... Next Iteration")
        modeloIterST = baseClassifier.fit(dataLabeled)
        prediIterST = modeloIterST.transform(dataUnLabeled)
        iter = iter+1
      }
      else{ //final del ciclo 
        modeloIterST = baseClassifier.fit(dataLabeled)
        //iter = maxIter
      }
      
      dataLabeled.unpersist()
      dataUnLabeled.unpersist()
    
      
    }
    //println("ACABAMOS SELF-TRAINING Y RETORNAMOS MODELO")
    modeloIterST



  }
  
  
  def transform(dataset: Dataset[_]): DataFrame = {
    // guardamos informacion de los datos etiquetados/No etiquetados totales, iniciales ...
    print ("transform ST")
    var newData =dataset.withColumn("dataLabeledIni", lit(dataLabeledIni))
    newData=newData.withColumn("dataLabeledFinal", lit(countDataLabeled))
    newData=newData.withColumn("dataUnLabeledIni", lit(dataUnLabeledIni))
    newData=newData.withColumn("dataUnLabeledFinal", lit(countDataUnLabeled))
    newData=newData.withColumn("Iteration", lit(iter))
    newData
  }

  //override def copy(extra: ParamMap): UnlabeledTransformer = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = schema
  
  override def copy(extra: org.apache.spark.ml.param.ParamMap):E = defaultCopy(extra)
}





// COMMAND ----------

// DBTITLE 1,Cross Validation - Function
import org.apache.spark.mllib.util.MLUtils.kFold
import org.apache.spark.sql.functions._
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

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
    classiType:String ="supervised",
    instanceModelST: SelfTraining[FeatureType,E,M] =new SelfTraining(new DecisionTreeClassifier())): (Double, Double, Double, Double, Long, Long, Long, Long, Int)=

{
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
    if (classiType == "selfTraining"){
      
      dataLabeledFinal = instanceModelST.getDataLabeledFinal() + dataLabeledFinal
      dataUnDataLabeledFinal = instanceModelST.getUnDataLabeledFinal() + dataUnDataLabeledFinal
      dataLabeledIni = instanceModelST.getDataLabeledIni() + dataLabeledIni
      dataUnLabeledIni =instanceModelST.getUnDataLabeledIni() + dataUnLabeledIni
      iteracionSemiSuper = instanceModelST.getIter() + iteracionSemiSuper 
    }
 
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


//crossValidation(datosDF_2,4,0.1,generadorPipelinePorClasificador(0))


// COMMAND ----------

// DBTITLE 1,FUNCIONES SUPERVISADO (RESULTADOS)
// LIBRERIAS necesarias (IMPORTACIONES)

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


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// GENERADOR DE DATA FRAME RESULTADOS (solo el esqueleto del DATA FRAME)
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/* QUEDARIA:

+----+------------+--------------------+--------+---+---+--------+
|data|clasificador|porcentajeEtiquetado|accuracy|AUC|PR |F1score |
+----+------------+--------------------+--------+---+---+--------+
| BCW|          DT|                0.02|     x.x|x.x|x.x|     x.x|
| BCW|          DT|                0.05|     x.x|x.x|x.x|     x.x|
| BCW|          DT|                 0.1|     x.x|x.x|x.x|     x.x|
| BCW|          DT|                 0.3|     x.x|x.x|x.x|     x.x|
| BCW|          LR|                0.02|     x.x|x.x|x.x|     x.x|
....
....
*/

def generadorDataFrameResultados(datos:String,clasificadores:Array[String],porcentaje:Array[Double]):DataFrame= 
{
  // creamos la array de salida con el type y el tamaño
  var seqValores=Seq[(String, String, Double, Double, Double, Double, Double)]() 
  var posicion = 0
  for(posClasi <-clasificadores) {
    for(posPorce <-porcentaje){
      seqValores = seqValores :+ (datos,posClasi,posPorce,0.00,0.00,0.00,0.00)
    }
  }
  // generamos el DataFrame que sera la salida
  spark.createDataFrame(seqValores).toDF("data", "clasificador", "porcentajeEtiquetado", "accuracy","AUC","PR","F1score")
}

//NUEVA ESTRUCTURA DE DATOS PARA SEMISUPERVISADO
//|data| Aprendizaje| clasifi| porcentaje| threshold| iter| LabeledInicial| UnlabeledInicial| LabeledFinal| UnlabeledFinal| porEtiFinal| Acc| AUC| F1Score|
//+----+------------+--------+-----------+----------+-----+---------------+-----------------+-------------+---------------+------------+----+----+--------+
//| BCW| SemiSupervi|      DT|      0.001|       0.6|    2|            130|           100600|        98000|           2600|       97.4 | x.x| x.x|     x.x|
//| BCW| SemiSupervi|      DT|      0.005|       0.6|    3|            270|           100460|        96000|           2800|       97.21| x.x| x.x|     x.x|
//....




//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ENTRENAMIENTO + GENERACION DEL MODELO + RESULTADO CON CONJUNTO DE TEST
//Para cada clasificador:
// 1.- particionamos los datos en funcion del % seleccionado 2%, 5%....
// 2.-Entrenamos el Pipeline Completo (por clasificador) + calculamos la accuracy
//*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//primero hacemos la particion de los datos segun el % y despues con DataSet reducido trabajamos con todos los clasificadores asi succesivamente para cada clasificador de esta forma tenemos el mismo dataSet reducido utilizado para cada modelo de entrenamiento



def generadorModeloResutladosCompleto2(datos:DataFrame,
                                      kFold:Int,
                                      featurization:Pipeline,
                                      modelStatePipeline:Array[(String,PipelineStage)], // TE Q SER ARRAY
                                      info:DataFrame,
                                      porcentajeEtiquetado:Array[Double], 
                                      tiposClasi:Array[String]):DataFrame= 
{
  var newdf= info
  var unlabeledProcess =new UnlabeledTransformer()
  for (posPorcentaje <-porcentajeEtiquetado){
    for(posClasi <-tiposClasi) {      
 
      datos.persist()
      
      for (posPipeline <- 0 to (modelStatePipeline.size-1)){
        
        //miramos que clasificador base es para despues poner su resultado en la posicion correspondiente dentro del DataFrame
        if (modelStatePipeline(posPipeline)._1.contentEquals(posClasi)){ 
           println("porcentaje: " + posPorcentaje)
           println("clasificador: " +posClasi)
           var pipeline = new Pipeline().setStages(Array(
                                                         featurization,
                                                         unlabeledProcess.setPercentage(posPorcentaje),
                                                         modelStatePipeline(posPipeline)._2))
          
          // CV 
          var results = crossValidation(datos,kFold,pipeline) // ._1 -->acierto, ._2 --> auROC, ._3 -->auPR, ._4 -->f1Score)

          datos.unpersist()

          //accuracy
          newdf = newdf.withColumn("accuracy",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi, results._1).
                          otherwise (newdf("accuracy")))
          //AUC
           newdf = newdf.withColumn("AUC",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi, results._2).
                          otherwise (newdf("AUC")))
          //PR
          newdf = newdf.withColumn("PR",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi, results._3).
                          otherwise (newdf("PR")))
          
          //f1score
          newdf = newdf.withColumn("F1score",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi,results._4).
                                   otherwise (newdf("F1score")))
        }  
      }
    }
  }
  newdf
}







//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CREACION DEL PIPELINE PARA CADA CLASIFICADOR SELECCIONADO (DT, RF , LR....)
// Utilizamos el stage pipeline definido de featurization y añadimos un nuevo stage del clasificador 
//*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// Funcion creador de pipelines por clasificador
def generadorPipelinesPorClasificador(instanciasModelos:Array[(String,PipelineStage)],
                                      featurization:Pipeline
                                     ):Array[(String,Pipeline)]= 
{
  // creamos la array de salida con el type y el tamaño
  var salida:Array[(String,Pipeline)] = new Array[(String,Pipeline)](instanciasModelos.size) 
  var posicion = 0
  for(inModel <-instanciasModelos) {
    
    // Generamos los Pipelines para cada clasificador
    val pipeline = new Pipeline().setStages(Array(
                                                  featurization,
                                                  instanciasModelos(posicion)._2 // cogemos solo el stage de la array
                                                   ))
    
    salida(posicion) = (instanciasModelos(posicion)._1,pipeline) // ponemos en la array el clasificador con su pipeline--> clasificador,pipeline
    posicion = posicion+1
    
  }
  salida
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// creacion un pipelineStage para cada columna de features de conversion de categorico a continuo
//salida de array[PipelineStage]
//*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def indexStringColumnsStagePipeline(df:DataFrame,cols:Array[String]):(Pipeline,Array[String])= {
  var intermedioStages:Array[(PipelineStage)] = new Array[(PipelineStage)](cols.size)
  var posicion = 0
  for(col <-cols) {
    val si= new StringIndexer().setInputCol(col).setOutputCol(col+"-num")
    intermedioStages(posicion) = si.setHandleInvalid("keep")
    posicion = posicion +1
  }
  val salida = new Pipeline().setStages(intermedioStages)
  (salida,df.columns.diff(cols))
}






// COMMAND ----------

// DBTITLE 1,FUNCIONES SEMISUPERVISADO

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CREAMOS EL TEMPLATE DE DATAFRAME PARA LOS RESULTADOS SEMISUPERVISADO
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//NUEVA ESTRUCTURA DE DATOS PARA SEMISUPERVISADO
//|data| clasifi| porcentaje| threshold| iter| LabeledInicial| UnlabeledInicial| LabeledFinal| UnlabeledFinal| porEtiFinal| Acc| AUC| PR| F1Score|
//+----+--------+----------+-----------+-----+---------------+-----------------+-------------+---------------+------------+----+----+----+-------+
//| BCW|   ST-DT|      0.001|       0.6|    2|            130|           100600|        98000|           2600|       97.4 | x.x| x.x| x.x|    x.x|
//| BCW|   ST-DT|      0.005|       0.6|    3|            270|           100460|        96000|           2800|       97.21| x.x| x.x| x.x|    x.x|
//....


def generadorDataFrameResultadosSemiSuper(datos:String,
                                          clasificadores:Array[String],
                                          porcentaje:Array[Double],
                                          threshold:Array[Double],    
                                         ):DataFrame= 
{
  // creamos la array de salida con el type y el tamaño
  var seqValores=Seq[(String, String,Double, Double,Int,Int,Long,Long,Long,Double,Double, Double,Double, Double)]() 
  //var seqValores=Seq[(String, String,Double, Double,Int,Int,Int,Int,Int,Double,Double, Double,Double, Double)]() 
  var posicion = 0
  for(posClasi <-clasificadores) {
    for(posPorce <-porcentaje){
      for (posThreshold <- threshold){
        seqValores = seqValores :+ (datos,posClasi,posPorce,posThreshold,0,0,0.toLong,0.toLong,0.toLong,0.00,0.00,0.00,0.00,0.00)
      }
    }
  }

  // generamos el DataFrame que sera la salida
  spark.createDataFrame(seqValores).toDF("data",
                                         "clasificador",
                                         "porcentajeEtiquetado",
                                         "threshold",
                                         "iteracion",
                                         "LabeledInicial",
                                         "UnLabeledInicial",
                                         "LabeledFinal",
                                         "UnLabeledFinal",
                                         "porEtiFinal", 
                                         "accuracy",
                                         "AUC",
                                         "PR",
                                         "F1score")
}



//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//Generamos resultado de los modelos semisupervisado SelfTraining (por ahora) y lo volcamos en el DF de resultados
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def generadorModeloSemiAndSuperResutladosCompleto2 [
    FeatureType,
    E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
    M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
  ] (featurization:Pipeline,
     kfold:Int,
     datos:DataFrame,
     modelosStatgePipeline:Array[(String,PipelineStage)], 
     info:DataFrame,
     porcentajeEtiquetado:Array[Double],
     threshold:Array[Double],
     tiposClasi:Array[String]):DataFrame= 
{
  var newdf= info
  var unlabeledProcess =new UnlabeledTransformer()
  for (posPorcentaje <-porcentajeEtiquetado){
    println("PERCENTAGE: " + posPorcentaje)
    for (posThres <-threshold){
      println("THRESHOLD: " + posThres)
      for(posClasi <-tiposClasi) {  
        println("CLASIFICADOR: " + posClasi)
        for (posPipeline <- 0 to (modelosStatgePipeline.size-1)){
          //miramos que clasificador base es para despues poner su resultado en la posicion correspondiente dentro del DataFrame
          if (modelosStatgePipeline(posPipeline)._1.contentEquals(posClasi)){ 
            
            // generamos la instacia para el SelfTrainning con los thresholds necesarios
             var semiSupervisorInstanciaModel = new SelfTraining(modelosStatgePipeline(posPipeline)._2.asInstanceOf[E]).setThreshold(posThres)
             //var semiSupervisorInstanciaModel = modelosStatgePipeline(posPipeline)._2//.setThreshold(posThres)
            //var semiSupervisorInstanciaModel = new SelfTraining(new DecisionTreeClassifier()).setThreshold(posThres)

            // Generamos pipelines para cada modelo
            var pipelineSemiSup = new Pipeline().setStages(Array(featurization,
                                                                         unlabeledProcess.setPercentage(posPorcentaje),
                                                                         semiSupervisorInstanciaModel)) // cogemos solo el stage de la array 
            
            
            var results = crossValidation(datos,kfold,pipelineSemiSup,"selfTraining",semiSupervisorInstanciaModel)
            // ._1 -->acierto, ._2 --> auROC, ._3 -->auPR, ._4 -->f1Score
            // ._5--> dataLabeledIni, ._6-->dataUnLabeledIni,._7-->dataLabeledFinal,._8-->dataUnDataLabeledFinal,._9-->iteracionSemiSuper)
            

            //Resultados para SemiSupervisado y supervisado
            //accuracy
            newdf = newdf.withColumn("accuracy",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                     newdf("clasificador")===posClasi &&
                                                     newdf("threshold")=== posThres
                                                     ,results._1).otherwise (newdf("accuracy")))
            //AUC
             newdf = newdf.withColumn("AUC",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                 newdf("clasificador")===posClasi &&
                                                 newdf("threshold")=== posThres
                                                 , results._2).otherwise (newdf("AUC")))
            //PR
            newdf = newdf.withColumn("PR",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi, results._3).
                            otherwise (newdf("PR")))

            //f1score
            newdf = newdf.withColumn("F1score",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                    newdf("clasificador")===posClasi &&
                                                    newdf("threshold")=== posThres
                                                    ,results._4).otherwise (newdf("F1score")))
            //iter
            newdf = newdf.withColumn("iteracion",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                      newdf("clasificador")===posClasi &&
                                                      newdf("threshold")=== posThres
                                                      , results._9).otherwise (newdf("iteracion")))
            //LabeledInicial 
            newdf = newdf.withColumn("LabeledInicial",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi
                                                           ,  results._5 ).otherwise (newdf("LabeledInicial")))
            //UnLabeledInicial
            newdf = newdf.withColumn("UnLabeledInicial",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                             newdf("clasificador")===posClasi &&
                                                             newdf("threshold")=== posThres
                                                           ,  results._6).otherwise (newdf("UnLabeledInicial")))   
            //LabeledFinal
            newdf = newdf.withColumn("LabeledFinal",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                         newdf("clasificador")===posClasi &&
                                                         newdf("threshold")=== posThres
                                                         ,  results._7).otherwise (newdf("LabeledFinal")))
            //UnLabeledFinal
            newdf = newdf.withColumn("UnLabeledFinal",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                           newdf("clasificador")===posClasi &&
                                                           newdf("threshold")=== posThres
                                                           ,  results._8).otherwise (newdf("UnLabeledFinal")))
            //Etiquetado % final
            // UnDataLabeledFinal() / UnDataLabeledIni())
            newdf = newdf.withColumn("porEtiFinal",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                        newdf("clasificador")===posClasi &&
                                                        newdf("threshold")=== posThres
                                                           , (1 -( results._8.toDouble/results._6.toDouble))).otherwise (newdf("porEtiFinal"))) 
            
            
          }  
        }
      } 
    }
  }
  newdf
}





// COMMAND ----------

val instanciaTrainingPipelineDT = new SelfTraining(new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label"))

val test =new UnlabeledTransformer().setPercentage(0.1)
val feat = featurizationPipelineBCW.fit(datosDF_2).transform(datosDF_2)

//val ff= new Pipeline().setStages(Array(test,instanciaTrainingPipelineDT))


val ff= new Pipeline().setStages(Array(test,instanciaTrainingPipelineDT))
val fff = new Pipeline().setStages(Array(instanciaTrainingPipelineDT))
val results = ff.fit((feat))//.transform(feat)


//val salida =(instanciaTrainingPipelineDT.transform(feat))
//display(results)

//valor.select("new_column").first.getInt(0)
//display(ffff.fit.transform(feat))
fff.getStages(0).getClass.getDeclaredMethods
val a=new UnlabeledTransformer()
//a.getClass.transform(feat)
def cr[
    FeatureType,
    E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
    M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
  ](data:DataFrame,
    instanceModelST: PipelineStage):DataFrame={
    instanceModelST.asInstanceOf[M].transform(data)
    
  
}

//fff.getStages(0).

val ddd =cr(feat,instanciaTrainingPipelineDT)
/*
val m = Array(1,2,3,4,5)
val n = Array("m1", "m2", "m3","m4","m5")
m.map(x=>((x+n(x-1)),1))
m.map(x=>((x+n(0)),m(x-1)))
val resultado2 =m.map(x=>(x,n.map(y=>(y,x))))
val resultado = m.map(x=>(n.map(y=>(y,x))))
val aaa =resultado2(0)
aaa.filter

*/

// COMMAND ----------

instanciaTrainingPipelineDT.getDataLabeledFinal

// COMMAND ----------

// MAGIC %md
// MAGIC ## ANALISIS DE LOS DIFERENTES CONJUNTO DE DATOS
// MAGIC ##### Apartir de aquí analizamos los diferentes conjuntos de datos, tanto **supervisado** con reduccion de labels como **Semisupervisado**
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

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//PIPELINE  Y EVALUACION PARA TODOS LOS CLASIFICADORES BASE
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val instanciaTrainingPipelineDT = new Supervised(new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineRF = new Supervised(new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineNB = new Supervised(new NaiveBayes().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineLR = new Supervised(new LogisticRegression().setFeaturesCol("features").setLabelCol("label"))
//val instanciaTrainingPipelineLSVM = new Supervised(new LinearSVC().setFeaturesCol("features").setLabelCol("label")


// array de instancias para cada clasificador base
val arrayInstanciasClasificadores:Array[(String,PipelineStage)] = Array(("DT",instanciaTrainingPipelineDT),
                                       ("LR",instanciaTrainingPipelineLR),
                                       ("RF",instanciaTrainingPipelineRF),
                                       ("NB",instanciaTrainingPipelineNB)
                                       //("LSVM",instanciaTrainingPipelineLSVM) 
                                      )



//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//preparamos los parametros:  
// 1.- tipo clasificadores, 
// 2.- porcentaje .... (la informacion que esta en el DataFrame de resultados)
// 3.- tipo de datos en esta caso breast cancer wisconsin --> BCW
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val clasificadoresBase = Array ("DT")//Array ("DT","LR","RF","NB")//,"LSVM")
val porcentajeLabeled = Array(0.01)//Array(0.01,0.05,0.08,0.10,0.20,0.30)
val datosCodigo = "BCW"


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CREAMOS EL TEMPLATE DE DATAFRAME PARA LOS RESULTADOS
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val breastCancerWisconResultados = generadorDataFrameResultados(datosCodigo,clasificadoresBase,porcentajeLabeled)


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PIPELINE FINAL AUTOMATIZADO PARA CADA CLASIFICADOR
//
// FEATURIZATIONT(BCW)--> Supervised(CLASIFICACOR (DT)) --> RESUTADO
//                    --> Supervised(CLASIFICACOR (RF)) --> RESUTADO
//                    --> Supervised(CLASIFICACOR (NB)) --> RESUTADO
//                    --> .....
//                    --> .....
//                    --> Supervised(CLASIFICACOR (XX)) --> RESUTADO
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val resultados_BCW = generadorModeloResutladosCompleto2(datosDF_2,
                                      4,
                                      featurizationPipelineBCW,
                                      arrayInstanciasClasificadores, // TE Q SER ARRAY
                                      breastCancerWisconResultados,
                                      porcentajeLabeled,
                                      clasificadoresBase)

display(resultados_BCW)


// COMMAND ----------

// DBTITLE 1,SelfTrainning - BCW (DT, LR, RF ....)


val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


val arrayInstanciasClasificadores_BCW_NewST:Array[(String,PipelineStage)] = Array(("ST-DT",instanciaTrainingPipelineDT),
                                      ("ST-LR",instanciaTrainingPipelineLR),
                                       ("ST-RF",instanciaTrainingPipelineRF),
                                       ("ST-NB",instanciaTrainingPipelineNB)
                                      )



val clasificadoresBase = Array("ST-DT","ST-LR","ST-RF","ST-NB")//Array ("DT","LR","RF","NB","LSVM")
val porcentajeLabeledBCW_ST = Array(0.01,0.05,0.08,0.10,0.20,0.30)
val umbral= Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val datosCodigo = "BCW"



val resultadosBCW_ST=generadorDataFrameResultadosSemiSuper(datosCodigo,
                                          clasificadoresBase,
                                          porcentajeLabeledBCW_ST,
                                          umbral    
                                         )


val resultadosBCW_STFinal =generadorModeloSemiAndSuperResutladosCompleto2(featurizationPipelineBCW, 
                                                                         4,
                                                                         datosDF_2,
                                                                         arrayInstanciasClasificadores_BCW_NewST,
                                                                         resultadosBCW_ST,
                                                                         porcentajeLabeledBCW_ST,
                                                                         umbral,
                                                                         clasificadoresBase)

display(resultadosBCW_STFinal)

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
//CLEANING y PRE-PROCESADO
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//Eliminamos lineasvacias

// training
val nonEmpty_training= lines_training.filter(_.nonEmpty).filter(y => ! y.contains("?")) // Dado que representa en el peor de los casos un 3-4% approx, lo eliminamos
val parsed_training= nonEmpty_training.map(line => line.split(","))


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//CREACION DEL ESQUEMA  Y DE LOS DATA FRAMES TANTO DE TEST COMO DE TRAINING
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
//DISTRIBUCION DE CLASES
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
//DATA SPLIT  TRAINING & TEST  (el split de los datos de training en 2%, 5% ... se hace posteriormente)
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//dividimos en datos de trainning 75% y datos de test 25%
val dataSplits= income_trainningDF_converted.randomSplit(Array(0.70, 0.30),seed = 8L)
val datosDFLabeled_trainningAdult = dataSplits(0)
val datosDFLabeled_testAdult = dataSplits(1)




//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//FEATURIZATION -> PREPARAMOS INSTANCIAS
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
//PIPELINE FEATURIZATION para Adult
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//generamos el pipeline poniendo correctamente el orden

val featurizationPipelineAdult = new Pipeline().setStages(Array(indexStringFeaturesTodasNumAdult,
                                                                assemblerFeaturesLabelPipelineAdult,
                                                                indiceClasePipelineAdult))






// COMMAND ----------

// DBTITLE 1,Supervised - ADULT  (DT, LR, RF,NB)
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//PIPELINE  Y EVALUACION PARA TODOS LOS CLASIFICADORES BASE
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val instanciaTrainingPipelineDT = new Supervised(new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label").setMaxBins(42))
val instanciaTrainingPipelineRF = new Supervised(new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label").setMaxBins(42))
val instanciaTrainingPipelineNB = new Supervised(new NaiveBayes().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineLR = new Supervised(new LogisticRegression().setFeaturesCol("features").setLabelCol("label"))
//val instanciaTrainingPipelineLSVM = new Supervised(new LinearSVC().setFeaturesCol("features").setLabelCol("label")


// array de instancias para cada clasificador base
val arrayInstanciasClasificadores:Array[(String,PipelineStage)] = Array(("DT",instanciaTrainingPipelineDT),
                                       ("LR",instanciaTrainingPipelineLR),
                                       ("RF",instanciaTrainingPipelineRF),
                                       ("NB",instanciaTrainingPipelineNB)
                                       //("LSVM",instanciaTrainingPipelineLSVM) 
                                      )


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//preparamos los parametros:  
// 1.- tipo clasificadores, 
// 2.- porcentaje .... (la informacion que esta en el DataFrame de resultados)
// 3.- tipo de datos en esta caso breast cancer wisconsin --> BCW
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val clasificadoresBase = Array("DT")//Array("DT","LR","RF","NB","LSVM")
val porcentajeLabeled = Array(0.002)//Array(0.002,0.005,0.01,0.05,0.10,0.30)
val datosCodigo = "ADULT"


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CREAMOS EL TEMPLATE DE DATAFRAME PARA LOS RESULTADOS
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val adultResultados = generadorDataFrameResultados(datosCodigo,clasificadoresBase,porcentajeLabeled)


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PIPELINE FINAL AUTOMATIZADO PARA CADA CLASIFICADOR
//
// FEATURIZATIO (BCW) --> CLASIFICACOR (DT) --> RESUTADO
//                    --> CLASIFICACOR (RF) --> RESUTADO
//                    --> CLASIFICACOR (NB) --> RESUTADO
//                    --> .....
//                    --> .....
//                    --> CLASIFICACOR (XX) --> RESUTADO
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


val resultados_ADULT = generadorModeloResutladosCompleto2(income_trainningDF_converted,
                                      4,
                                      featurizationPipelineAdult,
                                      arrayInstanciasClasificadores,
                                      adultResultados,
                                      porcentajeLabeled,
                                      clasificadoresBase)

display(resultados_ADULT)



// COMMAND ----------

display(resultados_ADULT)

// COMMAND ----------

// DBTITLE 1,SelfTrainning - ADULT (DT, LR, RF ....)



val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label").setMaxBins(42)
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label").setMaxBins(42)
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


val arrayInstanciasClasificadores_NewST:Array[(String,PipelineStage)] = Array(("ST-DT",instanciaTrainingPipelineDT),
                                      ("ST-LR",instanciaTrainingPipelineLR),
                                       ("ST-RF",instanciaTrainingPipelineRF),
                                       ("ST-NB",instanciaTrainingPipelineNB)
                                      )


val clasificadoresBase = Array("ST-DT")//Array("ST-DT","ST-LR","ST-RF","ST-NB")//Array ("DT","LR","RF","NB","LSVM")
val porcentajeLabeled = Array(0.002)//Array(0.002,0.005,0.01,0.05,0.10,0.30)//Array(0.001,0.005,0.01,0.05,0.10,0.30)
val umbral= Array(0.4)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val datosCodigo = "ADULT"



val resultadosAdultST=generadorDataFrameResultadosSemiSuper(datosCodigo,
                                          clasificadoresBase,
                                          porcentajeLabeled,
                                          umbral    
                                         )


val resultAdult_ST =generadorModeloSemiAndSuperResutladosCompleto2(featurizationPipelineAdult, 
                                                                         4,
                                                                         income_trainningDF_converted,
                                                                         arrayInstanciasClasificadores_NewST,
                                                                         resultadosAdultST,
                                                                         porcentajeLabeled,
                                                                         umbral,
                                                                         clasificadoresBase)

display(resultAdult_ST)


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
// LECTURA
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//leemos el documento
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
//CLEANING y PRE-PROCESADO
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//vamos a ver que valores nulos tenemos
val instanciasConNulos=datosDF.count() - datosDF.na.drop().count()
println("INSTANCIAS CON NULOS")
println(instanciasConNulos) //no hay nulos


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//DISTRIBUCION DE CLASES
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
//TRANSFORMACION TO BINARY CLASIFICATION
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
//DATA SPLIT  TRAINING & TEST  (el split de los datos de training en 2%, 5% ... se hace posteriormente)
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//dividimos en datos de trainning 75% y datos de test 25%
val dataSplits= datosDFNew.randomSplit(Array(0.75, 0.25),seed = 8L)
val datosDFLabeled_trainningPOKER = dataSplits(0)
val datosDFLabeled_testPOKER = dataSplits(1)

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//FEATURIZATION -> PREPARAMOS INSTANCIAS
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
//PIPELINE FEATURIZATION para POKER
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//generamos el pipeline
val featurizationPipelinePoker = new Pipeline().setStages(Array(
                                              indexStringFeaturesTodasNumPoker,
                                              assemblerFeaturesLabelPipelinePoker,
                                              indiceClasePipelinePoker))


//val featPoker=featurizationPipelinePoker.fit(datosDFLabeled_trainningPOKER).transform(datosDFLabeled_trainningPOKER)




// COMMAND ----------

// DBTITLE 1,SupervisadoClasificadoresBase - POKER (DT, LR, RF,NB) 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//PIPELINE  Y EVALUACION PARA TODOS LOS CLASIFICADORES BASE
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


val instanciaTrainingPipelineDT = new Supervised(new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineRF = new Supervised(new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineNB = new Supervised(new NaiveBayes().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineLR = new Supervised(new LogisticRegression().setFeaturesCol("features").setLabelCol("label"))
//val instanciaTrainingPipelineLSVM = new Supervised(new LinearSVC().setFeaturesCol("features").setLabelCol("label")


// array de instancias para cada clasificador base
val arrayInstanciasClasificadores:Array[(String,PipelineStage)] = Array(("DT",instanciaTrainingPipelineDT),
                                       ("LR",instanciaTrainingPipelineLR),
                                       ("RF",instanciaTrainingPipelineRF),
                                       ("NB",instanciaTrainingPipelineNB)
                                       //("LSVM",instanciaTrainingPipelineLSVM) 
                                      )


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//preparamos los parametros:  
// 1.- tipo clasificadores, 
// 2.- porcentaje .... (la informacion que esta en el DataFrame de resultados)
// 3.- tipo de datos en esta caso breast cancer wisconsin --> BCW
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val clasificadoresBase = Array("DT")//Array("DT","LR","RF","NB")//,"LSVM")
val porcentajeLabeled = Array(0.0001)//Array(0.0001,0.001,0.01,0.05,0.1,0.3)//Array(0.0001) //Array(0.0001,0.001,0.01,0.1,0.3)
val datosCodigo = "POKER"


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CREAMOS EL TEMPLATE DE DATAFRAME PARA LOS RESULTADOS
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val pokerResultados = generadorDataFrameResultados(datosCodigo,clasificadoresBase,porcentajeLabeled)


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PIPELINE FINAL AUTOMATIZADO PARA CADA CLASIFICADOR
//
// FEATURIZATIONT(BCW)--> CLASIFICACOR (DT) --> RESUTADO
//                    --> CLASIFICACOR (RF) --> RESUTADO
//                    --> CLASIFICACOR (NB) --> RESUTADO
//                    --> .....
//                    --> .....
//                    --> CLASIFICACOR (XX) --> RESUTADO
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


val resultados_Poker = generadorModeloResutladosCompleto2(datosDFNew,
                                      4,
                                      featurizationPipelinePoker,
                                      arrayInstanciasClasificadores, // TE Q SER ARRAY
                                      pokerResultados,
                                      porcentajeLabeled,
                                      clasificadoresBase)

display(resultados_Poker)


// COMMAND ----------

// DBTITLE 1,SelfTrainning - POKER (DT, LR, RF ....)




val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


val arrayInstanciasClasificadores_NewST:Array[(String,PipelineStage)] = Array(("ST-DT",instanciaTrainingPipelineDT),
                                      ("ST-LR",instanciaTrainingPipelineLR),
                                       ("ST-RF",instanciaTrainingPipelineRF),
                                       ("ST-NB",instanciaTrainingPipelineNB)
                                      )


val clasificadoresBase = Array("ST-DT")//Array("ST-DT","ST-LR","ST-RF","ST-NB")
val porcentajeLabeled = Array(0.0001)//Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val umbral= Array(0.4)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val datosCodigo = "POKER"



val resultsPOKER_ST=generadorDataFrameResultadosSemiSuper(datosCodigo,
                                          clasificadoresBase,
                                          porcentajeLabeled,
                                          umbral    
                                         )



val resultPoker_ST =generadorModeloSemiAndSuperResutladosCompleto2(featurizationPipelinePoker, 
                                                                         4,
                                                                         datosDFNew,
                                                                         arrayInstanciasClasificadores_NewST,
                                                                         resultsPOKER_ST,
                                                                         porcentajeLabeled,
                                                                         umbral,
                                                                         clasificadoresBase)

display(resultPoker_ST)



// COMMAND ----------

// DBTITLE 1,Data Processing & Data Preparation (Featurization) - POKER
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
// LECTURA
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// Creamos un RDD con los datos que vamos a explorar
import org.apache.spark.sql.functions.{concat, lit}
val PATH="/FileStore/tables/"  
val entreno = "train.csv"
val training = sc.textFile(PATH + entreno)

// Creamos un case class con el esquema de los datos
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
//CLEANING
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val nonEmpty_training= training.filter(_.nonEmpty)
// Separamos por , y eliminamos la primera linea head
val parsed_training= nonEmpty_training.map(line => line.split(",")).zipWithIndex().filter(_._2 >= 1).keys
// Asociamos los campos a la clase
val training_reg = parsed_training.map(x=>fields(x(0),
                                                 x(1).toInt,x(2),x(3),x(4).toInt,x(5).toDouble,x(6).toDouble,x(7).toDouble,x(8).toDouble,x(9),x(10).toInt)) 



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// MIRAMOS LA DISTRIBUCIÓN DE CLASES: 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val longTravel = training_reg.filter(x => x.trip_duration > 900).count
val shortTravel = training_reg.filter(x => x.trip_duration <= 900).count


val longTravelPorcentage = (longTravel.toDouble/training_reg.count())*100
val shortTravelPorcentage = (shortTravel.toDouble/training_reg.count())*100

println("Porcentaje de viajes cortos: " + shortTravelPorcentage)
println("Porcentaje de viajes largos: " + longTravelPorcentage)

// tenemos una distribucion 67% - 33% aprox con lo que la complejidad del clasificador no sera muy elevada.



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//TRANSFORMACION
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
println("creamos clase binaria")
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
//DATA SPLIT  TRAINING & TEST  (el split de los datos de training en 2%, 5% ... se hace posteriormente)
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//dividimos en datos de trainning 75% y datos de test 25%
val dataSplits=  datosDF_NY.randomSplit(Array(0.75, 0.25),seed = 8L)
val datosDFLabeled_trainning = dataSplits(0)
val datosDFLabeled_test = dataSplits(1)

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//FEATURIZATION -> PREPARAMOS INSTANCIAS
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
//PIPELINE FEATURIZATION para POKER
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//generamos el pipeline
val featurizationPipelineNY = new Pipeline().setStages(Array(
                                              indexStringFeaturesTodasNumNY,
                                              assemblerFeaturesLabelPipelineNY,
                                              indiceClasePipelineNY))


//val featNY=featurizationPipelineNY.fit(datosDFLabeled_trainning).transform(datosDFLabeled_trainning)




// COMMAND ----------

// DBTITLE 1,SupervisadoClasificadoresBase - TAXI NY(DT, LR, RF,NB) 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//PIPELINE  Y EVALUACION PARA TODOS LOS CLASIFICADORES BASE
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// instancia de todos los clasificadores que vamos a utilizar


val instanciaTrainingPipelineDT = new Supervised(new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineRF = new Supervised(new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineNB = new Supervised(new NaiveBayes().setFeaturesCol("features").setLabelCol("label"))
val instanciaTrainingPipelineLR = new Supervised(new LogisticRegression().setFeaturesCol("features").setLabelCol("label"))
//val instanciaTrainingPipelineLSVM = new Supervised(new LinearSVC().setFeaturesCol("features").setLabelCol("label")


// array de instancias para cada clasificador base
val arrayInstanciasClasificadores:Array[(String,PipelineStage)] = Array(("DT",instanciaTrainingPipelineDT),
                                       ("LR",instanciaTrainingPipelineLR),
                                       ("RF",instanciaTrainingPipelineRF),
                                       ("NB",instanciaTrainingPipelineNB)
                                       //("LSVM",instanciaTrainingPipelineLSVM) 
                                      )


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//preparamos los parametros:  
// 1.- tipo clasificadores, 
// 2.- porcentaje .... (la informacion que esta en el DataFrame de resultados)
// 3.- tipo de datos en esta caso breast cancer wisconsin --> BCW
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val clasificadoresBase =  Array("DT")//Array("DT","LR","RF","NB","LSVM")
val porcentajeLabeled =   Array(0.0001)//Array(0.0001,0.001,0.01,0.05,0.1,0.3)//(0.01,0.05,0.10,0.30) //Array(0.01)
val datosCodigo = "TXNY"


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CREAMOS EL TEMPLATE DE DATAFRAME PARA LOS RESULTADOS
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val NYResultados = generadorDataFrameResultados(datosCodigo,clasificadoresBase,porcentajeLabeled)


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PIPELINE FINAL AUTOMATIZADO PARA CADA CLASIFICADOR
//
// FEATURIZATIONT(BCW)--> CLASIFICACOR (DT) --> RESUTADO
//                    --> CLASIFICACOR (RF) --> RESUTADO
//                    --> CLASIFICACOR (NB) --> RESUTADO
//                    --> .....
//                    --> .....
//                    --> CLASIFICACOR (XX) --> RESUTADO
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


val resultados_NY = generadorModeloResutladosCompleto2(datosDF_NY,
                                      4,
                                      featurizationPipelineNY,
                                      arrayInstanciasClasificadores, // TE Q SER ARRAY
                                      NYResultados,
                                      porcentajeLabeled,
                                      clasificadoresBase)



display(resultados_NY)

// COMMAND ----------

// DBTITLE 1,SelfTrainning - TAXI NY (DT, LR, RF ....)




val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")


val arrayInstanciasClasificadores_NewST:Array[(String,PipelineStage)] = Array(("ST-DT",instanciaTrainingPipelineDT),
                                      ("ST-LR",instanciaTrainingPipelineLR),
                                       ("ST-RF",instanciaTrainingPipelineRF),
                                       ("ST-NB",instanciaTrainingPipelineNB)
                                      )


val clasificadoresBase = Array("ST-DT")//Array("ST-DT","ST-LR","ST-RF","ST-NB")
val porcentajeLabeled = Array(0.0001)//Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val umbral= Array(0.4)//Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val datosCodigo = "TXNY"



val resultsTaxi_ST=generadorDataFrameResultadosSemiSuper(datosCodigo,
                                          clasificadoresBase,
                                          porcentajeLabeled,
                                          umbral    
                                         )


val resultTaxiFinal_ST =generadorModeloSemiAndSuperResutladosCompleto2(featurizationPipelineNY, 
                                                                         4,
                                                                         datosDF_NY,
                                                                         arrayInstanciasClasificadores_NewST,
                                                                         resultsTaxi_ST,
                                                                         porcentajeLabeled,
                                                                         umbral,
                                                                         clasificadoresBase)



display(resultTaxiFinal_ST)

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


val unlabeledProcess = new UnlabeledTransformer ()

var semiSupervisorInstanciaModelo = new SelfTraining(new DecisionTreeClassifier())

// Generamos pipelines para cada modelo
//var pipelineSemiSup = new Pipeline().setStages(Array(featurizationPipelineBCW,unlabeledProcess,instanciaTrainingPipelineDT))

var pipelineSemiSup = new Pipeline().setStages(Array(featurizationPipelineBCW,unlabeledProcess,semiSupervisorInstanciaModelo))
val a = (pipelineSemiSup.fit(datosDF_2).transform(datosDF_2))

//val b =semiSupervisorInstanciaModelo.fit(a).transform(a)


var results = crossValidation(datosDF_2,4,pipelineSemiSup,"selfTraining",semiSupervisorInstanciaModelo)




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
