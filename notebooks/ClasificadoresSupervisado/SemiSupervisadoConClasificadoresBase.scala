// Databricks notebook source
// miramos los documentos adjuntados
display(dbutils.fs.ls("/FileStore/tables"))

// COMMAND ----------

// DBTITLE 1,FUNCIONES SUPERVISADO (PIPELINES Y AUTOMATIZACION DE LOS PROCESOS)
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


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//preparamos los parametros: clasificadores, porcentaje ....
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// NO SE USA
/*
//buscamos los diferentes clasificadores base con los que vamos a trabajar
val tiposClasificador = breastCancerWisconResultados.select("clasificador").distinct.collect()
// hemos perdido el tipado se lo devolvemos:
val tiposClasiCorrecto = tiposClasificador.map(row => (row(0).asInstanceOf[String]))

//buscamos los diferentes % --> 2, 5 , 20
val por100Etiquetaje = breastCancerWisconResultados.select("porcentajeEtiquetado").distinct.collect()
// hemos perdido el tipado se lo devolvemos
val por100EtiquetajeCorrecto = por100Etiquetaje.map(row => (row(0).asInstanceOf[Double]))
*/

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ENTRENAMIENTO + GENERACION DEL MODELO + RESULTADO CON CONJUNTO DE TEST
//Para cada clasificador:
// 1.- particionamos los datos en funcion del % seleccionado 2%, 5%....
// 2.-Entrenamos el Pipeline Completo (por clasificador) + calculamos la accuracy
//*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//primero hacemos la particion de los datos segun el % y despues con DataSet reducido trabajamos con todos los clasificadores asi succesivamente para cada clasificador de esta forma tenemos el mismo dataSet reducido utilizado para cada modelo de entrenamiento

def generadorModeloResutladosCompleto(datosTest:DataFrame,
                                      datosTraining:DataFrame,
                                      modelosPipeline:Array[(String,Pipeline)], // TE Q SER ARRAY
                                      info:DataFrame,
                                      porcentajeEtiquetado:Array[Double], 
                                      tiposClasi:Array[String]):DataFrame= 
{
  var newdf= info
  for (posPorcentaje <-porcentajeEtiquetado){
    for(posClasi <-tiposClasi) {
      
      // split datos cogemos X% para labeled data de los datos de trainning 
    
      val dataSp=  datosTraining.randomSplit(Array(posPorcentaje, 1-posPorcentaje),seed = 11L)
      val datosTrainningSplited=dataSp(0)
      for (posPipeline <- 0 to (modelosPipeline.size-1)){
        //miramos que clasificador base es para despues poner su resultado en la posicion correspondiente dentro del DataFrame
        if (modelosPipeline(posPipeline)._1.contentEquals(posClasi)){ 
          // entrenamos el modelo DT para diferentes %
          println("porcentaje")
          println(posPorcentaje)
          println("clasificador")
          println(posClasi)
          println("datos")
          println(datosTrainningSplited.count())
          println(datosTest.count())
          var model = modelosPipeline(posPipeline)._2.fit(datosTrainningSplited)

          // montar un for con tres iteraciones para buscar un error coerente cuando los datos pequeños ¿?¿?
          var resultados=model.transform(datosTest).select("prediction", "label")                                                                                        
          //Necesitamos un RDD de predicciones y etiquetas: es MLlib
          var predictionsAndLabelsRDD=resultados.rdd.map(row=> (row.getDouble(0), row.getDouble(1)))

          var metrics= new MulticlassMetrics(predictionsAndLabelsRDD)
          
          // miramos cada uno de los aciertos en funciuon de las profundidades
          var acierto = metrics.accuracy
          println ("resultado")
          println(acierto)
          
          
          //Para calcular el área bajo la curva ROC y el área bajo la curva PR necesitamos la clase BinaryClassificationMetrics
          val metrics2 = new BinaryClassificationMetrics(predictionsAndLabelsRDD)
          //var metrics3= new MulticlassMetrics(predictionsAndLabelsRDD)

           // Área bajo la curva ROC
          var auROC_arbol = metrics2.areaUnderROC

          // Área bajo la curva PR
          var auPR_arbol = metrics2.areaUnderPR
          
          var f1Score = metrics.fMeasure(1)

          //Aqui ira el resultado en el nuevo dataFrame newdf
          //accuracy
          newdf = newdf.withColumn("accuracy",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi, acierto).
                          otherwise (newdf("accuracy")))
          //AUC
           newdf = newdf.withColumn("AUC",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi, auROC_arbol).
                          otherwise (newdf("AUC")))
          //PR
          newdf = newdf.withColumn("PR",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi, auPR_arbol).
                          otherwise (newdf("PR")))
          
          //f1score
          newdf = newdf.withColumn("F1score",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi,f1Score).
                                   otherwise (newdf("F1score")))
          


          //newdf = newdf.withColumn("accuracy",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi, acierto))
          
         // break// Salimos del loop
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

// DBTITLE 1,FUNCIONES SEMISUPERVISADO (PIPELINES Y AUTOMATIZACION DE LOS PROCS)

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

def generadorModeloSemiAndSuperResutladosCompleto [
    FeatureType,
    E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
    M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
  ] (featurization:Pipeline,
     datosTest:DataFrame,
     datosTraining:DataFrame,
     modelosStatgePipeline:Array[(String,PipelineStage)],
     info:DataFrame,
     porcentajeEtiquetado:Array[Double],
     threshold:Array[Double],
     tiposClasi:Array[String]):DataFrame= 
{
  var newdf= info  
  for (posPorcentaje <-porcentajeEtiquetado){
    for (posThres <-threshold){
      for(posClasi <-tiposClasi) {        
        for (posPipeline <- 0 to (modelosStatgePipeline.size-1)){
          //miramos que clasificador base es para despues poner su resultado en la posicion correspondiente dentro del DataFrame
          if (modelosStatgePipeline(posPipeline)._1.contentEquals(posClasi)){ 
            var modeloTipadoSSC = modelosStatgePipeline(posPipeline)._2.asInstanceOf[E]
            // generamos la instacia para el SelfTrainning con los porcentajes y thresholds necesarios
            var semiSupervisorInstanciaModelo = new SelfTraining(modelosStatgePipeline(posPipeline)._2.asInstanceOf[E])
            .setPorcentaje(posPorcentaje)
            .setThreshold(posThres)
            // Generamos pipelines para cada modelo
            var pipelineSemiSupervisado = new Pipeline().setStages(Array(featurization,
                                                                         semiSupervisorInstanciaModelo)) // cogemos solo el stage de la array 
            var model = pipelineSemiSupervisado.fit(datosTraining)

            // montar un for con tres iteraciones para buscar un error coerente cuando los datos pequeños ¿?¿?
            var resultados=model.transform(datosTest).select("prediction", "label")                                                                                        
            //Necesitamos un RDD de predicciones y etiquetas: es MLlib
            var predictionsAndLabelsRDD=resultados.rdd.map(row=> (row.getDouble(0), row.getDouble(1)))
            var metrics= new MulticlassMetrics(predictionsAndLabelsRDD)

            // miramos cada uno de los aciertos en funciuon de las profundidades
            var acierto = metrics.accuracy

            //Para calcular el área bajo la curva ROC y el área bajo la curva PR necesitamos la clase BinaryClassificationMetrics
            val metrics2 = new BinaryClassificationMetrics(predictionsAndLabelsRDD)
            //var metrics3= new MulticlassMetrics(predictionsAndLabelsRDD)

             // Área bajo la curva ROC
            var auROC_arbol = metrics2.areaUnderROC

            // Área bajo la curva PR
            var auPR_arbol = metrics2.areaUnderPR

            var f1Score = metrics.fMeasure(1)

            //Resultados para SemiSupervisado y supervisado
            //accuracy
            newdf = newdf.withColumn("accuracy",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                     newdf("clasificador")===posClasi &&
                                                     newdf("threshold")=== posThres
                                                     , acierto).otherwise (newdf("accuracy")))
            //AUC
             newdf = newdf.withColumn("AUC",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                 newdf("clasificador")===posClasi &&
                                                 newdf("threshold")=== posThres
                                                 , auROC_arbol).otherwise (newdf("AUC")))
            //PR
            newdf = newdf.withColumn("PR",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi, auPR_arbol).
                            otherwise (newdf("PR")))

            //f1score
            newdf = newdf.withColumn("F1score",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                    newdf("clasificador")===posClasi &&
                                                    newdf("threshold")=== posThres
                                                    ,f1Score).otherwise (newdf("F1score")))
            //iter
            newdf = newdf.withColumn("iteracion",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                      newdf("clasificador")===posClasi &&
                                                      newdf("threshold")=== posThres
                                                      , semiSupervisorInstanciaModelo.getIter()).otherwise (newdf("iteracion")))
            //LabeledInicial 
            newdf = newdf.withColumn("LabeledInicial",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi
                                                           , semiSupervisorInstanciaModelo.getDataLabeledIni()).otherwise (newdf("LabeledInicial")))
            //UnLabeledInicial
            newdf = newdf.withColumn("UnLabeledInicial",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                             newdf("clasificador")===posClasi &&
                                                             newdf("threshold")=== posThres
                                                           , semiSupervisorInstanciaModelo.getUnDataLabeledIni()).otherwise (newdf("UnLabeledInicial")))   
            //LabeledFinal
            newdf = newdf.withColumn("LabeledFinal",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                         newdf("clasificador")===posClasi &&
                                                         newdf("threshold")=== posThres
                                                         , semiSupervisorInstanciaModelo.getDataLabeledFinal()).otherwise (newdf("LabeledFinal")))
            //UnLabeledFinal
            newdf = newdf.withColumn("UnLabeledFinal",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                           newdf("clasificador")===posClasi &&
                                                           newdf("threshold")=== posThres
                                                           , semiSupervisorInstanciaModelo.getUnDataLabeledFinal()).otherwise (newdf("UnLabeledFinal")))
            //Etiquetado % final
            // calculo -> 1 -(semiSupervisorInstanciaModelo.getUnDataLabeledFinal() / semiSupervisorInstanciaModelo.getUnDataLabeledIni())
            newdf = newdf.withColumn("porEtiFinal",when(newdf("porcentajeEtiquetado")===posPorcentaje && 
                                                        newdf("clasificador")===posClasi &&
                                                        newdf("threshold")=== posThres
                                                           , (1 -(semiSupervisorInstanciaModelo.getUnDataLabeledFinal().toDouble /
                                                                 semiSupervisorInstanciaModelo.getUnDataLabeledIni().toDouble))).otherwise (newdf("UnLabeledFinal"))) 
            
            
          }  
        }
      } 
    }
  }
  newdf
}





// COMMAND ----------

// DBTITLE 1,SSC - SelfTrainning - ClaseCompleta (Nuevo Clasificador)
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
    val baseClassifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M],
    
  ) extends 
org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]
//org.apache.spark.ml.classification.ClassifierParams[FeatureType, E, M]
with Serializable {
  
  //variables
  var porcentajeLabeled:Double =0.002
  var threshold:Double=0.7
  var maxIter:Int=10
  var criterion:String= "threshold"
  var kBest:Int=10 // porcentaje ...
  var countDataLabeled:Long = _
  var countDataUnLabeled:Long = _
  var dataLabeledIni:Long =_
  var dataUnLabeledIni:Long = _
  var iter:Int = _
  
  //uid
  def this(classifier: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M]) =
    this(Identifiable.randomUID("selfTrainning"), classifier)
  
  

  //SETTERS

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
    //splitamos con el % seleccionado 
    val dataSplited = resplit(dataset,porcentajeLabeled)
    var dataLabeled = dataSplited(0).toDF
    var dataUnLabeled = dataSplited(1).toDF
    //guardamos el tamaño inicial de los datos etiquetados y no etiquetados
    dataLabeledIni = dataLabeled.count()
    dataUnLabeledIni = dataUnLabeled.count()
    //seleccionamos features and labels
    dataLabeled = dataLabeled.select("features","label")
    dataUnLabeled = dataUnLabeled .select("features","label")
    countDataLabeled = dataLabeled.count()
    countDataUnLabeled = dataUnLabeled.count()

    // total data inicial
    println ("EMPIEZA EL SELF-TRAINING")
    println ("threshold: "+threshold)
    println ("porcentaje Labeled: " + porcentajeLabeled)
    println("iterancion: 0" )
    println ("total labeled Inicial " + countDataLabeled)
    println ("total unlabeled Inicial " + countDataUnLabeled)

    // generamos modelo y predecimos las unlabeled data
    var modeloIterST = baseClassifier.fit(dataLabeled)
    var prediIterST = modeloIterST.transform(dataUnLabeled)
    prediIterST

    while ((iter<maxIter) && (countDataUnLabeled>0)){

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


      if (countDataUnLabeled>0 && iter<maxIter ){
        println ("Trainning... Next Iteration")
        modeloIterST = baseClassifier.fit(dataLabeled)
        prediIterST = modeloIterST.transform(dataUnLabeled)
        iter = iter+1
      }
      else{ //final del ciclo
        
        modeloIterST = baseClassifier.fit(dataLabeled)
        //iter = maxIter
      }
    }
    println("ACABAMOS SELF-TRAINING Y RETORNAMOS MODELO")
    modeloIterST


  }
  
  override def copy(extra: org.apache.spark.ml.param.ParamMap):E = defaultCopy(extra)
}





// COMMAND ----------

// DBTITLE 1,SupervisadoClasificadoresBase - BCW  (DT, LR, RF,NB, LSVM)
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

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//PIPELINE  Y EVALUACION PARA TODOS LOS CLASIFICADORES BASE
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// instancia de todos los clasificadores que vamos a utilizar
val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLSVM = new LinearSVC().setFeaturesCol("features").setLabelCol("label")


// array de instancias para cada clasificador base
val arrayInstanciasClasificadores:Array[(String,PipelineStage)] = Array(("DT",instanciaTrainingPipelineDT),
                                       ("LR",instanciaTrainingPipelineLR),
                                       ("RF",instanciaTrainingPipelineRF),
                                       ("NB",instanciaTrainingPipelineNB),
                                       ("LSVM",instanciaTrainingPipelineLSVM) 
                                      )



//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//preparamos los parametros:  
// 1.- tipo clasificadores, 
// 2.- porcentaje .... (la informacion que esta en el DataFrame de resultados)
// 3.- tipo de datos en esta caso breast cancer wisconsin --> BCW
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val clasificadoresBase = Array ("DT","LR","RF","NB","LSVM")
val porcentajeLabeled = Array(0.01,0.05,0.08,0.10,0.20,0.30)
val datosCodigo = "BCW"


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CREAMOS EL TEMPLATE DE DATAFRAME PARA LOS RESULTADOS
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val breastCancerWisconResultados = generadorDataFrameResultados(datosCodigo,clasificadoresBase,porcentajeLabeled)


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


// generamos los pipelines para cada clasificador base:
val generadorPipelinePorClasificador = generadorPipelinesPorClasificador(arrayInstanciasClasificadores,featurizationPipelineBCW)

//una vez tenemos el array de pipelines clasificadores entrenamos el modelo, generamos los resultados y los guardamos en un DataFrame
val resultados_BCW=generadorModeloResutladosCompleto(datosDFLabeled_test,
                                  datosDFLabeled_trainning,
                                  generadorPipelinePorClasificador,
                                  breastCancerWisconResultados,
                                  porcentajeLabeled,
                                  clasificadoresBase)

display(resultados_BCW)



// COMMAND ----------

// DBTITLE 1,SelfTrainning - BCW (DT, LR, RF ....)
/*
// intancia para SelfTrainning Clasificadore Base
val instanciaTrainingPipelineST_DT = new SelfTraining(instanciaTrainingPipelineDT)
val instanciaTrainingPipelineST_RF = new SelfTraining(instanciaTrainingPipelineRF)
val instanciaTrainingPipelineST_NB = new SelfTraining(instanciaTrainingPipelineNB)
//val instanciaTrainingPipelineST_LR = new SelfTraining(instanciaTrainingPipelineLSVM)

// array de instancias para cada clasificadorBase en SelfTrainning
val arrayInstanciasClasificadoresST:Array[(String,PipelineStage)] = Array(("ST-DT",instanciaTrainingPipelineDT),
                                       ("ST-LR",instanciaTrainingPipelineLR),
                                       ("ST-RF",instanciaTrainingPipelineRF),
                                       ("ST-NB",instanciaTrainingPipelineNB))
*/

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


val resultadosBCW_STFinal =generadorModeloSemiAndSuperResutladosCompleto(featurizationPipelineBCW,
                                              datosDFLabeled_test,
                                              datosDFLabeled_trainning,
                                              arrayInstanciasClasificadores_BCW_NewST,
                                              resultadosBCW_ST,
                                              porcentajeLabeledBCW_ST,
                                              umbral,
                                              clasificadoresBase)

display(resultadosBCW_STFinal)

// COMMAND ----------

resultadosBCW_STFinal.select("porcentajeEtiquetado").distinct.collect

// COMMAND ----------

// DBTITLE 1,SupervisadoClasificadoresBase - Adult Income(DT, LR, RF,NB, LSVM)
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


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//PIPELINE  Y EVALUACION PARA TODOS LOS CLASIFICADORES BASE
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// instancia de todos los clasificadores que vamos a utilizar
val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
instanciaTrainingPipelineDT.setMaxBins(42) // se necesita el numero maximo de opciones en las features ya que es superior que el standard
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
instanciaTrainingPipelineRF.setMaxBins(42)
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLSVM = new LinearSVC().setFeaturesCol("features").setLabelCol("label")


// array de instancias para cada clasificador base
val arrayInstanciasClasificadores:Array[(String,PipelineStage)] = Array(("DT",instanciaTrainingPipelineDT),
                                       ("LR",instanciaTrainingPipelineLR),
                                       ("RF",instanciaTrainingPipelineRF),
                                       ("NB",instanciaTrainingPipelineNB),
                                       ("LSVM",instanciaTrainingPipelineLSVM) 
                                      )



//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//preparamos los parametros:  
// 1.- tipo clasificadores, 
// 2.- porcentaje .... (la informacion que esta en el DataFrame de resultados)
// 3.- tipo de datos en esta caso breast cancer wisconsin --> BCW
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val clasificadoresBase = Array("DT","LR","RF","NB","LSVM")
val porcentajeLabeled = Array(0.002,0.005,0.01,0.05,0.10,0.30)
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


// generamos los pipelines para cada clasificador base:
val generadorPipelinePorClasificadorAdult = generadorPipelinesPorClasificador(arrayInstanciasClasificadores,featurizationPipelineAdult)

val resultados_ADULT=generadorModeloResutladosCompleto(datosDFLabeled_testAdult,
                                  datosDFLabeled_trainningAdult,
                                  generadorPipelinePorClasificadorAdult,
                                  adultResultados,
                                  porcentajeLabeled,
                                  clasificadoresBase)

resultados_ADULT.show()








// COMMAND ----------

display(resultados_ADULT)

// COMMAND ----------

// DBTITLE 1,SelfTrainning - ADULT (DT, LR, RF ....)
/*
// intancia para SelfTrainning Clasificadore Base
val instanciaTrainingPipelineST_DT = new SelfTraining(instanciaTrainingPipelineDT)
val instanciaTrainingPipelineST_RF = new SelfTraining(instanciaTrainingPipelineRF)
val instanciaTrainingPipelineST_NB = new SelfTraining(instanciaTrainingPipelineNB)
//val instanciaTrainingPipelineST_LR = new SelfTraining(instanciaTrainingPipelineLSVM)

// array de instancias para cada clasificadorBase en SelfTrainning
val arrayInstanciasClasificadoresST:Array[(String,PipelineStage)] = Array(("ST-DT",instanciaTrainingPipelineDT),
                                       ("ST-LR",instanciaTrainingPipelineLR),
                                       ("ST-RF",instanciaTrainingPipelineRF),
                                       ("ST-NB",instanciaTrainingPipelineNB))
*/

val arrayInstanciasClasificadoresNewSSC:Array[(String,PipelineStage)] = Array(("ST-DT",instanciaTrainingPipelineDT),
                                       ("ST-LR",instanciaTrainingPipelineLR),
                                       ("ST-RF",instanciaTrainingPipelineRF),
                                       ("ST-NB",instanciaTrainingPipelineNB)
                                      )


val clasificadoresBase = Array("ST-DT","ST-LR","ST-RF","ST-NB")//Array ("DT","LR","RF","NB","LSVM")
val porcentajeLabeled = Array(0.002,0.005,0.01,0.05,0.10,0.30)//Array(0.001,0.005,0.01,0.05,0.10,0.30)
val umbral= Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val datosCodigo = "ADULT"



val resultadosAdultST=generadorDataFrameResultadosSemiSuper(datosCodigo,
                                          clasificadoresBase,
                                          porcentajeLabeled,
                                          umbral    
                                         )


val resultadosAdultSTFinal =generadorModeloSemiAndSuperResutladosCompleto(featurizationPipelineAdult,
                                              datosDFLabeled_testAdult,
                                              datosDFLabeled_trainningAdult,
                                              arrayInstanciasClasificadoresNewSSC,
                                              resultadosAdultST,
                                              porcentajeLabeled,
                                              umbral,
                                              clasificadoresBase)



// COMMAND ----------

// DBTITLE 1,SupervisadoClasificadoresBase - Poker Hand(DT, LR, RF,NB, LSVM)
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


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//PIPELINE  Y EVALUACION PARA TODOS LOS CLASIFICADORES BASE
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// instancia de todos los clasificadores que vamos a utilizar
val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLSVM = new LinearSVC().setFeaturesCol("features").setLabelCol("label")


// array de instancias para cada clasificador base
val arrayInstanciasClasificadores:Array[(String,PipelineStage)] = Array(("DT",instanciaTrainingPipelineDT),
                                       ("LR",instanciaTrainingPipelineLR),
                                       ("RF",instanciaTrainingPipelineRF),
                                       ("NB",instanciaTrainingPipelineNB),
                                       ("LSVM",instanciaTrainingPipelineLSVM) 
                                      )


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//preparamos los parametros:  
// 1.- tipo clasificadores, 
// 2.- porcentaje .... (la informacion que esta en el DataFrame de resultados)
// 3.- tipo de datos en esta caso breast cancer wisconsin --> BCW
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val clasificadoresBase = Array("DT","LR","RF","NB","LSVM")
val porcentajeLabeled = Array(0.0001,0.001,0.01,0.05,0.1,0.3)//Array(0.0001) //Array(0.0001,0.001,0.01,0.1,0.3)
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


// generamos los pipelines para cada clasificador base:
val generadorPipelinePorClasificador = generadorPipelinesPorClasificador(arrayInstanciasClasificadores,featurizationPipelinePoker)

//una vez tenemos el array de pipelines clasificadores entrenamos el modelo, generamos los resultados y los guardamos en un DataFrame
val resultados_Poker=generadorModeloResutladosCompleto(datosDFLabeled_testPOKER,
                                  datosDFLabeled_trainningPOKER,
                                  generadorPipelinePorClasificador,
                                  pokerResultados,
                                  porcentajeLabeled,
                                  clasificadoresBase)

resultados_Poker.show()
//resultados_Poker.write.save(PATH + "resultsPOKER")


// COMMAND ----------

display(resultados_Poker)

// COMMAND ----------

// DBTITLE 1,SelfTrainning - POKER (DT, LR, RF ....)
val arrayInstanciasClasificadores_POKER_NewST:Array[(String,PipelineStage)] = Array(("ST-DT",instanciaTrainingPipelineDT),
                                       ("ST-LR",instanciaTrainingPipelineLR),
                                       ("ST-RF",instanciaTrainingPipelineRF),
                                       ("ST-NB",instanciaTrainingPipelineNB)
                                      )


val clasificadoresBase = Array("ST-DT","ST-LR","ST-RF","ST-NB")//Array ("DT","LR","RF","NB","LSVM")
val porcentajeLabeledPOKER_ST = Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val umbral= Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val datosCodigo = "POKER"



val resultadosPOKER_ST=generadorDataFrameResultadosSemiSuper(datosCodigo,
                                          clasificadoresBase,
                                          porcentajeLabeledPOKER_ST,
                                          umbral    
                                         )


val resultadosPOKER_STFinal =generadorModeloSemiAndSuperResutladosCompleto(featurizationPipelinePoker,
                                              datosDFLabeled_testPOKER,
                                              datosDFLabeled_trainningPOKER,
                                              arrayInstanciasClasificadores_POKER_NewST,
                                              resultadosPOKER_ST,
                                              porcentajeLabeledPOKER_ST,
                                              umbral,
                                              clasificadoresBase)

display(resultadosPOKER_STFinal)

// COMMAND ----------

// DBTITLE 1,SupervisadoClasificadoresBase - TAXI NY(DT, LR, RF,NB, LSVM)
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


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//PIPELINE  Y EVALUACION PARA TODOS LOS CLASIFICADORES BASE
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// instancia de todos los clasificadores que vamos a utilizar
val instanciaTrainingPipelineDT = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineRF = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineNB = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")
val instanciaTrainingPipelineLSVM = new LinearSVC().setFeaturesCol("features").setLabelCol("label")


// array de instancias para cada clasificador base
val arrayInstanciasClasificadores:Array[(String,PipelineStage)] = Array(("DT",instanciaTrainingPipelineDT),
                                       ("LR",instanciaTrainingPipelineLR),
                                       ("RF",instanciaTrainingPipelineRF),
                                       ("NB",instanciaTrainingPipelineNB),
                                       ("LSVM",instanciaTrainingPipelineLSVM) 
                                      )


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//preparamos los parametros:  
// 1.- tipo clasificadores, 
// 2.- porcentaje .... (la informacion que esta en el DataFrame de resultados)
// 3.- tipo de datos en esta caso breast cancer wisconsin --> BCW
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val clasificadoresBase = Array("DT","LR","RF","NB","LSVM")
val porcentajeLabeled = Array(0.0001,0.001,0.01,0.05,0.1,0.3)//(0.01,0.05,0.10,0.30)
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


// generamos los pipelines para cada clasificador base:
val generadorPipelinePorClasificador = generadorPipelinesPorClasificador(arrayInstanciasClasificadores,featurizationPipelineNY)

//una vez tenemos el array de pipelines clasificadores entrenamos el modelo, generamos los resultados y los guardamos en un DataFrame
val resultados_NY=generadorModeloResutladosCompleto(datosDFLabeled_test,
                                  datosDFLabeled_trainning,
                                  generadorPipelinePorClasificador,
                                  NYResultados,
                                  porcentajeLabeled,
                                  clasificadoresBase)

display(resultados_NY)


// COMMAND ----------

// DBTITLE 1,SelfTrainning - TAXI NY (DT, LR, RF ....)
val arrayInstanciasClasificadores_TX_NewST:Array[(String,PipelineStage)] = Array(("ST-DT",instanciaTrainingPipelineDT),
                                       ("ST-LR",instanciaTrainingPipelineLR),
                                       ("ST-RF",instanciaTrainingPipelineRF),
                                       ("ST-NB",instanciaTrainingPipelineNB)
                                      )


val clasificadoresBase = Array("ST-DT","ST-LR","ST-RF","ST-NB")//Array ("DT","LR","RF","NB","LSVM")
val porcentajeLabeledTX_ST = Array(0.0001,0.001,0.01,0.05,0.1,0.3)
val umbral= Array(0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)
val datosCodigo = "TXNY"



val resultados_ST=generadorDataFrameResultadosSemiSuper(datosCodigo,
                                          clasificadoresBase,
                                          porcentajeLabeledTX_ST,
                                          umbral    
                                         )


val resultadosTX_STFinal =generadorModeloSemiAndSuperResutladosCompleto(featurizationPipelineNY,
                                              datosDFLabeled_test,
                                              datosDFLabeled_trainning,
                                              arrayInstanciasClasificadores_TX_NewST,
                                              resultados_ST,
                                              porcentajeLabeledTX_ST,
                                              umbral,
                                              clasificadoresBase)

display(resultadosTX_STFinal)

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
