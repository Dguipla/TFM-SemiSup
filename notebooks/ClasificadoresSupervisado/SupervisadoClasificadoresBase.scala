// Databricks notebook source
// miramos los documentos adjuntados
display(dbutils.fs.ls("/FileStore/tables"))

// COMMAND ----------

// DBTITLE 1,FUNCIONES  (PIPELINES Y AUTOMATIZACION DE LOS PROCESOS)
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


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// GENERADOR DE DATA FRAME RESULTADOS (solo el esqueleto del DATA FRAME)
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/* QUEDARIA:

+----+------------+--------------------+--------+
|data|clasificador|porcentajeEtiquetado|accuracy|
+----+------------+--------------------+--------+
| BCW|          DT|                0.02|     x.x|
| BCW|          DT|                0.05|     x.x|
| BCW|          DT|                 0.1|     x.x|
| BCW|          DT|                 0.3|     x.x|
| BCW|          LR|                0.02|     x.x|
....
....
*/

def generadorDataFrameResultados(datos:String,clasificadores:Array[String],porcentaje:Array[Double]):DataFrame= 
{
  // creamos la array de salida con el type y el tamaño
  var seqValores=Seq[(String, String, Double, Double)]() 
  var posicion = 0
  for(posClasi <-clasificadores) {
    for(posPorce <-porcentaje){
      seqValores = seqValores :+ (datos,posClasi,posPorce,0.00)
    }
  }
  // generamos el DataFrame que sera la salida
  spark.createDataFrame(seqValores).toDF("data", "clasificador", "porcentajeEtiquetado", "accuracy")
}


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
    
      val dataSp=  datosTraining.randomSplit(Array(posPorcentaje, 1-posPorcentaje))//,seed = 11L)
      val datosTrainningSplited= dataSp(0)
      
      for (posPipeline <- 0 to (modelosPipeline.size-1)){

        //miramos que clasificador base es para despues poner su resultado en la posicion correspondiente dentro del DataFrame
        if (modelosPipeline(posPipeline)._1.contentEquals(posClasi)){ 
        
          // entrenamos el modelo DT para diferentes %
          var model = modelosPipeline(posPipeline)._2.fit(datosTrainningSplited)

          // montar un for con tres iteraciones para buscar un error coerente cuando los datos pequeños ¿?¿?
          var resultados=model.transform(datosTest).select("prediction", "label")                                                                                        
          //Necesitamos un RDD de predicciones y etiquetas: es MLlib
          var predictionsAndLabelsRDD=resultados.rdd.map(row=> (row.getDouble(0), row.getDouble(1)))

          var metrics= new MulticlassMetrics(predictionsAndLabelsRDD)
          
          // miramos cada uno de los aciertos en funciuon de las profundidades
          var acierto = metrics.accuracy

          //Aqui ira el resultado en el nuevo dataFrame newdf
          newdf = newdf.withColumn("accuracy",when(newdf("porcentajeEtiquetado")===posPorcentaje && newdf("clasificador")===posClasi, acierto).
                          otherwise (newdf("accuracy")))
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
                                                  instanciasModelos(posicion)._2 // cogemos solo el stage del pipeline
                                                   ))
    
    salida(posicion) = (instanciasModelos(posicion)._1,pipeline) // ponemos en la array el clasificador con su pipeline--> clasificador,pipeline
    posicion = posicion+1
    
  }
  salida
}


// COMMAND ----------

// DBTITLE 1,SupervisadoClasificadoresBase - Breast Cancser Wisconsin  (DT, LR, RF,NB, LSVM)
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
val dataSplits= datosDF_2.randomSplit(Array(0.75, 0.25))
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
val indiceClasePipeline = new StringIndexer().setInputCol("diagnosis").setOutputCol("label")

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
val porcentajeLabeled = Array(0.02,0.05,0.10,0.30)
val datosCodigo = "BCW"


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CREAMOS EL TEMPLATE DE DATAFRAME PARA LOS RESULTADOS
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

val breastCancerWisconResultados = generadorDataFrameResultados(datosCodigo,clasificadoresBase,porcentajeLabeled)


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
val generadorPipelinePorClasificador = generadorPipelinesPorClasificador(arrayInstanciasClasificadores,featurizationPipelineBCW)

//una vez tenemos el array de pipelines clasificadores entrenamos el modelo, generamos los resultados y los guardamos en un DataFrame
val resultados_BCW=generadorModeloResutladosCompleto(datosDFLabeled_test,
                                  datosDFLabeled_trainning,
                                  generadorPipelinePorClasificador,
                                  breastCancerWisconResultados,
                                  porcentajeLabeled,
                                  clasificadoresBase)

resultados_BCW.show()



// COMMAND ----------

display(resultados_BCW.withColumn("accuracy",col("accuracy")*100))
