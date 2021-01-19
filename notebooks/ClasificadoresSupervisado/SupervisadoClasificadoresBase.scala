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
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics


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

val clasificadoresBase = Array("DT")//Array ("DT","LR","RF","NB","LSVM")
val porcentajeLabeled = Array(0.01)//Array(0.01,0.05,0.10,0.30)
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

resultados_BCW.show()



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

val clasificadoresBase = Array ("DT","LR","RF","NB","LSVM")
val porcentajeLabeled = Array(0.001,0.005,0.01,0.05,0.10,0.30)
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

//display(resultados_ADULT)

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
val datosDFLabeled_trainning = dataSplits(0)
val datosDFLabeled_test = dataSplits(1)

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


val featPoker=featurizationPipelinePoker.fit(datosDFLabeled_trainning).transform(datosDFLabeled_trainning)


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
val porcentajeLabeled = Array(0.0001,0.001,0.01,0.1,0.3)
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
val resultados_Poker=generadorModeloResutladosCompleto(datosDFLabeled_test,
                                  datosDFLabeled_trainning,
                                  generadorPipelinePorClasificador,
                                  pokerResultados,
                                  porcentajeLabeled,
                                  clasificadoresBase)

resultados_Poker.show()
//resultados_Poker.write.save(PATH + "resultsPOKER")


// COMMAND ----------

display(resultados_Poker)

// COMMAND ----------

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


val featNY=featurizationPipelineNY.fit(datosDFLabeled_trainning).transform(datosDFLabeled_trainning)


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
val porcentajeLabeled = Array(0.0001,0.001,0.01,0.1,0.3)//(0.01,0.05,0.10,0.30)
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

resultados_NY.show()


// COMMAND ----------

display (resultados_NY)
