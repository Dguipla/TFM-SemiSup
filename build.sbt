name := "spark_semisupervised_v3"
version := "1.0.0"
organization := "ubu"
licenses := Seq("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.html"))
scalaVersion := "2.12.10"

// plugin spark packages
sparkVersion := "3.0.1"
//sparkComponents ++= Seq("core","sql","mllib")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.0.1",
  "org.apache.spark" %% "spark-sql" % "3.0.1",
  "org.apache.spark" %% "spark-mllib" % "3.0.1",
)

