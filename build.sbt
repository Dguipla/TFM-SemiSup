name := "spark_semisupervised_v3"
version := "1.0.0"
organization := "ubu"
licenses := Seq("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0$
scalaVersion := "2.12.10"

// plugin spark packages
sparkVersion := "2.4.5"
sparkComponents ++= Seq("sql", "mllib")
