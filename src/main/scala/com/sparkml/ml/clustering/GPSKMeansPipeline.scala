package com.sparkml.ml.clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

/**
 * 聚类算法--K-means聚类算法--基于用户位置信息的商业选址案例
 * 基于dataFrame + Pipeline实现
 */
object GPSKMeansPipeline {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("").master("local").getOrCreate()
    //加载数据
    val schema = StructType(List(
      //1,30.624806,104.136604,211846
      StructField("userId", StringType, true),
      StructField("lat", DoubleType, true), //纬度
      StructField("lon", DoubleType, true), //经度
      StructField("timestamp", StringType, true)
    ))
    val df = spark.read.schema(schema).csv("data/clustering/gps.csv")

    //todo VectorAssembler 是一个转换器，用来将合并列，提取特征
    val assembler = new VectorAssembler()
      .setInputCols(Array("lat", "lon"))
      .setOutputCol("features")

    //模型学习器
    val kMeans = new KMeans() //.setFeaturesCol(assembler.getOutputCol)
      //设置三个类中心点
      .setK(3)
      //设置迭代次数
      .setMaxIter(12)

    //将转换器和模型学习器封装到管道
    val pipeline = new Pipeline()
      .setStages(Array(assembler, kMeans))

    val pipelineModel = pipeline.fit(df)

    //获取管道中聚类模型转换器
    val meansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    //打印类中心点
    meansModel.clusterCenters.foreach(println)

    //进行预测
    val predictsDF = pipelineModel.transform(df)
    predictsDF.show(false)


    spark.stop()
  }
}
