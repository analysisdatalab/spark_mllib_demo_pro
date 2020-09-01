package com.sparkml.ml.clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

/**
 * 聚类算法--K-means聚类算法--基于用户位置信息的商业选址案例
 * 基于dataFrame实现
 */
object GPSKMeans {
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
    //VectorAssembler 是一个转换器，用来将合并列，提取特征
    val assembler = new VectorAssembler()
      .setInputCols(Array("lat", "lon"))
      .setOutputCol("features")

    //对df数据进行转换，合并lat 和lon这两列作为features特征
    val df2 = assembler.transform(df)

    //模型学习器
    val kMeans = new KMeans()
      //设置三个类中心点
      .setK(3)
      //设置迭代次数
      .setMaxIter(12)

    //调用模型学习器的fit方法，拿到转换器
    val kmeansModel = kMeans.fit(df2)

    //打印三个类中心点
    /**
     * [30.64631364707616,104.08771217582184]
     * [30.894232056457263,103.65216802517202]
     * [30.656352647553756,104.01705392292364]
     */
    kmeansModel.clusterCenters.foreach(println)

    //使用kmeansModel转换器进行预测
    val predictsDF = kmeansModel.transform(df2)
    //预测出来的结果用类中心的角标标识
    /**
     * +------+---------+----------+---------+----------------------+----------+
     * |userId|lat      |lon       |timestamp|features              |prediction|
     * +------+---------+----------+---------+----------------------+----------+
     * |1     |30.624806|104.136604|211846   |[30.624806,104.136604]|0         |
     * |1     |30.624809|104.136612|211815   |[30.624809,104.136612]|0         |
     * |1     |30.624811|104.136587|212017   |[30.624811,104.136587]|0         |
     * +------+---------+----------+---------+----------------------+----------+
     */
    predictsDF.show(false)

    /**
     * 使用评估器进行评估
     *
     */
    // 聚类评估器，对预测结果进行评估
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictsDF)
    //使用欧式距离平方指标进行评估
    println(s"Silhouette with squared euclidean distance = $silhouette")


    spark.stop()
  }
}
