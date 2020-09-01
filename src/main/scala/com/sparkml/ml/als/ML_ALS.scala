package com.sparkml.ml.als

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

/**
 * Spark ML--协同过滤--使用ALS算法实现给用户推荐电影、给电影推荐用户、模型评估
 */
object ML_ALS {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("ML_ALS").master("local[4]").getOrCreate()
    val df = spark.read.schema(
      StructType(List(
        StructField("user", IntegerType, true),
        StructField("item", IntegerType, true),
        StructField("rating", DoubleType, true)
      ))
    ).option("sep", "\t")
      .csv("data/als/movielens/ml-100k/u.data")

    df.cache().count()

    //将数据划分成训练集和测试集
    val Array(trainingDF, testDF) = df.randomSplit(Array(0.7, 0.3))

    //创建一个模型训练器（模型学习器，算法）
    val als = new ALS()
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      //将新的item或user预测的结果NaN，不参与到评估器中进行评估
      .setColdStartStrategy("drop")
      .setMaxIter(5)
      .setRegParam(0.01)
    //进行训练
    val alsModel = als.fit(trainingDF)

    //对用户的评分进行预测
    val predictionsDF = alsModel.transform(testDF)

    //打印预测的结果
    predictionsDF.show(false)
    //创建模型评估器，als模型是一个回归模型，所有需要创建回归模型评估器，来进行评估，评估模型的好坏
    val evaluator = new RegressionEvaluator()
      //评估指标 均方根误差进行评估RMSE
      .setMetricName("rmse")
      //实际评分
      .setLabelCol("rating")
      //预测评分
      .setPredictionCol("prediction")
    //对模型预测的结果进行评估
    val rmse = evaluator.evaluate(predictionsDF)

    //我们在进行训练的时候，有的用户或物品没有参与到模型的训练，那么在进行预测的时候，就会给出一个NaN值，那么在进行模型评估的
    //时候，这些NaN也会参数到评估中，那么评估器就会显示NaN，甚至抛出异常，所以这对这种情况，需要做一个设置
    println(s"Root-mean-square error = $rmse")

    //给每一个用户推荐10部电影
    val userRecsDF = alsModel.recommendForAllUsers(10)
    println("给每一个用户推荐10部电影")
    userRecsDF.show(false)

    // 为每一部电影推荐10个用户
    val movieRecsDF = alsModel.recommendForAllItems(10)
    println("为每一部电影推荐10个用户")
    movieRecsDF.show(false)

    //给指定的一组用户推荐10部电影
    /**
     * def recommendForUserSubset(
     * dataset: Dataset[_],//一组用户
     * numItems: Int//给这组用户推荐多少部电影
     * ): DataFrame
     */
    val userDS = spark.createDataFrame(
      Seq((196, 196), (186, 186), (22, 22))
    ).toDF("user", "userId").select("user").distinct()

    // val recommend10MovieForUserDF = alsModel.recommendForUserSubset(trainingDF.select("user").distinct().limit(3), 10)
    val recommend10MovieForUserDF = alsModel.recommendForUserSubset(userDS, 10)
    println("给指定的一组用户推荐10部电影")
    recommend10MovieForUserDF.show(false)

    /**
     * 给指定的一组电影推荐10个用户
     */
    val moviesDS = spark.createDataFrame(
      Seq((242, 242), (302,302 ), (377, 377))
    ).toDF("item","itemId" ).select("item").distinct()

    //val recommend10UserForMovieDF = alsModel.recommendForItemSubset(trainingDF.select("item").distinct().limit(3), 10)
    val recommend10UserForMovieDF = alsModel.recommendForItemSubset(moviesDS, 10)
    println("给指定的一组电影推荐10个用户")
    recommend10UserForMovieDF.show(false)

    spark.stop()
  }
}
