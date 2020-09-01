package com.sparkml.recommand

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 *
 * 需求：通过ALS算法在电影评分数据集上进行训练，获取推荐模型，给用户推荐商品
 * userId    productId    rating    timestamp
 * 196   	     242	      3	        881250949
 *
 * 数据（Rating）-->ALS算法进行训练--->矩阵分解模型（MatrixFactorizationModel）-->预测和推荐
 */
object MovieALSReCommand2 {
  def main(args: Array[String]): Unit = {
    //设置日志级别
    Logger.getLogger("org").setLevel(Level.WARN)
    val sparkConf = new SparkConf().setAppName("MovieALSReCommand").setMaster("local")
    val sc = new SparkContext(sparkConf)
    //sc.setLogLevel("WARN")

    //todo 1，加载数据
    val ratingRDD = sc.textFile("data/als/movielens/ml-100k/u.data")
      .mapPartitions(part => {
        part.map(line => {
          /**
           * case class Rating @Since("0.8.0") (
           *
           * @Since("0.8.0") user: Int,  用户id
           * @Since("0.8.0") product: Int, 产品id
           * @Since("0.8.0") rating: Double 用户对产品的评分
           *                 )
           */

          val fields = line.split("\t")
          Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
        })
      })

    ratingRDD.cache()

    //将数据集划分成训练集和测试集
    val Array(trainRDD, testRDD) = ratingRDD.randomSplit(Array(0.8, 0.2))


    /**
     * 模型评估方法，返回模型对应的RMSE均方根误差
     *
     * @param model
     * @param testRDD
     */
    def evaluationModel(model: MatrixFactorizationModel, testRDD: RDD[Rating]) = {
      //用户对产品实际评分
      val actualRatingRDD = testRDD.map(r => ((r.user, r.product), r.rating))
      //从actualRatingRDD抽取出用户和产信息出来，以便模型对每条数据进行预测
      val userAndProductRDD = actualRatingRDD.map(_._1)
      //使用矩阵分解模型对每条数据进行预测，得出一个预测评分
      val predictRatingRDD = model.predict(userAndProductRDD).map(r => ((r.user, r.product), r.rating))
      //((user,product),(predictRating,actualRating))
      val predictAndActualRatingRDD = predictRatingRDD.join(actualRatingRDD)
        //取出(predictRating,actualRating)
        .map(_._2)

      //使用回归模型评估器对预测的结果进行评估
      val metrics = new RegressionMetrics(predictAndActualRatingRDD)
      metrics.rootMeanSquaredError
    }

    //调整训练超参数
    val tuple4Array = for {
      rank <- Array(3, 5, 10)
      iteration <- Array(5, 10, 15)
    } yield {

      val model: MatrixFactorizationModel = ALS.train(trainRDD, rank, iteration)
      //调用模型评估方法对模型预测的结果进行评估
      val rmse = evaluationModel(model, testRDD)
      println(s"rmse=${rmse} rank=${rank}  iteration=${iteration}")
      (rmse, rank, iteration, model)
    }
    val tuple4 = tuple4Array.sortBy(_._1).take(1)(0)
    println("=================================")
    println(s"最佳模型参数：RMSE=${tuple4._1}  rank=${tuple4._2} iteration=${tuple4._3}")

    //保存最佳模型，以便下次使用
    //val bestMode = tuple4._4
    //bestMode.save(sc, "data/bestModel/")
    //将最佳模型加载出来，进行预测或推荐
    //val bestModel2 = MatrixFactorizationModel.load(sc, "data/bestModel/")
    //val predictRating = bestModel2.predict(196, 242)
    //println(predictRating)

    sc.stop()

  }
}
