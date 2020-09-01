package com.sparkml.recommand

import org.apache.log4j.{Level, Logger}
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
object MovieALSReCommand {
  def main(args: Array[String]): Unit = {
    //设置日志级别
    Logger.getLogger("org").setLevel(Level.WARN)
    val sparkConf = new SparkConf().setAppName("MovieALSReCommand").setMaster("local")
    val sc = new SparkContext(sparkConf)

    //sc.setLogLevel("WARN")

    //todo 1，加载数据

    // 用户对电影的评分数据
    val ratingRDD = sc.textFile("data/als/movielens/ml-100k/u.data")
      .mapPartitions(part => {
        part.map(line => {
          /**
           * case class Rating  (
           * user: Int,  用户id
           * product: Int, 产品id
           * rating: Double 用户对产品的评分
           * )
           */

          val fields = line.split("\t")
          Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
        })
      })

    //ratingRDD.take(8).foreach(println)
    //return

    //将数据集划分成训练集和测试集; 80%作为训练集，20%作为测试集
    val Array(trainRDD, testRDD) = ratingRDD.randomSplit(Array(0.8, 0.2))

    //todo 2，ALS算法进行训练：将数据放入ALS算法中进行训练
    /**
     * def train(ratings: RDD[Rating], rank: Int, iterations: Int)  : MatrixFactorizationModel
     *    ratings 用户对产品的评价数据 trainRDD
     *    rank，特征数目个数 10
     *    iterations：迭代次数（交替次数），默认是10次
     *
     *
     * class MatrixFactorizationModel @Since("0.8.0") (
     *    val rank: Int, 特征数目个数 10
     *    val userFeatures: RDD[(Int, Array[Double])],用户因子矩阵 (用户id，用户特征)
     *    val productFeatures: RDD[(Int, Array[Double])]，产品因子矩阵（产品id，产品特征）
     * )
     *
     */
    // 矩阵分解模型：用户因子矩阵，产品因子矩阵
    val matrixMode: MatrixFactorizationModel = ALS.train(trainRDD, 10, 10)
    // 如果用户数据是隐式数据，则需要调用trainImplicit
    // ALS.trainImplicit(trainRDD, 10, 10)

    //用户因子矩阵
    //matrixMode.userFeatures.take(5).foreach(t2 => println(s"userId ${t2._1}===> ${t2._2.mkString("[", ",", "]")}"))
    //产品因子矩阵
    //matrixMode.productFeatures.take(5).foreach(t2 => println(s"productId ${t2._1}===> ${t2._2.mkString("[", ",", "]")}"))

    //todo 3 使用模型进行预测和推荐

    //预测195 这个用户，对242这部电影的评分
    val predictRating = matrixMode.predict(196, 242)
    println("预测196用户对242电影的评分：" + predictRating)

    //todo  为195这个用户，推荐5部电影
    /** 推荐结果
     * Rating(195,1512,5.679301790216906)
     * Rating(195,1131,5.4702748596941415)
     * Rating(195,854,5.3102560597591975)
     * Rating(195,697,5.293633609827637)
     * Rating(195,1463,5.2095091969572085)
     */
    val rmd1: Array[Rating] = matrixMode.recommendProducts(195, 5)
    rmd1.foreach(println)

    //todo  为所有用户推荐5部电影
    //RDD[(Int, Array[Rating])]   (用户id,给这个用户推荐的结果)
    val rmd2: RDD[(Int, Array[Rating])] = matrixMode.recommendProductsForUsers(5)
    //查看5个用户，给他们分别推荐了哪5部电影
    rmd2.take(5).foreach(t2 => {
      println(s"为所有用户推荐5部电影 userId=> ${t2._1}")
      t2._2.foreach(println)
      println("=============================================")
    })


    //todo 为242这部电影，推荐5个用户
    val rmd3: Array[Rating] = matrixMode.recommendUsers(242, 5)
    rmd3.foreach(println)


    //todo 为所有电影，推荐5个用户
    //RDD[(Int, Array[Rating])]  (产品id，这个产品推荐的5个用户)
    val rmd4: RDD[(Int, Array[Rating])] = matrixMode.recommendUsersForProducts(5)
    rmd4.take(5).foreach(t2 => {
      println(s"为所有电影，推荐5个用户 productId ${t2._1}")
      t2._2.foreach(println)
      println("==============================================")
    })


    //todo 使用回归模型评估器对矩阵分解模型进行评估
    //在进行评估的时候使用测试集数据testRDD进行评估
    //testRDD  Rating==>((userId,productId),rating)
    //用户对产品的实际评分
    val actualRatingRDD: RDD[((Int, Int), Double)] = testRDD.map(r => ((r.user, r.product), r.rating))

    //将用户和产品抽取出来封装成RDD，以便模型对每条数据都给出一个预测评分
    val userAndProductRDD = actualRatingRDD.map(_._1)
    //使用模型预测出每条数据的预测评分
    val predictRatingRDD: RDD[Rating] = matrixMode.predict(userAndProductRDD)
    //用户对产品的预测评分
    val predictRatingRDD2 = predictRatingRDD.map(r => ((r.user, r.product), r.rating))

    //((user,product),(预测评分，实际评分))
    val predictAndActualRatingRDD = predictRatingRDD2.join(actualRatingRDD)
      //取出(预测评分，实际评分)
      .map(_._2)
    // 打印：(预测评分，实际评分)
    predictAndActualRatingRDD.take(3).foreach(println)
    println("==============================================")

    //导入回归模型评估器
    import org.apache.spark.mllib.evaluation.RegressionMetrics
    //创建回归模型评估器
    val metrics = new RegressionMetrics(predictAndActualRatingRDD)

    println("RMSE 均方根误差:" + metrics.rootMeanSquaredError)  // RMSE 均方根误差:1.081484357011266
    println("MSE 均方误差:" + metrics.meanSquaredError)         // MSE 均方误差:1.1696084144600714
    println("MAE 平均绝对误差:" + metrics.meanAbsoluteError)     // MAE 平均绝对误差:0.814539727196231

    sc.stop()

  }
}
