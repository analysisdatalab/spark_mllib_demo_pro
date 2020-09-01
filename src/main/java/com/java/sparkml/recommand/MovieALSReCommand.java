package com.java.sparkml.recommand;

/**
 * @Author:
 * @Date: 2020-09-01 14:31
 * @Version: 1.0
 * @Modified By:
 * @Description:
 */

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

//导入回归模型评估器
//导入回归模型评估器

/**
 *
 * 需求：通过ALS算法在电影评分数据集上进行训练，获取推荐模型，给用户推荐商品
 * userId    productId    rating    timestamp
 * 196   	     242	      3	        881250949
 *
 * 数据（Rating）-->ALS算法进行训练--->矩阵分解模型（MatrixFactorizationModel）-->预测和推荐
 */
public class MovieALSReCommand {
    public static void main(String[] args) {
        //设置日志级别
        Logger.getLogger("org").setLevel(Level.WARN);
        SparkConf sparkConf = new SparkConf().setAppName("MovieALSReCommand").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        // todo 1，加载数据
        // 用户对电影的评分数据
        JavaRDD<Rating> ratingRDD = jsc.textFile("data/als/movielens/ml-100k/u.data")
                .mapPartitions(part -> {
                    List<Rating> list = new ArrayList<>();
                    while (part.hasNext()) {
                        /**
                         * case class Rating  (
                         * user: Int,  用户id
                         * product: Int, 产品id
                         * rating: Double 用户对产品的评分
                         * )
                         */
                        String line = part.next();
                        String[] fields = line.split("\t");
                        Rating rating = new Rating(Integer.parseInt(fields[0]), Integer.parseInt(fields[1]), Double.parseDouble(fields[2]));
                        list.add(rating);
                    }
                    return list.iterator();
                });

        // ratingRDD.take(8).forEach(System.out::println);

        //将数据集划分成训练集和测试集; 80%作为训练集，20%作为测试集
        JavaRDD<Rating>[] javaRDDS = ratingRDD.randomSplit(new double[]{0.8, 0.2});
        JavaRDD<Rating> trainRDD = javaRDDS[0];
        JavaRDD<Rating> testRDD = javaRDDS[1];
        // todo 2，ALS算法进行训练：将数据放入ALS算法中进行训练

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
        MatrixFactorizationModel matrixMode = ALS.train(trainRDD.rdd(), 10, 10);
        // 如果用户数据是隐式数据，则需要调用trainImplicit
        // ALS.trainImplicit(trainRDD, 10, 10)

        // 用户因子矩阵
        // matrixMode.userFeatures.take(5).foreach(t2 => println(s"userId ${t2._1}===> ${t2._2.mkString("[", ",", "]")}"))
        // 产品因子矩阵
        // matrixMode.productFeatures.take(5).foreach(t2 => println(s"productId ${t2._1}===> ${t2._2.mkString("[", ",", "]")}"))

        // todo 3 使用模型进行预测和推荐

        // 预测195 这个用户，对242这部电影的评分
        double predictRating = matrixMode.predict(196, 242);
        System.out.println("预测196用户对242电影的评分：" + predictRating);

        // todo  为195这个用户，推荐5部电影
        /** 推荐结果
         * Rating(195,1512,5.679301790216906)
         * Rating(195,1131,5.4702748596941415)
         * Rating(195,854,5.3102560597591975)
         * Rating(195,697,5.293633609827637)
         * Rating(195,1463,5.2095091969572085)
         */
        Rating[] rmd1 = matrixMode.recommendProducts(195, 5);
        for (Rating rating : rmd1) {
            System.out.println("给为195这个用户，推荐5部电影：" + rating);
        }

        // todo  为所有用户推荐5部电影
        //RDD[(Int, Array[Rating])]   (用户id,给这个用户推荐的结果)
        //val rmd2: RDD[(Int, Array[Rating])] =
        RDD<Tuple2<Object, Rating[]>> rmd2 = matrixMode.recommendProductsForUsers(5);
        List<Tuple2<Object, Rating[]>> rmd2List = rmd2.toJavaRDD().take(5);
        for (Tuple2<Object, Rating[]> t2 : rmd2List) {
            System.out.println("为所有用户推荐5部电影 userId=> " + t2._1);
            for (Rating rating : t2._2) {
                System.out.println(rating.toString());
            }
            System.out.println("=============================================");
        }
        // 查看5个用户，给他们分别推荐了哪5部电影
        // todo 为242这部电影，推荐5个用户
        // val rmd3: Array[Rating] =
        Rating[] rmd3 = matrixMode.recommendUsers(242, 5);
        for (Rating rating : rmd3) {
            System.out.println("为242这部电影，推荐5个用户: " + rating);
        }

        // todo 为所有电影，推荐5个用户
        // RDD[(Int, Array[Rating])]  (产品id，这个产品推荐的5个用户)
        RDD<Tuple2<Object, Rating[]>> rmd4 = matrixMode.recommendUsersForProducts(5);
        List<Tuple2<Object, Rating[]>> rmd4List = rmd4.toJavaRDD().take(5);
        for (Tuple2<Object, Rating[]> t2 : rmd4List) {
            System.out.println("为所有电影，推荐5个用户: productId: " + t2._1);
            for (Rating rating : t2._2) {
                System.out.println(rating);
            }
            System.out.println("==============================================");
        }

        // todo 使用回归模型评估器对矩阵分解模型进行评估
        // 在进行评估的时候使用测试集数据testRDD进行评估
        // testRDD  Rating==>((userId,productId),rating)
        // 用户对产品的实际评分
        JavaRDD<Tuple2<Tuple2<Integer, Integer>, Double>> actualRatingRDD
                = testRDD.map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating()));

        // 将用户和产品抽取出来封装成RDD，以便模型对每条数据都给出一个预测评分
        JavaRDD<Tuple2<Object, Object>> userAndProductRDD = actualRatingRDD.map(t -> {
            return new Tuple2(t._1._1, t._1._2);
        });

        // 使用模型预测出每条数据的预测评分
        RDD<Rating> predictRatingRDD = matrixMode.predict(JavaRDD.toRDD(userAndProductRDD));
        // 用户对产品的预测评分
        JavaRDD<Tuple2<Tuple2<Integer, Integer>, Double>> predictRatingRDD2
                = predictRatingRDD.toJavaRDD().map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating()));

        // ((user,product),(预测评分，实际评分))
        JavaPairRDD predictAndActualRatingRDD = JavaPairRDD.fromJavaRDD(predictRatingRDD2)
                .join(JavaPairRDD.fromJavaRDD(actualRatingRDD))
                // 取出(预测评分，实际评分)
                .mapToPair(t -> {
                    return new Tuple2(t._2._1, t._2._2);
                }
            );

        List list = predictAndActualRatingRDD.take(5);
        for (Object o : list) {
            System.out.println("预测评分, 实际评分: " + o.toString());
        }
        System.out.println("==============================================");

        // 创建回归模型评估器
        RegressionMetrics metrics = new RegressionMetrics(predictAndActualRatingRDD.rdd());

        System.out.println("RMSE 均方根误差:" + metrics.rootMeanSquaredError()); // RMSE 均方根误差:1.081484357011266
        System.out.println("MSE 均方误差:" + metrics.meanSquaredError());        // MSE 均方误差:1.1696084144600714
        System.out.println("MAE 平均绝对误差:" + metrics.meanAbsoluteError());    // MAE 平均绝对误差:0.814539727196231


        jsc.stop();
    }
}
